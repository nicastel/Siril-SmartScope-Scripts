import argparse
import math
import sys
from pathlib import Path


DEFAULT_MODEL_FILENAME = "scunet_color_real_psnr_fp16.onnx"
FP32_MODEL_FILENAME = "scunet_color_real_psnr_fp32.onnx"
TILE_PAD = 144
SCUNET_SIZE_MULTIPLE = 64
DEFAULT_TILE_SIZE = 128
GHS_BG_TARGET = 0.25
GHS_SHADOW_K = 0.0
GHS_D_SCALE = 1.5


def _tile_size(value):
    try:
        size = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("tile size must be an integer") from error
    if size not in (64, 128, 256, 384, 512):
        raise argparse.ArgumentTypeError(
            "tile size must be one of 64, 128, 256, 384, or 512"
        )
    return size


def _strength(value):
    try:
        strength = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("strength must be a number from 0 to 1") from error
    if not math.isfinite(strength) or not 0.0 <= strength <= 1.0:
        raise argparse.ArgumentTypeError("strength must be a number from 0 to 1")
    return strength


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Denoise Siril's loaded image with an exported SCUNet ONNX model."
    )
    parser.add_argument(
        "--model",
        type=Path,
        help="path to an exported SCUNet ONNX model (defaults to the standard FP16 model)",
    )
    parser.add_argument(
        "--precision",
        choices=("fp16", "fp32"),
        default="fp16",
        help="default exported model precision when --model is omitted (default: %(default)s)",
    )
    parser.add_argument(
        "--strength",
        type=_strength,
        default=1.0,
        help="blend denoised pixels with the input (0 keeps input, 1 uses full denoise; default: %(default)s)",
    )
    parser.add_argument(
        "--tile-size",
        type=_tile_size,
        default=DEFAULT_TILE_SIZE,
        help="tile core size in pixels (default: %(default)s)",
    )
    parser.add_argument(
        "--pre-stretch",
        action="store_true",
        help="apply the original script's GHS prestretch/destretch for linear images",
    )
    parser.add_argument(
        "--provider",
        choices=("directml", "cpu"),
        default="directml",
        help="ONNX Runtime execution provider (default: %(default)s)",
    )
    return parser.parse_args(argv)


def _sigma_clip_stats(np, flat, n_iter=5):
    data = flat[flat > 1e-10].copy()
    if len(data) < 100:
        data = flat.copy()
    for _ in range(n_iter):
        median = np.median(data)
        mad = float(np.median(np.abs(data - median)))
        sigma = mad * 1.4826
        if sigma < 1e-12:
            break
        mask = np.abs(data - median) < 3.0 * sigma
        if mask.sum() < 50:
            break
        data = data[mask]
    median = float(np.median(data))
    mad = float(np.median(np.abs(data - median)))
    return median, max(mad * 1.4826, 1e-12)


def _compute_ghs_params(np, hwc):
    arr = np.asarray(hwc, dtype=np.float64)
    if arr.ndim == 2:
        arr = arr[:, :, np.newaxis]
    hp_norm = float(np.nanmax(arr))
    if hp_norm < 1e-10:
        hp_norm = 1.0
    arr_norm = arr / hp_norm
    if arr_norm.shape[2] == 3:
        luminance = (
            arr_norm[:, :, 0] * 0.2126
            + arr_norm[:, :, 1] * 0.7152
            + arr_norm[:, :, 2] * 0.0722
        ).ravel()
    else:
        luminance = arr_norm[:, :, 0].ravel()
    sky_med, _ = _sigma_clip_stats(np, luminance)
    lp = float(np.clip(GHS_SHADOW_K * sky_med, 0.0, sky_med * 0.9999))
    xp = sky_med - lp
    sp_base = max(xp, 1e-6)
    bright_sky = xp > GHS_BG_TARGET

    def ghs_at_base(d_value):
        sp_search = 0.999 if bright_sky else sp_base
        a_value = np.arcsinh(d_value * sp_search)
        b_value = np.arcsinh(d_value * (1.0 - sp_search)) + a_value
        if b_value < 1e-10:
            return sp_search
        return float((np.arcsinh(d_value * (xp - sp_search)) + a_value) / b_value)

    d_lo, d_hi = 1.0, 100000000.0
    for _ in range(40):
        d_mid = np.sqrt(d_lo * d_hi)
        if bright_sky:
            if ghs_at_base(d_mid) > GHS_BG_TARGET:
                d_lo = d_mid
            else:
                d_hi = d_mid
        elif ghs_at_base(d_mid) < GHS_BG_TARGET:
            d_lo = d_mid
        else:
            d_hi = d_mid
    d_value = float(np.sqrt(d_lo * d_hi) * GHS_D_SCALE)
    y_max = float(np.arcsinh(d_value * xp) / max(np.arcsinh(d_value), 1e-10))
    sp_feasible = y_max >= GHS_BG_TARGET

    def y_at_sp(test_sp):
        a_value = np.arcsinh(d_value * test_sp)
        b_value = np.arcsinh(d_value * (1.0 - test_sp)) + a_value
        if b_value < 1e-10:
            return 0.0
        return (np.arcsinh(d_value * (xp - test_sp)) + a_value) / b_value

    if sp_feasible:
        sp_lo, sp_hi = 1e-6, 0.999
        for _ in range(40):
            sp_mid = (sp_lo + sp_hi) / 2.0
            if y_at_sp(sp_mid) > GHS_BG_TARGET:
                sp_lo = sp_mid
            else:
                sp_hi = sp_mid
        sp = float((sp_lo + sp_hi) / 2.0)
    else:
        sp = sp_base

    a_value = float(np.arcsinh(d_value * sp))
    b_value = float(np.arcsinh(d_value * (1.0 - sp)) + a_value)
    return {
        "lp": lp,
        "sp": sp,
        "d": d_value,
        "a": a_value,
        "b": b_value,
        "hp_norm": hp_norm,
    }


def _apply_stretch(np, hwc, params):
    arr = np.asarray(hwc, dtype=np.float64)
    return (
        np.arcsinh(params["d"] * (arr / params["hp_norm"] - params["lp"] - params["sp"]))
        + params["a"]
    ) / params["b"]


def _apply_destretch(np, hwc, params):
    arr = np.asarray(hwc, dtype=np.float64)
    xp = np.sinh(np.clip(arr * params["b"] - params["a"], -500.0, 500.0))
    xp = xp / params["d"] + params["sp"]
    return np.clip((xp + params["lp"]) * params["hp_norm"], 0.0, params["hp_norm"])


def _tile_starts(length, tile_size):
    stride = tile_size * 3 // 4
    starts = list(range(0, max(1, length - tile_size), stride))
    if not starts or starts[-1] + tile_size < length:
        starts.append(max(0, length - tile_size))
    return starts


def _tile_weight(np, height, width):
    x = np.linspace(0.0, 1.0, width, dtype=np.float32)
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)
    wx = np.maximum(np.minimum(x, 1.0 - x) * 2.0, 0.1)
    wy = np.maximum(np.minimum(y, 1.0 - y) * 2.0, 0.1)
    return wy[:, np.newaxis] * wx[np.newaxis, :]


def _run_tiled(
    np,
    session,
    input_name,
    output_name,
    input_dtype,
    image,
    tile_size,
    pre_stretch,
    progress_callback=None,
    model_height=None,
    model_width=None,
):
    channels, height, width = image.shape
    if model_height is None or model_width is None:
        pad_y = TILE_PAD
        pad_x = TILE_PAD
        fixed_height = None
        fixed_width = None
    else:
        if tile_size > model_height or tile_size > model_width:
            raise ValueError(
                f"Tile size {tile_size} exceeds the fixed ONNX input size "
                f"{model_height}x{model_width}."
            )
        pad_y = (model_height - tile_size) // 2
        pad_x = (model_width - tile_size) // 2
        fixed_height = model_height
        fixed_width = model_width

    output_sum = np.zeros((channels, height, width), dtype=np.float32)
    output_weight = np.zeros((height, width), dtype=np.float32)
    starts_y = _tile_starts(height, tile_size)
    starts_x = _tile_starts(width, tile_size)
    total_tiles = len(starts_y) * len(starts_x)
    ghs_params = _compute_ghs_params(np, image.transpose(1, 2, 0)) if pre_stretch else None
    tile_number = 0

    for y_start in starts_y:
        for x_start in starts_x:
            tile_number += 1
            y_end = min(y_start + tile_size, height)
            x_end = min(x_start + tile_size, width)
            input_y_start = max(y_start - pad_y, 0)
            input_x_start = max(x_start - pad_x, 0)
            input_y_end = min(y_end + pad_y, height)
            input_x_end = min(x_end + pad_x, width)
            tile = image[:, input_y_start:input_y_end, input_x_start:input_x_end]
            if fixed_height is None:
                pad_height = (-tile.shape[1]) % SCUNET_SIZE_MULTIPLE
                pad_width = (-tile.shape[2]) % SCUNET_SIZE_MULTIPLE
            else:
                pad_height = fixed_height - tile.shape[1]
                pad_width = fixed_width - tile.shape[2]
                if pad_height < 0 or pad_width < 0:
                    raise ValueError("Tile context exceeds the fixed ONNX input size.")
            if pre_stretch:
                tile_hwc = _apply_stretch(np, tile.transpose(1, 2, 0), ghs_params)
                tile = tile_hwc.transpose(2, 0, 1).astype(np.float32)
            if pad_height or pad_width:
                pad_mode = (
                    "reflect"
                    if tile.shape[1] > 1
                    and tile.shape[2] > 1
                    and pad_height < tile.shape[1]
                    and pad_width < tile.shape[2]
                    else "edge"
                )
                tile = np.pad(
                    tile,
                    ((0, 0), (0, pad_height), (0, pad_width)),
                    mode=pad_mode,
                )
            model_input = np.ascontiguousarray(
                tile[np.newaxis].astype(input_dtype, copy=False)
            )
            prediction = session.run([output_name], {input_name: model_input})[0]
            if prediction.ndim != 4 or prediction.shape[0] != 1:
                raise ValueError(f"Unexpected ONNX output shape: {prediction.shape}")
            if prediction.shape[2:] != model_input.shape[2:]:
                raise ValueError(
                    "SCUNet ONNX output height and width must match the input tile."
                )
            if prediction.shape[1] != channels:
                raise ValueError(
                    f"Model returned {prediction.shape[1]} channels for a "
                    f"{channels}-channel input."
                )

            core_y_start = y_start - input_y_start
            core_x_start = x_start - input_x_start
            core_height = y_end - y_start
            core_width = x_end - x_start
            core = prediction[
                0,
                :,
                core_y_start : core_y_start + core_height,
                core_x_start : core_x_start + core_width,
            ].astype(np.float32)
            if core.shape[1:] != (core_height, core_width):
                raise ValueError(
                    "ONNX output spatial dimensions do not match the input tile."
                )
            if pre_stretch:
                core = _apply_destretch(np, core.transpose(1, 2, 0), ghs_params)
                core = core.transpose(2, 0, 1).astype(np.float32)

            weight = _tile_weight(np, core_height, core_width)
            output_sum[:, y_start:y_end, x_start:x_end] += core * weight[np.newaxis]
            output_weight[y_start:y_end, x_start:x_end] += weight
            if progress_callback is not None:
                progress_callback(tile_number, total_tiles)

    return output_sum / np.maximum(output_weight[np.newaxis], 1e-8)


def _process_image(
    np, session, image_data, args, input_meta, output_meta, progress_callback=None
):
    original_dtype = image_data.dtype
    if original_dtype == np.uint8:
        divisor = 255.0
        source = image_data.astype(np.float32) / divisor
    elif original_dtype == np.uint16:
        actual_max = int(image_data.max())
        divisor = 255.0 if actual_max <= 255 else 65535.0
        source = image_data.astype(np.float32) / divisor
    elif np.issubdtype(original_dtype, np.floating):
        divisor = None
        source = image_data.astype(np.float32)
    else:
        raise TypeError(f"Unsupported image pixel type: {original_dtype}")

    if not np.all(np.isfinite(source)):
        raise ValueError("The loaded image contains non-finite pixel values.")
    is_mono = source.ndim == 2 or (source.ndim == 3 and source.shape[0] == 1)
    if source.ndim == 2:
        source_chw = source[np.newaxis]
    elif source.ndim == 3 and source.shape[0] in (1, 3):
        source_chw = source
    else:
        raise ValueError(
            "Expected a monochrome or RGB Siril image in channel-first layout."
        )

    model_shape = input_meta.shape
    if len(model_shape) != 4:
        raise ValueError(f"Expected a rank-four ONNX input, got {model_shape}.")
    model_channels = model_shape[1]
    if not isinstance(model_channels, int) or model_channels not in (1, 3):
        raise ValueError(f"Unsupported ONNX input channel dimension: {model_channels}")
    if is_mono and model_channels == 3:
        model_image = np.repeat(source_chw[:1], 3, axis=0)
    elif is_mono and model_channels == 1:
        model_image = source_chw[:1]
    elif not is_mono and model_channels == 3:
        model_image = source_chw
    else:
        raise ValueError(
            "This monochrome ONNX model cannot process an RGB Siril image."
        )

    static_height, static_width = model_shape[2:]
    has_static_dimensions = isinstance(static_height, int) and isinstance(
        static_width, int
    )
    if isinstance(static_height, int) != isinstance(static_width, int):
        raise ValueError("ONNX input height and width must both be fixed or dynamic.")
    if args.provider == "directml" and not has_static_dimensions:
        raise ValueError(
            "DirectML requires fixed ONNX input dimensions; the selected model "
            f"has dynamic dimensions {model_shape[2:]}. Run "
            "`exportScunetToONNX.py` again without --dynamic-shapes to export "
            "fixed-shape models."
        )
    output_shape = output_meta.shape
    if len(output_shape) != 4:
        raise ValueError(f"Expected a rank-four ONNX output, got {output_shape}.")
    if isinstance(output_shape[1], int) and output_shape[1] != model_channels:
        raise ValueError("ONNX input and output channel counts must match.")
    if has_static_dimensions and output_shape[2:] != model_shape[2:]:
        raise ValueError("Static ONNX input and output spatial dimensions must match.")

    input_dtype = np.float16 if input_meta.type == "tensor(float16)" else np.float32
    if input_meta.type not in ("tensor(float)", "tensor(float16)"):
        raise ValueError(f"Unsupported ONNX input type: {input_meta.type}")

    denoised = _run_tiled(
        np,
        session,
        input_meta.name,
        output_meta.name,
        input_dtype,
        model_image,
        args.tile_size,
        args.pre_stretch,
        progress_callback,
        static_height if has_static_dimensions else None,
        static_width if has_static_dimensions else None,
    )
    if is_mono:
        denoised = denoised[0]
        original = source_chw[0]
    else:
        original = source_chw
    result = denoised * args.strength + original * (1.0 - args.strength)
    if original_dtype == np.uint8 or original_dtype == np.uint16:
        result = np.clip(result, 0.0, 1.0) * divisor
    return result.astype(original_dtype)


def _create_inference_session(ort, model_path, provider):
    if provider == "cpu":
        return ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )

    if sys.platform != "win32":
        raise RuntimeError("The DirectML execution provider is available only on Windows.")
    if "DmlExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError(
            "ONNX Runtime did not load DmlExecutionProvider. Install "
            "`onnxruntime-directml` and remove the conflicting CPU-only "
            "`onnxruntime` package from this Python environment."
        )

    options = ort.SessionOptions()
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.enable_mem_pattern = False
    session = ort.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=["DmlExecutionProvider"],
    )
    if "DmlExecutionProvider" not in session.get_providers():
        raise RuntimeError(
            "ONNX Runtime failed to activate DmlExecutionProvider; no CPU "
            "fallback was selected."
        )
    return session


def _default_model_path(user_data_dir, precision="fp16"):
    model_filename = (
        DEFAULT_MODEL_FILENAME if precision == "fp16" else FP32_MODEL_FILENAME
    )
    return Path(user_data_dir) / "scunet_models" / model_filename


def _run(args, sirilpy, np, ort):
    siril = sirilpy.SirilInterface()
    try:
        siril.connect()
    except Exception as error:
        raise RuntimeError(
            "Could not connect to Siril. Start Siril and load an image first."
        ) from error

    try:
        if not siril.is_image_loaded():
            raise RuntimeError("No image is currently loaded in Siril.")

        model_path = args.model
        if model_path is None:
            model_path = _default_model_path(
                siril.get_siril_userdatadir(), args.precision
            )
        model_path = model_path.expanduser().resolve()
        if not model_path.is_file():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")

        session = _create_inference_session(ort, model_path, args.provider)
        inputs = session.get_inputs()
        outputs = session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError("Expected an ONNX model with exactly one input and output.")
        input_meta, output_meta = inputs[0], outputs[0]

        siril.log(f"SCUNet ONNX: {model_path}", sirilpy.LogColor.BLUE)
        siril.log(
            f"ONNX Runtime providers: {', '.join(session.get_providers())}",
            sirilpy.LogColor.BLUE,
        )
        image = siril.get_image()
        original_max = int(image.data.max()) if image.data.dtype == np.uint16 else None

        def progress_callback(done, total):
            siril.update_progress("SCUNet ONNX", done / total)

        processed = _process_image(
            np,
            session,
            image.data,
            args,
            input_meta,
            output_meta,
            progress_callback,
        )
        siril.undo_save_state(
            f"SCUNet ONNX - {model_path.stem} | Strength: {args.strength:.2f}"
        )
        with siril.image_lock():
            siril.set_image_pixeldata(processed)
            if processed.dtype == np.uint8:
                siril.set_siril_slider_lohi(0, 255)
            elif processed.dtype == np.uint16:
                slider_hi = 255 if original_max <= 255 else 65535
                siril.set_siril_slider_lohi(0, slider_hi)
            elif np.issubdtype(processed.dtype, np.floating):
                siril.set_siril_slider_lohi(0.0, 1.0)
        siril.log(
            f"SCUNet ONNX denoise complete: {model_path.name}",
            sirilpy.LogColor.GREEN,
        )
    except Exception as error:
        siril.log(
            f"SCUNet ONNX inference failed: {error}",
            sirilpy.LogColor.RED,
        )
        raise
    finally:
        try:
            siril.reset_progress()
        except Exception:
            pass
        try:
            siril.disconnect()
        except Exception:
            pass


def main(argv=None):
    args = _parse_args(argv)
    try:
        import sirilpy as s

        if not s.check_module_version(">=1.0.10"):
            raise RuntimeError("sirilpy >= 1.0.10 is required.")
        if args.provider == "directml" and sys.platform != "win32":
            raise RuntimeError("The DirectML execution provider is available only on Windows.")
        runtime_package = (
            "onnxruntime-directml" if args.provider == "directml" else "onnxruntime"
        )
        s.ensure_installed("numpy", runtime_package)

        import numpy as np
        import onnxruntime as ort

        _run(args, s, np, ort)
    except Exception as error:
        print(f"SCUNet ONNX inference failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
