import argparse
import sys
import tempfile
from pathlib import Path


DEFAULT_MODEL_FILENAME = "scunet_color_real_psnr.pth"
DEFAULT_SIZE = 256
SCUNET_SIZE_MULTIPLE = 64


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Export a Spandrel-compatible SCUNet checkpoint to FP32 and FP16 ONNX."
    )
    parser.add_argument(
        "model",
        nargs="?",
        type=Path,
        help="SCUNet .pth checkpoint (defaults to the standard model in Siril's scunet_models folder)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="directory for the ONNX files (defaults to the checkpoint directory)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=DEFAULT_SIZE,
        help=f"export input height; must be a positive multiple of {SCUNET_SIZE_MULTIPLE} (default: %(default)s)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=DEFAULT_SIZE,
        help=f"export input width; must be a positive multiple of {SCUNET_SIZE_MULTIPLE} (default: %(default)s)",
    )
    parser.add_argument(
        "--dynamic-shapes",
        action="store_true",
        help="export dynamic batch/height/width axes instead of the DirectML-friendly fixed dimensions",
    )
    parser.add_argument(
        "--precision",
        choices=("fp16", "fp32", "both"),
        default="fp16",
        help="ONNX model variant(s) to write (default: %(default)s)",
    )
    args = parser.parse_args(argv)
    for label, dimension in (("height", args.height), ("width", args.width)):
        if dimension <= 0 or dimension % SCUNET_SIZE_MULTIPLE:
            parser.error(
                f"--{label} must be a positive multiple of {SCUNET_SIZE_MULTIPLE}"
            )
    return args


def _prepare_runtime():
    try:
        import sirilpy as s
    except ImportError:
        s = None

    if s is not None:
        if not s.check_module_version(">=1.0.10"):
            raise RuntimeError("sirilpy >= 1.0.10 is required when using Siril.")
        s.ensure_installed("spandrel", "onnx", "onnxconverter-common")
        s.TorchHelper().ensure_torch()

    import onnx
    import torch
    from onnxconverter_common import float16
    from spandrel import ImageModelDescriptor, ModelLoader

    return s, onnx, torch, float16, ImageModelDescriptor, ModelLoader


def _default_model_path(sirilpy):
    if sirilpy is None:
        raise RuntimeError(
            "Pass a checkpoint path when running outside Siril; Siril is needed "
            "to locate its default model folder."
        )

    siril = sirilpy.SirilInterface()
    try:
        siril.connect()
    except Exception as error:
        raise RuntimeError(
            "Could not connect to Siril. Open Siril or pass a checkpoint path."
        ) from error

    models_dir = Path(siril.get_siril_userdatadir()) / "scunet_models"
    return models_dir / DEFAULT_MODEL_FILENAME


def _export(args):
    sirilpy, onnx, torch, float16, image_model_descriptor, model_loader = (
        _prepare_runtime()
    )
    model_path = args.model if args.model is not None else _default_model_path(sirilpy)
    model_path = model_path.expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"SCUNet checkpoint not found: {model_path}")

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = model_path.parent
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    descriptor = model_loader().load_from_file(str(model_path))
    if not isinstance(descriptor, image_model_descriptor):
        raise TypeError(f"Expected an image model checkpoint: {model_path}")
    architecture_name = type(descriptor.model).__name__
    if "SCUNet" not in architecture_name:
        raise ValueError(
            f"Expected a SCUNet checkpoint, but detected {architecture_name}."
        )
    if descriptor.input_channels not in (1, 3):
        raise ValueError(
            f"Expected a one- or three-channel SCUNet model, got "
            f"{descriptor.input_channels} input channels."
        )

    model = descriptor.model.eval()
    dummy_input = torch.randn(
        1, descriptor.input_channels, args.height, args.width, dtype=torch.float32
    )
    base_name = model_path.stem
    fp32_path = output_dir / f"{base_name}_fp32.onnx"
    fp16_path = output_dir / f"{base_name}_fp16.onnx"
    if args.precision == "fp16":
        temp_file = tempfile.NamedTemporaryFile(
            prefix=f"{base_name}_export_",
            suffix=".onnx",
            dir=output_dir,
            delete=False,
        )
        export_path = Path(temp_file.name)
        temp_file.close()
    else:
        export_path = fp32_path

    export_options = {
        "export_params": True,
        "opset_version": 17,
        "do_constant_folding": True,
        "input_names": ["input"],
        "output_names": ["output"],
        "dynamo": False,
    }
    if args.dynamic_shapes:
        export_options["dynamic_axes"] = {
            "input": {0: "batch_size", 2: "height", 3: "width"},
            "output": {0: "batch_size", 2: "height", 3: "width"},
        }

    should_export_fp32 = args.precision in ("fp32", "both")
    should_export_fp16 = args.precision in ("fp16", "both")
    try:
        with torch.no_grad():
            torch.onnx.export(model, dummy_input, str(export_path), **export_options)

        model_fp32 = onnx.load(str(export_path))
        onnx.checker.check_model(model_fp32)
        if should_export_fp16:
            model_fp16 = float16.convert_float_to_float16(
                model_fp32, keep_io_types=True
            )
            onnx.checker.check_model(model_fp16)
            onnx.save(model_fp16, str(fp16_path))
    finally:
        if export_path != fp32_path and export_path.exists():
            export_path.unlink()

    print(f"Exported {descriptor.architecture.name} from {model_path}")
    if should_export_fp32:
        print(f"FP32: {fp32_path}")
    if should_export_fp16:
        print(f"FP16: {fp16_path}")


def main(argv=None):
    args = _parse_args(argv)
    try:
        _export(args)
    except Exception as error:
        print(f"ONNX export failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
