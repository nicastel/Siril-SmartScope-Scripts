#
# ***********************************************
#
# Original Author:  Nicolas CASTEL <nic.castel (at) gmail.com>
# Copyright (C) 2023 - Nicolas CASTEL
#
# Version 2.x Author: Carlo Mollicone - AstroBOH
# Copyright (C) 2025 - Carlo Mollicone
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Contact:
#   https://www.astroboh.it
#   https://www.facebook.com/carlo.mollicone.9
#
# ***********************************************
#
# NOTICE ON DERIVATIVE WORKS (as permitted by GPL-3.0):
#
# The contributions introduced in Version 2.x by Carlo Mollicone
# are original creative works of the author.
#
# Redistribution and modification are permitted under the terms
# of the GNU General Public License v3.0 or later, provided that:
#   1. All copyright notices and author attributions are preserved.
#   2. Any modifications to the code are clearly indicated.
#   3. Redistribution must comply with GPL-3.0 §4.
#   4. Derivative works are distributed under the same GPL-3.0-or-later license.
#
# This program is distributed WITHOUT ANY WARRANTY.
# See <https://www.gnu.org/licenses/gpl-3.0.html> for full terms.
#
# ***********************************************
# 
#
# Description:
# ------------------------------------------------------------------------------
# Project: Python siril script to run SCUNet denoiser via spandrel
#          using model from https://github.com/cszn/SCUNet
#          and https://ubersmooth.com/
#
#          GHS Stretch Engine based on the mathematical formulation by
#          Nightingale & Rowbottom (2021) — https://ghsastro.co.uk
#
#          Now supports:
#           - GUI Framework: PyQt6
#           - Single Image and Sequence Processing
#           - Model Management: models are stored in a dedicated folder
#           - Mono image support
# ------------------------------------------------------------------------------
#
# Version History
# 1.0.3 - Original release by Nicolas CASTEL
# 2.0.0 - Ported to PyQt6 by Carlo Mollicone - AstroBOH
#       - Added Sequence Processing support
#       - Added Mono image support (via RGB replication), useful for Solar/Planetary workflows
#       - Improved Model Management: models are now stored in a dedicated 'scunet_models'
#         folder to keep the working directory clean. Added download progress bar.
#       - Performance:
#           Added Auto-Tile tuning to detect max safe tile size and prevent VRAM crashes.
#           Quality: Implemented Weighted Soft Blending to completely eliminate tile grid seams.
#       - Fix tile processing edge cases to avoid artifacts at image borders.
#       - Fix tile dimension mismatches to ensure accurate processing.
#       - Added support for Intel Arc / XPU devices via PyTorch
#       - Added instructions
#       - Added ROI preview
# 2.0.1 - Fixed GPU acceleration detection order
#         Optimized package installation sequence to prevent CPU fallback
#       - Added "Cancel" functionality and fixed UnboundLocalError in tile processing
# 2.0.2 - Added explicit 'torchvision' dependency check to ensure binary compatibility
#         and prevent environment corruption when switching Torch versions (interoperability fix).
# 2.0.3 - Added PreStretch for linear (unstretched) image support.
#         New checkbox "Apply PreStretch" in the Parameters section of the GUI.
# 2.0.4 - Fix : Cannot set version_counter for inference tensor
# 2.0.5 - Fix : progess_callback in preview worker, now it updates the progress bar correctly during preview processing.
#         Better error handling in image_inference_tensor to catch DirectML-related runtime errors and provide user-friendly messages about GPU compatibility issues.
#         GUI improvements: model selection layout with two columns (Non-Linear only vs Linear & Non-Linear), added subtitles for clarity, and a vertical separator for better visual organization.
# 2.0.6 - Windows inference uses ONNX Runtime DirectML; PyTorch DirectML is no longer required.
#         Automatically select the OpenVINO GPU plugin when a compatible Intel GPU is detected.
#

from __future__ import annotations

VERSION = "2.0.6"

import importlib.util
import sys
import os
import urllib.request
import ssl
import math
import zipfile
import traceback
import base64
import subprocess
import time

# Attempt to import sirilpy. If not running inside Siril, the import will fail.
try:
    import sirilpy as s

    # Check the module version
    if not s.check_module_version('>=1.0.10'):
        print("Error: requires sirilpy module >= 1.0.10")
        sys.exit(1)

    from sirilpy import SirilError

    s.ensure_installed("PyQt6")

except ImportError as e:
    print(f"Warning: module imports failed: {e}")
    sys.exit(1)

_USE_ONNX_DIRECTML = sys.platform == "win32"
_onnx_tools = None
_OPENVINO_GPU_DEVICE = None
if _USE_ONNX_DIRECTML:
    onnx_helper_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "SCUNet_ONNX_Denoise.py"
    )
    onnx_spec = importlib.util.spec_from_file_location(
        "scunet_onnx_denoise", onnx_helper_path
    )
    if onnx_spec is None or onnx_spec.loader is None:
        raise RuntimeError(f"Could not load ONNX inference helpers: {onnx_helper_path}")
    _onnx_tools = importlib.util.module_from_spec(onnx_spec)
    onnx_spec.loader.exec_module(_onnx_tools)
    _onnx_tools._ensure_runtime_dependencies(s, "directml")
    import numpy as np
    import onnxruntime as ort
else:
    s.ensure_installed("spandrel", "torchvision")
    s.TorchHelper().ensure_torch()
    import torch
    import numpy as np
    from spandrel import ImageModelDescriptor, ModelLoader

# --- PyQt6 Imports ---
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QComboBox, QCheckBox, QMessageBox, QGroupBox,
    QProgressBar, QDoubleSpinBox, QLineEdit, QFormLayout,
    QRadioButton, QSlider, QFrame, QStyle, QSizePolicy, QButtonGroup
)
from PyQt6.QtGui import QCloseEvent, QIcon, QPixmap, QImage
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject, QTimer



_GHS_BG_TARGET = 0.25
_GHS_SHADOW_K = 0.0
_GHS_D_SCALE = 1.5
_GHS_SIGMA_ITER = 5

def _sigma_clip_stats(flat: np.ndarray, n_iter: int=_GHS_SIGMA_ITER) -> tuple:
    data = flat[flat > 1e-10].copy()
    if len(data) < 100:
        data = flat.copy()
    for _ in range(n_iter):
        med = np.median(data)
        mad = float(np.median(np.abs(data - med)))
        sigma = mad * 1.4826
        if sigma < 1e-12:
            break
        mask = np.abs(data - med) < 3.0 * sigma
        if mask.sum() < 50:
            break
        data = data[mask]
    med = float(np.median(data))
    mad = float(np.median(np.abs(data - med)))
    sigma = max(mad * 1.4826, 1e-12)
    return (med, sigma)

def _compute_ghs_params(hwc: np.ndarray, bg_target: float=_GHS_BG_TARGET, shadow_k: float=_GHS_SHADOW_K, d_scale: float=_GHS_D_SCALE) -> tuple:
    arr = np.asarray(hwc, dtype=np.float64)
    if arr.ndim == 2:
        arr = arr[:, :, np.newaxis]
    hp_norm = float(np.nanmax(arr))
    if hp_norm < 1e-10:
        hp_norm = 1.0
    arr_norm = arr / hp_norm
    if arr_norm.shape[2] == 3:
        luminance = (arr_norm[:, :, 0] * 0.2126 + arr_norm[:, :, 1] * 0.7152 + arr_norm[:, :, 2] * 0.0722).ravel()
    else:
        luminance = arr_norm[:, :, 0].ravel()
    sky_med, sky_sigma = _sigma_clip_stats(luminance)
    LP = float(np.clip(shadow_k * sky_med, 0.0, sky_med * 0.9999))
    xp = sky_med - LP
    SP_base = max(xp, 1e-06)
    _bright_sky = xp > bg_target

    def _ghs_at_base(D_val: float) -> float:
        sp_search = 0.999 if _bright_sky else SP_base
        A_t = np.arcsinh(D_val * sp_search)
        B_t = np.arcsinh(D_val * (1.0 - sp_search)) + A_t
        if B_t < 1e-10:
            return sp_search
        return float((np.arcsinh(D_val * (xp - sp_search)) + A_t) / B_t)
    D_lo, D_hi = (1.0, 100000000.0)
    for _ in range(40):
        D_mid = np.sqrt(D_lo * D_hi)
        if _bright_sky:
            if _ghs_at_base(D_mid) > bg_target:
                D_lo = D_mid
            else:
                D_hi = D_mid
        elif _ghs_at_base(D_mid) < bg_target:
            D_lo = D_mid
        else:
            D_hi = D_mid
    D_base = float(np.sqrt(D_lo * D_hi))
    D = D_base * d_scale
    _y_max_approx = float(np.arcsinh(D * xp) / max(np.arcsinh(D), 1e-10))
    _sp_feasible = _y_max_approx >= bg_target

    def y_at_sp(test_sp: float) -> float:
        A_t = np.arcsinh(D * test_sp)
        B_t = np.arcsinh(D * (1.0 - test_sp)) + A_t
        if B_t < 1e-10:
            return 0.0
        return (np.arcsinh(D * (xp - test_sp)) + A_t) / B_t
    if _sp_feasible:
        SP_lo, SP_hi = (1e-06, 0.999)
        for _ in range(40):
            SP_mid = (SP_lo + SP_hi) / 2.0
            if y_at_sp(SP_mid) > bg_target:
                SP_lo = SP_mid
            else:
                SP_hi = SP_mid
        SP = float((SP_lo + SP_hi) / 2.0)
    else:
        SP = SP_base
        import warnings
        warnings.warn(f'[GHS v5] d_scale={d_scale:.2f} troppo basso per ri-ancorare SP. Fallback a SP=xp={SP_base:.6f}. y_max≈{_y_max_approx:.4f} < bg_target={bg_target:.4f}. Aumentare d_scale o ridurre bg_target.', RuntimeWarning, stacklevel=2)
    A = float(np.arcsinh(D * SP))
    B = float(np.arcsinh(D * (1.0 - SP)) + A)
    master_param = {'LP': LP, 'SP': SP, 'D': D, 'A': A, 'B': B, 'hp_norm': hp_norm}
    params = [master_param] * arr.shape[2]
    return (params, hp_norm)

def _apply_stretch(hwc: np.ndarray, params: list) -> np.ndarray:
    arr = np.asarray(hwc, dtype=np.float64)
    is_2d = arr.ndim == 2
    if is_2d:
        arr = arr[:, :, np.newaxis]
    hp_norm = params[0].get('hp_norm', 1.0)
    arr_norm = arr / hp_norm
    out = np.empty_like(arr_norm)
    for c, p in enumerate(params):
        LP, SP, D, A, B = (p['LP'], p['SP'], p['D'], p['A'], p['B'])
        xp = arr_norm[:, :, c] - LP
        out[:, :, c] = (np.arcsinh(D * (xp - SP)) + A) / B
    return out[:, :, 0] if is_2d else out

def _apply_destretch(hwc: np.ndarray, params: list) -> np.ndarray:
    arr = np.asarray(hwc, dtype=np.float64)
    is_2d = arr.ndim == 2
    if is_2d:
        arr = arr[:, :, np.newaxis]
    hp_norm = params[0].get('hp_norm', 1.0)
    out = np.empty_like(arr)
    for c, p in enumerate(params):
        LP, SP, D, A, B = (p['LP'], p['SP'], p['D'], p['A'], p['B'])
        y = arr[:, :, c]
        xp = np.sinh(np.clip(y * B - A, -500.0, 500.0)) / D + SP
        x_norm = xp + LP
        out[:, :, c] = np.clip(x_norm * hp_norm, 0.0, hp_norm)
    return out[:, :, 0] if is_2d else out
_MTF_BG_TARGET = 0.2
_MTF_SHADOW_K = 2.8
_MTF_SIGMA_ITER = 5

def _mtf_sky_stats(flat: np.ndarray) -> tuple:
    data = flat[flat > 1e-10].copy()
    if len(data) < 100:
        data = flat.copy()
    for _ in range(_MTF_SIGMA_ITER):
        med = float(np.median(data))
        mad = float(np.median(np.abs(data - med)))
        sigma = max(mad * 1.4826, 1e-12)
        mask = np.abs(data - med) < 3.0 * sigma
        if mask.sum() < 50:
            break
        data = data[mask]
    med = float(np.median(data))
    mad = float(np.median(np.abs(data - med)))
    sigma = max(mad * 1.4826, 1e-12)
    return (med, sigma)

def _mtf_solve_mid(x: float, target: float) -> float:
    if x <= 0.0 or x >= 1.0 or target <= 0.0 or (target >= 1.0):
        return 0.25
    num = (target - 1.0) * x
    den = (2.0 * target - 1.0) * x - target
    if abs(den) < 1e-10:
        return 0.25
    return float(np.clip(num / den, 1e-06, 1.0 - 1e-06))

def _mtf_curve(x: np.ndarray, lo: float, mid: float, hi: float) -> np.ndarray:
    span = max(hi - lo, 1e-10)
    xp = np.clip((x - lo) / span, 0.0, 1.0)
    with np.errstate(divide='ignore', invalid='ignore'):
        y = np.where(xp <= 0.0, 0.0, np.where(xp >= 1.0, 1.0, (mid - 1.0) * xp / ((2.0 * mid - 1.0) * xp - mid)))
    return np.clip(y, 0.0, 1.0)

def _compute_mtf_display_params(hwc: np.ndarray) -> tuple:
    arr = np.asarray(hwc, dtype=np.float64)
    if arr.ndim == 2:
        lum = arr.ravel()
    elif arr.shape[2] >= 3:
        lum = (arr[:, :, 0] * 0.2126 + arr[:, :, 1] * 0.7152 + arr[:, :, 2] * 0.0722).ravel()
    else:
        lum = arr[:, :, 0].ravel()
    sky_med, sky_sigma = _mtf_sky_stats(lum)
    lo = float(np.clip(sky_med - _MTF_SHADOW_K * sky_sigma, 0.0, 0.9999))
    x_sky = float(np.clip((sky_med - lo) / max(1.0 - lo, 1e-10), 1e-06, 0.9999))
    mid = _mtf_solve_mid(x_sky, _MTF_BG_TARGET)
    hi = 1.0
    return (lo, mid, hi)

def _mtf_display(data: np.ndarray, lo: float, mid: float, hi: float, boost: float=0.5) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float64)
    out = _mtf_curve(arr, lo, mid, hi)
    if boost < 0.5:
        gamma = 1.0 + (0.5 - boost) / 0.5 * 1.0
    else:
        gamma = 1.0 - (boost - 0.5) / 0.5 * 0.7
    if abs(gamma - 1.0) > 0.001:
        out = np.power(np.clip(out, 0.0, 1.0), gamma)
    return np.clip(out, 0.0, 1.0).astype(np.float32)
models_list = [['SCUNet Color Real PSNR', 'https://github.com/cszn/KAIR/releases/download/v1.0/scunet_color_real_psnr.pth', 'Best all around model but can be too aggressive on stars'], ['SCUNet Color Real GAN', 'https://github.com/cszn/KAIR/releases/download/v1.0/scunet_color_real_gan.pth', 'Less aggressive denoise'], ['SCUNet Color 15', 'https://github.com/cszn/KAIR/releases/download/v1.0/scunet_color_15.pth', 'Gaussian noise level 15'], ['SCUNet Color 25', 'https://github.com/cszn/KAIR/releases/download/v1.0/scunet_color_25.pth', 'Gaussian noise level 25'], ['SCUNet Color 50', 'https://github.com/cszn/KAIR/releases/download/v1.0/scunet_color_50.pth', 'Gaussian noise level 50'], ['UberSmooth dso stars 0.1', 'https://ubersmooth.com/uberSmooth-dso-stars-v0.1.zip', 'Pretty good on stars but too aggressive on Hii regions'], ['UberSmooth dso stars 0.2', 'https://ubersmooth.com/uberSmooth-dso-stars-v0.2.zip', 'Not as aggressive as UberSmooth 0.1, but also not great'], ['UberSmooth planetary 0.1', 'https://ubersmooth.com/uberSmooth-planetary-v0.1.zip', 'Only denoise/deblur no extra star treatment']]

def _load_onnx_model(
    siril,
    checkpoint_path,
    tile_size,
    progress_callback=None,
    provider="directml",
):
    requested_provider = provider
    provider = _onnx_tools._select_windows_provider(
        provider, _OPENVINO_GPU_DEVICE is not None
    )
    if provider == "openvino":
        _onnx_tools._ensure_runtime_dependencies(s, provider)
    checkpoint_path = os.path.abspath(checkpoint_path)
    exporter_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "exportScunetToONNX.py"
    )
    if not os.path.isfile(exporter_path):
        raise FileNotFoundError(f"SCUNet ONNX exporter not found: {exporter_path}")

    def load_variant(core_size):
        active_provider = provider
        model_dimension = core_size + 2 * _onnx_tools.TILE_PAD
        halo_size = _onnx_tools.TILE_PAD
        suffix = f"_tile{core_size}_halo{halo_size}"
        model_path = (
            os.path.splitext(checkpoint_path)[0]
            + suffix
            + "_fp16.onnx"
        )
        if not os.path.isfile(model_path):
            if progress_callback is not None:
                progress_callback(
                    f"Exporting SCUNet ONNX model for {core_size}px tiles..."
                )
            siril.log(
                "No cached ONNX model found; PyTorch will run on CPU for "
                "one-time model export only. Image inference will use "
                f"ONNX Runtime {provider} ({core_size}px core + "
                f"{_onnx_tools.TILE_PAD}px context).",
                s.LogColor.BLUE,
            )
            command = [
                sys.executable,
                exporter_path,
                checkpoint_path,
                "--precision",
                "fp16",
                "--height",
                str(model_dimension),
                "--width",
                str(model_dimension),
                "--output-suffix",
                suffix,
            ]
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
            )
            for line in result.stdout.splitlines():
                siril.log(line, s.LogColor.BLUE)
            if result.returncode:
                details = result.stderr.strip() or result.stdout.strip()
                raise RuntimeError(
                    f"Could not export {os.path.basename(checkpoint_path)} "
                    f"for {core_size}px tiles to ONNX."
                    + (f"\n{details}" if details else "")
                )
            if not os.path.isfile(model_path):
                raise FileNotFoundError(
                    f"ONNX exporter completed without creating {model_path}"
                )
        else:
            siril.log(f"Using exported ONNX model: {model_path}", s.LogColor.BLUE)

        siril.log(
            f"Loading SCUNet ONNX model with {provider}; CPU execution-provider "
            "fallback is disabled.",
            s.LogColor.BLUE,
        )
        session, active_provider, fallback_error = (
            _onnx_tools._create_windows_inference_session(
                ort,
                model_path,
                requested_provider,
                _OPENVINO_GPU_DEVICE is not None,
            )
        )
        if fallback_error is not None:
            siril.log(
                "OpenVINO GPU could not activate this SCUNet model; "
                f"switching explicitly to DirectML: {fallback_error}",
                s.LogColor.RED,
            )
        input_shape = session.get_inputs()[0].shape
        if (
            len(input_shape) != 4
            or input_shape[0] != 1
            or input_shape[1] not in (1, 3)
            or input_shape[2:] != [model_dimension, model_dimension]
        ):
            raise RuntimeError(
                f"Expected a fixed one- or three-channel ONNX input with "
                f"spatial shape {model_dimension}x{model_dimension}, got "
                f"{input_shape} from {model_path}."
            )
        return session, active_provider

    core_size = 128 if tile_size == "Auto" else int(tile_size)
    if tile_size == "Auto":
        siril.log(
            f"{provider} Auto tile size is fixed at 128px to reduce the risk "
            "of a Windows GPU timeout (TDR).",
            s.LogColor.BLUE,
        )
    session, provider = load_variant(core_size)
    return session, core_size, provider


def _detect_openvino_gpu(siril):
    global _OPENVINO_GPU_DEVICE
    _OPENVINO_GPU_DEVICE = None
    try:
        _onnx_tools._ensure_runtime_dependencies(s, "openvino")
        _OPENVINO_GPU_DEVICE = _onnx_tools._get_openvino_gpu_device(ort)
    except Exception as error:
        siril.log(
            "OpenVINO GPU auto-detection failed; using DirectML by default. "
            f"Reason: {error}",
            s.LogColor.RED,
        )
        return None

    if _OPENVINO_GPU_DEVICE is None:
        siril.log(
            "No OpenVINO-compatible Intel GPU detected; using DirectML by default.",
            s.LogColor.BLUE,
        )
    else:
        device_name = _OPENVINO_GPU_DEVICE.ep_metadata.get(
            "FULL_DEVICE_NAME", "Intel GPU"
        )
        siril.log(
            f"OpenVINO-compatible GPU detected: {device_name}; selecting "
            "OpenVINO automatically (CPU fallback disabled).",
            s.LogColor.GREEN,
        )
    return _OPENVINO_GPU_DEVICE


def _onnx_tile_size(session, requested_size):
    input_shape = session.get_inputs()[0].shape
    if (
        len(input_shape) != 4
        or not isinstance(input_shape[2], int)
        or not isinstance(input_shape[3], int)
    ):
        raise ValueError(
            "The Windows ONNX backends require a fixed-shape model. "
            "Re-export the selected checkpoint without --dynamic-shapes."
        )
    core_height = input_shape[2] - 2 * _onnx_tools.TILE_PAD
    core_width = input_shape[3] - 2 * _onnx_tools.TILE_PAD
    if core_height <= 0 or core_height != core_width:
        raise ValueError(
            "ONNX model dimensions must equal the tile core plus "
            f"{2 * _onnx_tools.TILE_PAD}px of context."
        )
    if requested_size != "Auto" and int(requested_size) != core_height:
        raise ValueError(
            f"The ONNX model was exported for {core_height}px tile cores, "
            f"but {requested_size}px was requested."
        )
    return core_height


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        return torch.device('xpu')
    return torch.device('cpu')

def image_to_tensor(device: torch.device, img: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(img)
    return tensor.to(device)

def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    return np.rollaxis(tensor.cpu().detach().numpy(), 1, 4).squeeze(0).astype(np.float32)

def image_inference_tensor(model: ImageModelDescriptor, tensor: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        try:
            return model.model(tensor)
        except RuntimeError as e:
            if 'CreateOperator' in str(e) or 'IID_PPV_ARGS' in str(e):
                raise RuntimeError("Failed to run the SCUNet model on your hardware.\nPlease run the 'GPU_Manager.py' script from Siril's script manager and install the recommended PyTorch configuration for your system.") from e
            raise

def determine_optimal_tile_size(model, device, start_size=512):
    if device.type == 'cpu':
        return 256
    test_sizes = [512, 384, 256, 128]
    test_sizes = [s for s in test_sizes if s <= start_size]
    for size in test_sizes:
        try:
            dummy_input = torch.zeros(1, 3, size, size).to(device)
            with torch.no_grad():
                model(dummy_input)
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            elif device.type == 'xpu':
                torch.xpu.empty_cache()
            return size
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                elif device.type == 'xpu':
                    torch.xpu.empty_cache()
                continue
            else:
                raise e
    return 128

def get_tile_weight(h, w, device):
    x = torch.linspace(0, 1, w, device=device)
    y = torch.linspace(0, 1, h, device=device)
    wx = torch.min(x, 1 - x) * 2
    wy = torch.min(y, 1 - y) * 2
    wx = torch.clamp(wx, min=0.1)
    wy = torch.clamp(wy, min=0.1)
    weight = wy.unsqueeze(1) * wx.unsqueeze(0)
    return weight.unsqueeze(0)

def tile_process(device, model, data, scale, tile_size, yield_extra_details=False, apply_prestretch=False, precomputed_ghs_params=None):
    tile_pad = 144
    data = np.rollaxis(data, 2, 0)
    data = np.expand_dims(data, axis=0)
    batch, channel, height, width = data.shape
    stride = tile_size * 3 // 4

    def make_starts(dim, ts, st):
        starts = list(range(0, max(1, dim - ts), st))
        if not starts or starts[-1] + ts < dim:
            starts.append(max(0, dim - ts))
        return starts
    global_ghs_params = None
    if apply_prestretch:
        if precomputed_ghs_params is not None:
            global_ghs_params = precomputed_ghs_params
        else:
            full_img_hwc = data[0].transpose(1, 2, 0).astype(np.float64)
            global_ghs_params, _ = _compute_ghs_params(full_img_hwc)
    h_starts = make_starts(height, tile_size, stride)
    w_starts = make_starts(width, tile_size, stride)
    total_tiles = len(h_starts) * len(w_starts)
    tile_count = 0
    for input_start_y in h_starts:
        for input_start_x in w_starts:
            tile_count += 1
            input_end_x = input_start_x + tile_size
            input_end_y = input_start_y + tile_size
            input_start_x_pad = max(input_start_x - tile_pad, 0)
            input_end_x_pad = min(input_end_x + tile_pad, width)
            input_start_y_pad = max(input_start_y - tile_pad, 0)
            input_end_y_pad = min(input_end_y + tile_pad, height)
            input_tile_width = tile_size
            input_tile_height = tile_size
            input_tile = data[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad].astype(np.float32)
            tile_ghs_params = None
            if apply_prestretch:
                tile_hwc = input_tile[0].transpose(1, 2, 0).astype(np.float64)
                tile_stretched = _apply_stretch(tile_hwc, global_ghs_params)
                input_tile = tile_stretched.astype(np.float32).transpose(2, 0, 1)[np.newaxis]
            output_tile = image_inference_tensor(model, image_to_tensor(device, input_tile))
            progress = tile_count / total_tiles
            output_start_x_tile = (input_start_x - input_start_x_pad) * scale
            output_end_x_tile = output_start_x_tile + input_tile_width * scale
            output_start_y_tile = (input_start_y - input_start_y_pad) * scale
            output_end_y_tile = output_start_y_tile + input_tile_height * scale
            output_tile = output_tile[:, :, output_start_y_tile:output_end_y_tile, output_start_x_tile:output_end_x_tile]
            output_tile = tensor_to_image(output_tile)
            if apply_prestretch:
                out_hwc = output_tile.astype(np.float64)
                out_hwc = _apply_destretch(out_hwc, global_ghs_params)
                output_tile = out_hwc.astype(np.float32)
            if yield_extra_details:
                yield (output_tile, input_start_y, input_start_x, input_tile_width, input_tile_height, progress)
            else:
                yield output_tile
    yield None

def process_image_buffer(image_data, model, device, strength, tile_size, progress_callback=None, apply_prestretch=False, precomputed_ghs_params=None, provider="directml"):
    if _USE_ONNX_DIRECTML:
        from types import SimpleNamespace

        final_tile_size = _onnx_tile_size(model, tile_size)
        input_meta = model.get_inputs()[0]
        output_meta = model.get_outputs()[0]

        def onnx_progress(done, total, tile_elapsed, inference_elapsed):
            if progress_callback is None:
                return True
            return progress_callback(
                done / total,
                tile_elapsed,
                inference_elapsed,
            )

        args = SimpleNamespace(
            provider=provider,
            tile_size=final_tile_size,
            strength=strength,
            pre_stretch=apply_prestretch,
        )
        if isinstance(precomputed_ghs_params, (list, tuple)):
            precomputed_ghs_params = precomputed_ghs_params[0]
        if precomputed_ghs_params is not None and "d" not in precomputed_ghs_params:
            precomputed_ghs_params = {
                "lp": precomputed_ghs_params["LP"],
                "sp": precomputed_ghs_params["SP"],
                "d": precomputed_ghs_params["D"],
                "a": precomputed_ghs_params["A"],
                "b": precomputed_ghs_params["B"],
                "hp_norm": precomputed_ghs_params["hp_norm"],
            }
        try:
            result, _ = _onnx_tools._process_image(
                np,
                model,
                image_data,
                args,
                input_meta,
                output_meta,
                progress_callback=onnx_progress,
                precomputed_ghs_params=precomputed_ghs_params,
            )
        except _onnx_tools.InferenceCancelled:
            return None
        return result

    original_dtype = image_data.dtype
    if original_dtype == np.uint8:
        pixel_data = image_data.astype(np.float32) / 255.0
    elif original_dtype == np.uint16:
        actual_max = image_data.max()
        divisor = 255.0 if actual_max <= 255 else 65535.0
        pixel_data = image_data.astype(np.float32) / divisor
    else:
        pixel_data = image_data
    is_mono = False
    if pixel_data.ndim == 2:
        is_mono = True
        pixel_data = np.stack((pixel_data,) * 3, axis=0)
    elif pixel_data.ndim == 3 and pixel_data.shape[0] == 1:
        is_mono = True
        pixel_data = np.repeat(pixel_data, 3, axis=0)
    c, h, w = pixel_data.shape
    pixel_data_hwc = np.transpose(pixel_data, (1, 2, 0))
    output_sum = torch.zeros((c, h, w), dtype=torch.float32, device='cpu')
    output_weight = torch.zeros((c, h, w), dtype=torch.float32, device='cpu')
    base_weight_mask = get_tile_weight(tile_size, tile_size, 'cpu')
    scale = 1
    for i, tile_info in enumerate(tile_process(device, model, pixel_data_hwc, scale, tile_size, yield_extra_details=True, apply_prestretch=apply_prestretch, precomputed_ghs_params=precomputed_ghs_params)):
        if tile_info is None:
            break
        tile_data_numpy, y_start, x_start, _, _, p = tile_info
        if progress_callback:
            if progress_callback(p) is False:
                return None
        tile_tensor = torch.from_numpy(tile_data_numpy.transpose(2, 0, 1))
        c_real, h_real, w_real = tile_tensor.shape
        if h_real <= 0 or w_real <= 0:
            continue
        y_end = y_start + h_real
        x_end = x_start + w_real
        y_end_safe = min(y_end, h)
        x_end_safe = min(x_end, w)
        y_start_safe = max(0, y_start)
        x_start_safe = max(0, x_start)
        write_h = y_end_safe - y_start_safe
        write_w = x_end_safe - x_start_safe
        if write_h <= 0 or write_w <= 0:
            continue
        if write_h != h_real or write_w != w_real:
            tile_tensor = tile_tensor[:, :write_h, :write_w]
            h_real = write_h
            w_real = write_w
        if h_real == tile_size and w_real == tile_size:
            mask = base_weight_mask
        else:
            mask = get_tile_weight(h_real, w_real, 'cpu')
        output_sum[:, y_start_safe:y_end_safe, x_start_safe:x_end_safe] += tile_tensor * mask
        output_weight[:, y_start_safe:y_end_safe, x_start_safe:x_end_safe] += mask
    output_image_tensor = output_sum / (output_weight + 1e-08)
    output_image = output_image_tensor.numpy()
    if is_mono:
        output_image = output_image[0, :, :]
    final_dtype = original_dtype
    if original_dtype == np.uint8:
        output_image = np.clip(output_image, 0, 1) * 255.0
        output_image = output_image.astype(np.uint8)
    elif original_dtype == np.uint16:
        output_image = np.clip(output_image, 0, 1) * divisor
        output_image = output_image.astype(np.uint16)
    if strength != 1.0:
        if is_mono and image_data.ndim == 3:
            bg = image_data[0]
        else:
            bg = image_data
        blended = output_image * strength + bg * (1 - strength)
        return blended.astype(final_dtype)
    else:
        return output_image

class ProcessingWorker(QObject):
    finished = pyqtSignal()
    progress_update = pyqtSignal(int, str)
    error_occurred = pyqtSignal(str)

    def __init__(self, siril, params):
        super().__init__()
        self.siril = siril
        self.params = params
        self._is_running = True

    def run(self):
        try:
            model_url = self.params['model_url']
            strength = self.params['strength']
            is_sequence = self.params['is_sequence']
            seq_prefix = self.params['seq_prefix']
            apply_prestretch = self.params.get('apply_prestretch', False)
            provider = self.params.get('provider', 'directml')
            req_tile = self.params['tile_size']
            self.progress_update.emit(0, 'Checking Model...')
            user_dir = self.siril.get_siril_userdatadir()
            models_dir = os.path.join(user_dir, 'scunet_models')
            if not os.path.exists(models_dir):
                os.makedirs(models_dir)
            model_filename = os.path.basename(model_url)
            modelpath = os.path.join(models_dir, model_filename)

            def download_progress_hook(block_num, block_size, total_size):
                if not self._is_running:
                    raise Exception('Download cancelled')
                if total_size > 0:
                    downloaded = block_num * block_size
                    percent = int(downloaded / total_size * 100)
                    if percent % 2 == 0:
                        self.progress_update.emit(percent, f'Downloading Model: {percent}%')
                else:
                    self.progress_update.emit(0, 'Downloading Model... (size unknown)')
            if not os.path.isfile(modelpath):
                self.siril.log(f'Downloading model to: {modelpath}', s.LogColor.BLUE)
                ssl._create_default_https_context = ssl._create_stdlib_context
                try:
                    urllib.request.urlretrieve(model_url, modelpath, reporthook=download_progress_hook)
                    self.siril.log('Model download completed.', s.LogColor.GREEN)
                except Exception as e:
                    if os.path.exists(modelpath):
                        os.remove(modelpath)
                    raise e
            else:
                self.siril.log(f'Using existing model at: {modelpath}', s.LogColor.BLUE)
            if zipfile.is_zipfile(modelpath):
                with zipfile.ZipFile(modelpath, 'r') as zip_ref:
                    zip_ref.extractall(models_dir)
                    modelpath = modelpath.replace('.zip', '.pth')
            self.progress_update.emit(0, 'Loading Model into Memory...')
            if _USE_ONNX_DIRECTML:
                model, auto_tile_size, provider = _load_onnx_model(
                    self.siril,
                    modelpath,
                    req_tile,
                    lambda message: self.progress_update.emit(0, message),
                    provider=provider,
                )
                device = None
                backend_name = (
                    "ONNX Runtime DirectML"
                    if provider == "directml"
                    else f"ONNX Runtime OpenVINO GPU ({_onnx_tools._OPENVINO_SELECTED_DEVICE})"
                )
                self.siril.log(
                    f"SCUNet inference backend: {backend_name} "
                    "(CPU execution-provider fallback disabled).",
                    s.LogColor.GREEN,
                )
                self.siril.log(
                    f"ONNX Runtime registered providers: {', '.join(model.get_providers())}",
                    s.LogColor.BLUE,
                )
            else:
                device = get_device()
                self.siril.log('------ Hardware Info ------', s.LogColor.BLUE)
                cuda_available = torch.cuda.is_available()
                self.siril.log(f'CUDA Available: {cuda_available}', s.LogColor.GREEN if cuda_available else s.LogColor.RED)
                if cuda_available:
                    num_gpus = torch.cuda.device_count()
                    for i in range(num_gpus):
                        self.siril.log(f'  - GPU {i}: {torch.cuda.get_device_name(i)}', s.LogColor.GREEN)
                mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
                self.siril.log(f'MPS (Apple) Available: {mps_available}', s.LogColor.GREEN if mps_available else s.LogColor.RED)
                xpu_available = hasattr(torch, 'xpu') and torch.xpu.is_available()
                self.siril.log(f'XPU (Intel) Available: {xpu_available}', s.LogColor.GREEN if xpu_available else s.LogColor.RED)
                self.siril.log(f"Active Device: {device.type.upper()}", s.LogColor.BLUE)
                self.siril.log('---------------------------', s.LogColor.BLUE)
                if device.type == 'cuda':
                    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Unknown NVIDIA GPU'
                    self.siril.log(f'Acceleration: CUDA ({device_name})', s.LogColor.GREEN)
                elif device.type == 'mps':
                    self.siril.log('Acceleration: Apple Metal Performance Shaders (MPS)', s.LogColor.GREEN)
                elif device.type == 'xpu':
                    self.siril.log('Acceleration: Intel XPU (Arc GPU detected)', s.LogColor.GREEN)
                else:
                    self.siril.log('Acceleration: CPU (No GPU detected)', s.LogColor.GREEN)
                model = ModelLoader().load_from_file(str(modelpath)).eval().to(device)
                architecture_name = type(model.model).__name__
                if 'SCUNet' not in architecture_name:
                    raise RuntimeError(f"Invalid model selected. Expected a SCUNet model, but detected: '{architecture_name}'.\nPlease select a valid SCUNet .pth file.")
                assert isinstance(model, ImageModelDescriptor)
            if _USE_ONNX_DIRECTML:
                final_tile_size = (
                    auto_tile_size
                    if req_tile == 'Auto'
                    else _onnx_tile_size(model, req_tile)
                )
                provider_label = (
                    "DirectML" if provider == "directml" else "OpenVINO GPU"
                )
                self.siril.log(
                    f'ONNX {provider_label} tile size: {final_tile_size}px',
                    s.LogColor.GREEN,
                )
            elif req_tile == 'Auto':
                self.progress_update.emit(0, 'Auto-tuning Tile Size...')
                final_tile_size = determine_optimal_tile_size(model, device)
                self.siril.log(f'Auto-Tuning: Selected Tile Size {final_tile_size}px', s.LogColor.GREEN)
            else:
                final_tile_size = req_tile
                self.siril.log(f'Manual Tile Size: {final_tile_size}px', s.LogColor.BLUE)
            if is_sequence:
                if not self.siril.is_sequence_loaded():
                    raise RuntimeError('No sequence loaded in Siril.')
                current_sequence = self.siril.get_seq()
                num_images = current_sequence.number
                for i in range(num_images):
                    if not self._is_running:
                        break
                    if not current_sequence.imgparam[i].incl:
                        continue
                    filename = self.siril.get_seq_frame_filename(i)
                    _, file_extension = os.path.splitext(filename)
                    if not file_extension:
                        file_extension = '.fit'
                    ffit_image = self.siril.load_image_from_file(filename, with_pixels=True)
                    image_data = ffit_image.data
                    header = ffit_image.header

                    frame_started = time.perf_counter()
                    inference_totals = [0.0]

                    def callback(tile_p, tile_elapsed=None, inference_elapsed=None):
                        if not self._is_running:
                            return False
                        global_p = (i + tile_p) / num_images
                        status = f'Processing frame {i + 1}/{num_images}'
                        if tile_elapsed is not None:
                            inference_totals[0] = inference_elapsed
                            status += (
                                f' | tile {tile_elapsed:.2f}s'
                                f' | inference {inference_elapsed:.2f}s'
                            )
                        self.progress_update.emit(int(global_p * 100), status)
                        return True
                    processed_data = process_image_buffer(image_data, model, device, strength, final_tile_size, callback, apply_prestretch=apply_prestretch, provider=provider)
                    if processed_data is None:
                        self.siril.log('Processing cancelled. No data saved.', s.LogColor.RED)
                        self.finished.emit()
                        return
                    if _USE_ONNX_DIRECTML:
                        self.siril.log(
                            f'SCUNet timing frame {i + 1}: inference '
                            f'{inference_totals[0]:.2f}s; total '
                            f'{time.perf_counter() - frame_started:.2f}s',
                            s.LogColor.BLUE,
                        )
                    new_filename = f'{seq_prefix}{i + 1:05d}{file_extension}'
                    self.siril.save_image_file(processed_data, header=header, filename=new_filename)
                self.progress_update.emit(100, f'Sequence saved: {seq_prefix}...')
            else:
                if not self.siril.is_image_loaded():
                    raise RuntimeError('No image loaded in Siril.')
                image = self.siril.get_image()
                image_data = image.data

                image_started = time.perf_counter()
                inference_totals = [0.0]

                def callback(tile_p, tile_elapsed=None, inference_elapsed=None):
                    if not self._is_running:
                        return False
                    status = 'Denoising Image...'
                    if tile_elapsed is not None:
                        inference_totals[0] = inference_elapsed
                        status += (
                            f' | tile {tile_elapsed:.2f}s'
                            f' | inference {inference_elapsed:.2f}s'
                        )
                    self.progress_update.emit(int(tile_p * 100), status)
                    return True
                processed_data = process_image_buffer(image_data, model, device, strength, final_tile_size, callback, apply_prestretch=apply_prestretch, provider=provider)
                if processed_data is None:
                    self.siril.log('Processing cancelled. No data saved.', s.LogColor.RED)
                    self.finished.emit()
                    return
                if _USE_ONNX_DIRECTML:
                    self.siril.log(
                        f'SCUNet timing: model inference '
                        f'{inference_totals[0]:.2f}s; total processing '
                        f'{time.perf_counter() - image_started:.2f}s',
                        s.LogColor.BLUE,
                    )
                try:
                    model_name = next((m[0] for m in models_list if m[1] == self.params['model_url']), 'Unknown Model')
                    self.siril.undo_save_state(f'SCUNet denoise - {model_name} | Strength: {strength:.2f}')
                    with self.siril.image_lock():
                        self.siril.set_image_pixeldata(processed_data)
                        self.siril.log(f'SCUNet denoise - {model_name} | Strength: {strength:.2f} - Done.', s.LogColor.GREEN)
                        try:
                            if processed_data.dtype == np.uint8:
                                self.siril.set_siril_slider_lohi(0, 255)
                            elif processed_data.dtype == np.uint16:
                                orig_max = int(image_data.max())
                                slider_hi = 255 if orig_max <= 255 else 65535
                                self.siril.set_siril_slider_lohi(0, slider_hi)
                            elif np.issubdtype(processed_data.dtype, np.floating):
                                self.siril.set_siril_slider_lohi(0.0, 1.0)
                        except Exception:
                            pass
                except Exception as e:
                    print(f'Could not send output to Siril: {e}')
                self.progress_update.emit(100, 'Done.')
            self.finished.emit()
        except Exception as e:
            traceback.print_exc()
            self.error_occurred.emit(str(e))

    def stop(self):
        self._is_running = False

def numpy_to_qpixmap(img_data):
    if img_data.ndim == 3:
        img_data = np.flip(img_data, axis=1)
    else:
        img_data = np.flip(img_data, axis=0)
    if img_data.dtype == np.uint16:
        actual_max = int(img_data.max())
        divisor = 255 if actual_max <= 255 else 65535
        img_disp = (np.clip(img_data, 0, divisor) / divisor * 255.0).astype(np.uint8)
    elif img_data.dtype == np.float32 or img_data.dtype == np.float64:
        img_disp = (np.clip(img_data, 0.0, 1.0) * 255.0).astype(np.uint8)
    else:
        img_disp = np.clip(img_data, 0, 255).astype(np.uint8)
    if img_disp.ndim == 3:
        img_disp = np.transpose(img_disp, (1, 2, 0))
        h, w, c = img_disp.shape
        img_disp = np.ascontiguousarray(img_disp)
        fmt = QImage.Format.Format_RGB888
        bytes_per_line = c * w
    else:
        h, w = img_disp.shape
        img_disp = np.ascontiguousarray(img_disp)
        fmt = QImage.Format.Format_Grayscale8
        bytes_per_line = w
    qimg = QImage(img_disp.data, w, h, bytes_per_line, fmt)
    return QPixmap.fromImage(qimg.copy())

class PreviewWorker(QObject):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    progress_update = pyqtSignal(int, str)

    def __init__(self, siril, img_data, params):
        super().__init__()
        self.siril = siril
        self.img_data = img_data
        self.params = params

    def run(self):
        try:
            model_url = self.params['model_url']
            strength = self.params['strength']
            apply_prestretch = self.params.get('apply_prestretch', False)
            provider = self.params.get('provider', 'directml')
            user_dir = self.siril.get_siril_userdatadir()
            models_dir = os.path.join(user_dir, 'scunet_models')
            if not os.path.exists(models_dir):
                os.makedirs(models_dir)
            model_filename = os.path.basename(model_url)
            modelpath = os.path.join(models_dir, model_filename)

            def download_progress_hook(block_num, block_size, total_size):
                if total_size > 0:
                    downloaded = block_num * block_size
                    percent = int(downloaded / total_size * 100)
                    if percent % 2 == 0:
                        self.progress_update.emit(percent, f'Downloading Model: {percent}%')
                else:
                    self.progress_update.emit(0, 'Downloading Model... (size unknown)')
            if not os.path.isfile(modelpath):
                self.progress_update.emit(0, 'Starting Download for Preview...')
                ssl._create_default_https_context = ssl._create_stdlib_context
                urllib.request.urlretrieve(model_url, modelpath, reporthook=download_progress_hook)
                self.progress_update.emit(100, 'Download Complete.')
            if zipfile.is_zipfile(modelpath):
                with zipfile.ZipFile(modelpath, 'r') as zip_ref:
                    zip_ref.extractall(models_dir)
                    modelpath = modelpath.replace('.zip', '.pth')
            if self.img_data.ndim == 3:
                h_img, w_img = (self.img_data.shape[1], self.img_data.shape[2])
            else:
                h_img, w_img = (self.img_data.shape[0], self.img_data.shape[1])
            tile_size = min(128 if _USE_ONNX_DIRECTML else 512, h_img, w_img)
            self.progress_update.emit(0, 'Loading Model into Memory...')
            if _USE_ONNX_DIRECTML:
                model, tile_size, provider = _load_onnx_model(
                    self.siril,
                    modelpath,
                    tile_size,
                    lambda message: self.progress_update.emit(0, message),
                    provider=provider,
                )
                device = None
                backend_name = (
                    "ONNX Runtime DirectML"
                    if provider == "directml"
                    else f"ONNX Runtime OpenVINO GPU ({_onnx_tools._OPENVINO_SELECTED_DEVICE})"
                )
                self.siril.log(
                    f"SCUNet preview backend: {backend_name} "
                    "(CPU execution-provider fallback disabled).",
                    s.LogColor.GREEN,
                )
                self.siril.log(
                    f"ONNX Runtime registered providers: {', '.join(model.get_providers())}",
                    s.LogColor.BLUE,
                )
            else:
                device = get_device()
                self.siril.log('------ Hardware Info ------', s.LogColor.BLUE)
                cuda_available = torch.cuda.is_available()
                self.siril.log(f'CUDA Available: {cuda_available}', s.LogColor.GREEN if cuda_available else s.LogColor.RED)
                if cuda_available:
                    num_gpus = torch.cuda.device_count()
                    for i in range(num_gpus):
                        self.siril.log(f'  - GPU {i}: {torch.cuda.get_device_name(i)}', s.LogColor.GREEN)
                mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
                self.siril.log(f'MPS (Apple) Available: {mps_available}', s.LogColor.GREEN if mps_available else s.LogColor.RED)
                xpu_available = hasattr(torch, 'xpu') and torch.xpu.is_available()
                self.siril.log(f'XPU (Intel) Available: {xpu_available}', s.LogColor.GREEN if xpu_available else s.LogColor.RED)
                self.siril.log(f"Active Device: {device.type.upper()}", s.LogColor.BLUE)
                self.siril.log('---------------------------', s.LogColor.BLUE)
                if device.type == 'cuda':
                    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Unknown NVIDIA GPU'
                    self.siril.log(f'Acceleration: CUDA ({device_name})', s.LogColor.GREEN)
                elif device.type == 'mps':
                    self.siril.log('Acceleration: Apple Metal Performance Shaders (MPS)', s.LogColor.GREEN)
                elif device.type == 'xpu':
                    self.siril.log('Acceleration: Intel XPU (Arc GPU detected)', s.LogColor.GREEN)
                else:
                    self.siril.log('Acceleration: CPU (No GPU detected)', s.LogColor.GREEN)
                model = ModelLoader().load_from_file(str(modelpath)).eval().to(device)
            if _USE_ONNX_DIRECTML:
                tile_size = _onnx_tile_size(model, tile_size)
            self.siril.log(f'Tile Size {tile_size}px', s.LogColor.GREEN)

            preview_started = time.perf_counter()
            inference_totals = [0.0]

            def callback(tile_p, tile_elapsed=None, inference_elapsed=None):
                status = 'Processing ROI Preview...'
                if tile_elapsed is not None:
                    inference_totals[0] = inference_elapsed
                    status += (
                        f' | tile {tile_elapsed:.2f}s'
                        f' | inference {inference_elapsed:.2f}s'
                    )
                self.progress_update.emit(int(tile_p * 100), status)
                return True
            precomputed_ghs_params = self.params.get('precomputed_ghs_params', None)
            processed_roi = process_image_buffer(self.img_data, model, device, strength, tile_size, progress_callback=callback, apply_prestretch=apply_prestretch, precomputed_ghs_params=precomputed_ghs_params, provider=provider)
            if _USE_ONNX_DIRECTML:
                self.siril.log(
                    f'SCUNet preview timing: model inference '
                    f'{inference_totals[0]:.2f}s; total processing '
                    f'{time.perf_counter() - preview_started:.2f}s',
                    s.LogColor.BLUE,
                )
            self.finished.emit(processed_roi)
            self.progress_update.emit(0, 'Completed / Ready')
        except Exception as e:
            traceback.print_exc()
            self.error.emit(str(e))

class PreviewWindow(QWidget):
    progress_update = pyqtSignal(int, str)
    window_closed = pyqtSignal()

    def __init__(self, siril, main_window_params):
        super().__init__()
        self.setWindowTitle(f'SCUNet Denoise - Preview & Blink - v{VERSION}')
        self.resize(600, 650)
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)
        self.siril = siril
        self.params = main_window_params
        self.original_data = None
        self.processed_data = None
        self.pixmap_orig = None
        self.pixmap_proc = None
        self.thread = None
        self.worker = None
        self.current_poly = None
        self.current_poly_coords = None
        self.global_mtf_params = (0.0, 0.25, 1.0)
        self.display_boost = 0.5
        self.setup_ui()
        QTimer.singleShot(0, self._deferred_init)

    def _deferred_init(self):
        self._compute_global_mtf()
        self.fetch_and_process()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        self.lbl_image = QLabel('Processing Preview...')
        self.lbl_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_image.setStyleSheet('background-color: #222; border: 1px solid #444;')
        self.lbl_image.setMinimumSize(500, 500)
        layout.addWidget(self.lbl_image, 1)
        ctrl_layout = QHBoxLayout()
        self.btn_update = QPushButton('Update ROI')
        self.btn_update.setToolTip('Update preview with current selection from Siril.')
        self.btn_update.clicked.connect(self.fetch_and_process)
        self.btn_blink = QPushButton('Hold to BLINK (Show Original)')
        self.btn_blink.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_blink.pressed.connect(self.show_original)
        self.btn_blink.released.connect(self.show_processed)
        self.btn_blink.setEnabled(False)
        ctrl_layout.addWidget(self.btn_update)
        ctrl_layout.addWidget(self.btn_blink)
        layout.addLayout(ctrl_layout)
        self.gamma_widget = QWidget()
        gamma_layout = QHBoxLayout(self.gamma_widget)
        gamma_layout.setContentsMargins(0, 0, 0, 0)
        gamma_layout.addWidget(QLabel('Display Boost:'))
        self.gamma_slider = QSlider(Qt.Orientation.Horizontal)
        self.gamma_slider.setMinimum(0)
        self.gamma_slider.setMaximum(100)
        self.gamma_slider.setValue(50)
        self.gamma_slider.setTickInterval(10)
        self.gamma_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.gamma_slider.valueChanged.connect(self._on_boost_changed)
        gamma_layout.addWidget(self.gamma_slider)
        self.gamma_label = QLabel('0.50')
        self.gamma_label.setFixedWidth(35)
        gamma_layout.addWidget(self.gamma_label)
        self.gamma_widget.setVisible(self.params.get('apply_prestretch', False))
        layout.addWidget(self.gamma_widget)
        lbl_info = QLabel('<i>Move selection in Siril window and click <b>Update ROI</b>.</i>')
        lbl_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(lbl_info)

    def _compute_global_mtf(self):
        try:
            ref_data = self.siril.get_image_pixeldata(preview=False)
            if ref_data is not None:
                if ref_data.ndim == 3:
                    ref_hwc = np.transpose(ref_data.astype(np.float32), (1, 2, 0))
                else:
                    ref_hwc = ref_data.astype(np.float32)
                lo, mid, hi = _compute_mtf_display_params(ref_hwc)
                self.global_mtf_params = (lo, mid, hi)
                print(f'[MTF Display] Params globali - lo={lo:.6f}, mid={mid:.6f}, hi={hi:.6f}')
                self.siril.log(f'MTF Display: lo={lo:.4f}  mid={mid:.4f}  (sky target={_MTF_BG_TARGET})', s.LogColor.BLUE)
        except Exception as e:
            print(f'[MTF Display] Fallback params (errore: {e})')
            self.global_mtf_params = (0.0, 0.25, 1.0)

    def fetch_and_process(self):
        try:
            raw_selection = self.siril.get_siril_selection()
            if raw_selection:
                selection = list(raw_selection)
            else:
                selection = []
            if not selection or selection[2] <= 0 or selection[3] <= 0:
                self.lbl_image.setText('Invalid Selection. Creating default...')
                shape = self.siril.get_image_shape()
                if len(shape) == 3:
                    H, W = (shape[1], shape[2])
                else:
                    H, W = (shape[0], shape[1])
                safe_size = min(500, W - 50, H - 50)
                roi_w, roi_h = (safe_size, safe_size)
                cx, cy = (W // 2, H // 2)
                x = max(0, cx - roi_w // 2)
                y = max(0, cy - roi_h // 2)
                self.siril.set_siril_selection(x, y, roi_w, roi_h)
                selection = [x, y, roi_w, roi_h]
                self.siril.log(f'Preview ROI set at x={x}, y={y} ({roi_w}x{roi_h})', s.LogColor.BLUE)
            x, y, w, h = selection
            square_size = min(w, h)
            final_selection = [x, y, square_size, square_size]
            if final_selection != selection:
                self.siril.set_siril_selection(*final_selection)
            selection = final_selection
            if self.current_poly is None or selection != self.current_poly_coords:
                if self.current_poly is not None:
                    try:
                        self.siril.overlay_delete_polygon(self.current_poly.polygon_id)
                    except Exception:
                        pass
                    self.current_poly = None
                try:
                    x, y, w, h = selection
                    poly = s.Polygon.from_rectangle((x, y, w, h), color=16711808, fill=False, legend='ROI Preview')
                    self.current_poly = self.siril.overlay_add_polygon(poly)
                    self.current_poly_coords = selection
                except AttributeError:
                    pass
            roi_data = self.siril.get_image_pixeldata(preview=False, shape=selection)
            if roi_data is None:
                raise ValueError('Could not retrieve pixel data from Siril.')
            self.original_data = roi_data.copy()
            if self.params.get('apply_prestretch', False):
                full_data = self.siril.get_image_pixeldata(preview=False)
                full_hwc = np.transpose(full_data.astype(np.float64), (1, 2, 0) if full_data.ndim == 3 else (0, 1))
                global_ghs, _ = _compute_ghs_params(full_hwc)
                self.params['precomputed_ghs_params'] = global_ghs
                lo, mid, hi = self.global_mtf_params
                orig_display = _mtf_display(self.original_data.astype(np.float32), lo, mid, hi, self.display_boost)
                self.pixmap_orig = numpy_to_qpixmap(orig_display)
            else:
                self.pixmap_orig = numpy_to_qpixmap(self.original_data)
            self.lbl_image.setText('Processing ROI...')
            self.btn_blink.setEnabled(False)
            self.btn_update.setEnabled(False)
            self.thread = QThread()
            self.worker = PreviewWorker(self.siril, self.original_data, self.params)
            self.worker.moveToThread(self.thread)
            self.thread.started.connect(self.worker.run)
            self.worker.finished.connect(self.on_process_done)
            self.worker.error.connect(self.on_process_error)
            self.worker.progress_update.connect(self.progress_update)
            self.worker.finished.connect(self.thread.quit)
            self.worker.finished.connect(self.worker.deleteLater)
            self.thread.finished.connect(self.thread.deleteLater)
            self.thread.start()
        except Exception as e:
            self.lbl_image.setText(f'Error fetching ROI:\n{e}')
            print(f'Preview Error: {e}')
            traceback.print_exc()

    def on_process_done(self, processed_array):
        self.processed_data = processed_array
        if self.params.get('apply_prestretch', False):
            lo, mid, hi = self.global_mtf_params
            proc_display = _mtf_display(processed_array.astype(np.float32), lo, mid, hi, self.display_boost)
        else:
            proc_display = processed_array
        self.pixmap_proc = numpy_to_qpixmap(proc_display)
        self.show_processed()
        self.btn_blink.setEnabled(True)
        self.btn_update.setEnabled(True)

    def on_process_error(self, err_msg):
        self.lbl_image.setText(f'Processing Failed:\n{err_msg}')
        self.btn_update.setEnabled(True)

    def _on_boost_changed(self, value):
        self.display_boost = value / 100.0
        self.gamma_label.setText(f'{self.display_boost:.2f}')
        lo, mid, hi = self.global_mtf_params
        if self.params.get('apply_prestretch', False):
            if self.processed_data is not None:
                self.pixmap_proc = numpy_to_qpixmap(_mtf_display(self.processed_data.astype(np.float32), lo, mid, hi, self.display_boost))
            if self.original_data is not None:
                self.pixmap_orig = numpy_to_qpixmap(_mtf_display(self.original_data.astype(np.float32), lo, mid, hi, self.display_boost))
        else:
            if self.processed_data is not None:
                self.pixmap_proc = numpy_to_qpixmap(self.processed_data)
            if self.original_data is not None:
                self.pixmap_orig = numpy_to_qpixmap(self.original_data)
        if self.btn_blink.isDown():
            self.show_original()
        else:
            self.show_processed()

    def show_original(self):
        if self.pixmap_orig:
            self.lbl_image.setPixmap(self.pixmap_orig.scaled(self.lbl_image.size(), Qt.AspectRatioMode.KeepAspectRatio))

    def show_processed(self):
        if self.pixmap_proc:
            self.lbl_image.setPixmap(self.pixmap_proc.scaled(self.lbl_image.size(), Qt.AspectRatioMode.KeepAspectRatio))

    def resizeEvent(self, event):
        if self.btn_blink.isDown():
            self.show_original()
        else:
            self.show_processed()
        super().resizeEvent(event)

    def closeEvent(self, event: QCloseEvent):
        if self.thread:
            try:
                if self.thread.isRunning():
                    self.thread.quit()
                    self.thread.wait()
            except RuntimeError:
                pass
        try:
            self.siril.set_siril_selection(0, 0, 0, 0)
            self.siril.overlay_clear_polygons()
            if self.current_poly is not None:
                self.siril.overlay_delete_polygon(self.current_poly.polygon_id)
            else:
                self.siril.overlay_clear_polygons()
            self.siril.log('Preview Window (ROI) closed by User. Selection cleared.', s.LogColor.BLUE)
        except Exception as e:
            print(f'Error clearing selection: {e}')
        self.window_closed.emit()
        super().closeEvent(event)

class ScunetWindow(QWidget):

    def __init__(self):
        super().__init__()
        self.setWindowTitle(f'SCUNet Denoise - v{VERSION}')
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)
        self.siril = s.SirilInterface()
        try:
            self.siril.connect()
        except Exception:
            QMessageBox.critical(self, 'Connection Error', 'Connection to Siril failed. Make sure Siril is open and ready.')
            sys.exit(1)
        if _USE_ONNX_DIRECTML:
            _detect_openvino_gpu(self.siril)
        backend_name = (
            (
                "ONNX Runtime OpenVINO GPU "
                f"({_OPENVINO_GPU_DEVICE.ep_metadata.get('FULL_DEVICE_NAME', 'Intel GPU')})"
                if _OPENVINO_GPU_DEVICE is not None
                else "ONNX Runtime DirectML"
            )
            + " (CPU EP fallback disabled)"
            if _USE_ONNX_DIRECTML
            else "PyTorch"
        )
        self.siril.log(
            f"SCUNet Denoise v{VERSION}; selected backend: {backend_name}.",
            s.LogColor.GREEN,
        )
        try:
            self.siril.cmd('requires', '1.4.2')
        except s.CommandError:
            sys.exit(1)
        image_loaded = self.siril.is_image_loaded()
        seq_loaded = self.siril.is_sequence_loaded()
        if seq_loaded:
            self.siril.log('Context: Sequence loaded.', s.LogColor.BLUE)
            self.is_sequence_context = True
        elif image_loaded:
            self.siril.log('Context: Single image loaded.', s.LogColor.BLUE)
            self.is_sequence_context = False
        else:
            self.siril.error_messagebox('No image or sequence loaded')
            sys.exit(1)
        self.siril.set_siril_selection(0, 0, 0, 0)
        self.siril.overlay_clear_polygons()
        self.preview_poly = None
        self.setup_ui()
        self.thread = None
        self.worker = None
        self.center_window()

    def center_window(self):
        screen_geometry = self.screen().availableGeometry()
        self.resize(520, 500)
        self.move(int((screen_geometry.width() - self.width()) / 2), int((screen_geometry.height() - self.height()) / 2))

    def setup_ui(self):
        layout = QVBoxLayout(self)
        lbl_credits = QLabel("<span style='color:#f4d742;'><b>Original version by Nicolas CASTEL</b></span><br>Refactoring by Carlo Mollicone - <a href='https://www.astroboh.it' style='color:#4E8AFC; text-decoration:none;'>AstroBOH.it</a>")
        lbl_credits.setOpenExternalLinks(True)
        lbl_credits.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_credits.setStyleSheet('color: #888; font-size: 12px; margin-bottom: 5px;')
        layout.addWidget(lbl_credits)
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(line)
        gb_model = QGroupBox('Model Selection')
        layout_model = QHBoxLayout()
        left_indices = [2, 3, 4]
        right_indices = [0, 1, 5, 6, 7]
        self.model_buttons = []
        self.model_button_group = QButtonGroup(self)

        def make_column(title, subtitle, indices):
            container = QWidget()
            col = QVBoxLayout(container)
            col.setSpacing(4)
            lbl_title = QLabel(f'<b>{title}</b>')
            lbl_sub = QLabel(f'<small>{subtitle}</small>')
            lbl_sub.setWordWrap(True)
            col.addWidget(lbl_title)
            col.addWidget(lbl_sub)
            col.addSpacing(4)
            for i in indices:
                m = models_list[i]
                rb = QRadioButton(m[0])
                rb.setToolTip(m[2])
                rb.setProperty('url', m[1])
                self.model_button_group.addButton(rb, i)
                col.addWidget(rb)
                self.model_buttons.append(rb)
            col.addStretch()
            return container
        layout_model.addWidget(make_column('Non-Linear Only', 'Pre-stretched images', left_indices))
        line = QFrame()
        line.setFrameShape(QFrame.Shape.VLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        layout_model.addWidget(line)
        layout_model.addWidget(make_column('Linear & Non-Linear', "For linear images, enable 'Apply PreStretch'", right_indices))
        self.model_button_group.button(0).setChecked(True)
        gb_model.setLayout(layout_model)
        layout.addWidget(gb_model)
        gb_params = QGroupBox('Parameters')
        layout_params = QVBoxLayout()
        h_slider = QHBoxLayout()
        self.slider_strength = QSlider(Qt.Orientation.Horizontal)
        self.slider_strength.setRange(0, 100)
        self.slider_strength.setValue(50)
        self.lbl_strength_val = QLabel('0.50')
        self.slider_strength.valueChanged.connect(lambda val: self.lbl_strength_val.setText(f'{val / 100:.2f}'))
        h_slider.addWidget(QLabel('Strength:'))
        h_slider.addWidget(self.slider_strength)
        h_slider.addWidget(self.lbl_strength_val)
        layout_params.addLayout(h_slider)
        h_tile = QHBoxLayout()
        self.combo_tile = QComboBox()
        self.combo_tile.addItems(['Auto', '512', '384', '256', '128'])
        self.combo_tile.setToolTip(
            "Tile size for processing.\n"
            "'Auto' selects 128px on Windows and tunes to the available device elsewhere.\n"
            "Windows exports a fixed ONNX input with 128px of context around the selected tile core."
        )
        h_tile.addWidget(QLabel('Tile Size:'))
        h_tile.addWidget(self.combo_tile)
        layout_params.addLayout(h_tile)
        if _USE_ONNX_DIRECTML:
            h_backend = QHBoxLayout()
            self.combo_backend = QComboBox()
            self.combo_backend.addItem(
                'Automatic (OpenVINO GPU if detected)', 'auto'
            )
            self.combo_backend.addItem('DirectML', 'directml')
            self.combo_backend.addItem('OpenVINO GPU (manual)', 'openvino')
            self.combo_backend.setToolTip(
                "Automatic selects the compatible Intel OpenVINO GPU when "
                "available; otherwise it uses DirectML. If OpenVINO cannot "
                "activate the selected model, Automatic explicitly retries "
                "with DirectML. CPU inference-provider fallback stays disabled."
            )
            h_backend.addWidget(QLabel('Windows backend:'))
            h_backend.addWidget(self.combo_backend)
            layout_params.addLayout(h_backend)
        gb_params.setLayout(layout_params)
        layout.addWidget(gb_params)
        gb_linear = QGroupBox('Linear Image Options')
        layout_linear = QVBoxLayout()
        self.chk_prestretch = QCheckBox('Apply PreStretch (for linear/unstretched images)')
        self.chk_prestretch.setChecked(False)
        self.chk_prestretch.toggled.connect(self._on_prestretch_toggled)
        self.chk_prestretch.setToolTip('Enable this if your image is still in linear (unstretched) state.\nLeave OFF for standard non-linear (already stretched) images.')
        layout_linear.addWidget(self.chk_prestretch)
        gb_linear.setLayout(layout_linear)
        layout.addWidget(gb_linear)
        gb_seq = QGroupBox('Sequence Options')
        layout_seq = QVBoxLayout()
        self.chk_seq = QCheckBox('Process Sequence')
        self.layout_prefix = QHBoxLayout()
        self.lbl_prefix = QLabel('Output Prefix:')
        self.txt_prefix = QLineEdit('scunet_')
        if self.is_sequence_context:
            self.chk_seq.setChecked(True)
            self.chk_seq.setEnabled(False)
            self.chk_seq.setText('Sequence Detected (Process as Sequence)')
            self.txt_prefix.setEnabled(True)
        else:
            self.chk_seq.setChecked(False)
            self.chk_seq.setEnabled(False)
            self.chk_seq.setText('Process Sequence (Disabled in Single Image Mode)')
            self.txt_prefix.setEnabled(False)
        layout_seq.addWidget(self.chk_seq)
        self.layout_prefix.addWidget(self.lbl_prefix)
        self.layout_prefix.addWidget(self.txt_prefix)
        layout_seq.addLayout(self.layout_prefix)
        gb_seq.setLayout(layout_seq)
        layout.addWidget(gb_seq)
        self.lbl_status = QLabel('Ready')
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        layout.addWidget(self.lbl_status)
        layout.addWidget(self.progress_bar)
        layout.addSpacing(15)
        btn_layout = QHBoxLayout()
        preview_icon = self.style().standardIcon(QStyle.StandardPixmap.SP_FileDialogDetailedView)
        self.btn_preview = IconTextButton('Open ROI Preview', preview_icon, self)
        self.btn_preview.setProperty('class', 'ROIButton')
        self.btn_preview.setToolTip('Open a preview window to test denoise on a small area.\nIf the window is already open, click again to update the preview with the new settings.')
        self.btn_preview.clicked.connect(self.open_preview)
        btn_layout.addWidget(self.btn_preview)
        btn_layout.addStretch()
        apply_icon = self.style().standardIcon(QStyle.StandardPixmap.SP_DialogApplyButton)
        self.btn_apply = IconTextButton('Apply', apply_icon, self)
        self.btn_apply.setProperty('class', 'accent')
        self.btn_apply.clicked.connect(self.start_processing)
        btn_layout.addWidget(self.btn_apply)
        close_icon = self.style().standardIcon(QStyle.StandardPixmap.SP_DialogCancelButton)
        self.btn_close = IconTextButton('Close', close_icon, self)
        self.btn_close.setProperty('class', 'secondary')
        self.btn_close.clicked.connect(self.handle_close_cancel)
        btn_layout.addWidget(self.btn_close)
        layout.addLayout(btn_layout)

    def toggle_seq_options(self, checked):
        self.txt_prefix.setEnabled(checked)

    def _on_prestretch_toggled(self, checked):
        self.slider_strength.setValue(70 if checked else 50)

    def open_preview(self):
        selected_url = ''
        for rb in self.model_buttons:
            if rb.isChecked():
                selected_url = rb.property('url')
                break
        strength_val = self.slider_strength.value() / 100.0
        preview_params = {
            'model_url': selected_url,
            'strength': strength_val,
            'apply_prestretch': self.chk_prestretch.isChecked(),
            'provider': self.combo_backend.currentData()
            if _USE_ONNX_DIRECTML
            else 'directml',
        }
        if not hasattr(self, 'preview_window') or not self.preview_window.isVisible():
            self.btn_preview.setText('Update ROI Preview')
            self.preview_window = PreviewWindow(self.siril, preview_params)
            self.preview_window.progress_update.connect(self.update_progress)
            self.preview_window.window_closed.connect(self.reset_preview_button)
            self.preview_window.show()
        else:
            self.preview_window.params = preview_params
            self.preview_window.fetch_and_process()
            self.preview_window.gamma_widget.setVisible(preview_params.get('apply_prestretch', False))
            self.btn_preview.setText('Update ROI Preview')

    def reset_preview_button(self):
        self.btn_preview.setText('Open ROI Preview')

    def start_processing(self):
        try:
            self.siril.overlay_clear_polygons()
            self.preview_poly = None
        except Exception:
            pass
        selected_url = ''
        for rb in self.model_buttons:
            if rb.isChecked():
                selected_url = rb.property('url')
                break
        if not selected_url:
            QMessageBox.warning(self, 'Warning', 'Please select a model.')
            return
        strength_val = self.slider_strength.value() / 100.0
        tile_choice = self.combo_tile.currentText()
        tile_param = 'Auto' if tile_choice == 'Auto' else int(tile_choice)
        params = {
            'model_url': selected_url,
            'strength': strength_val,
            'tile_size': tile_param,
            'is_sequence': self.chk_seq.isChecked(),
            'seq_prefix': self.txt_prefix.text(),
            'apply_prestretch': self.chk_prestretch.isChecked(),
            'provider': self.combo_backend.currentData()
            if _USE_ONNX_DIRECTML
            else 'directml',
        }
        self.thread = QThread()
        self.worker = ProcessingWorker(self.siril, params)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress_update.connect(self.update_progress)
        self.worker.error_occurred.connect(self.handle_error)
        self.worker.finished.connect(self.process_finished)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.lbl_status.setText('Starting...')
        self.btn_apply.setEnabled(False)
        self.btn_preview.setEnabled(False)
        self.btn_close.setText('Cancel')
        self.thread.start()

    def update_progress(self, val, text):
        self.progress_bar.setValue(val)
        self.lbl_status.setText(text)

    def handle_error(self, msg):
        QMessageBox.critical(self, 'Processing Error', msg)
        self.lbl_status.setText('Error occurred.')
        self.btn_apply.setEnabled(True)
        self.btn_close.setText('Close')

    def handle_close_cancel(self):
        is_running = False
        if self.worker and self.thread:
            try:
                is_running = self.thread.isRunning()
            except RuntimeError:
                pass
        if is_running:
            self.worker.stop()
            self.btn_close.setText('Cancelling...')
            self.siril.log('User requested cancellation...', s.LogColor.RED)
            self.progress_bar.setValue(0)
        else:
            self.close()

    def process_finished(self):
        self.btn_apply.setEnabled(True)
        self.btn_preview.setEnabled(True)
        self.btn_close.setText('Close')
        self.lbl_status.setText('Completed / Ready')

    def closeEvent(self, event: QCloseEvent):
        if hasattr(self, 'preview_window') and self.preview_window.isVisible():
            self.preview_window.close()
        if self.worker:
            self.worker.stop()
        if self.thread:
            try:
                if self.thread.isRunning():
                    self.thread.quit()
                    self.thread.wait()
            except RuntimeError:
                pass
        try:
            if self.siril:
                self.siril.set_siril_selection(0, 0, 0, 0)
                self.siril.overlay_clear_polygons()
                self.preview_poly = None
                self.siril.log('Window closed. Script cancelled by user.', s.LogColor.BLUE)
                self.siril.disconnect()
        except Exception as e:
            print(f'An error occurred during cleanup: {e}')
        event.accept()

class IconTextButton(QPushButton):

    def __init__(self, text, icon=None, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        layout = QHBoxLayout()
        layout.setContentsMargins(8, 0, 8, 0)
        layout.setSpacing(5)
        icon_label = QLabel()
        if icon:
            pixmap = icon.pixmap(self.iconSize())
            icon_label.setPixmap(pixmap)
        self.text_label = QLabel(text)
        layout.addWidget(icon_label, 0, Qt.AlignmentFlag.AlignVCenter)
        layout.addStretch(1)
        layout.addWidget(self.text_label)
        layout.addStretch(1)
        self.setLayout(layout)

    def setText(self, text):
        if hasattr(self, 'text_label'):
            self.text_label.setText(text)

def main():
    try:
        qapp = QApplication(sys.argv)
        qapp.setApplicationName(f'SCUNet Denoise - v{VERSION}')
        icon_data = base64.b64decode('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')
        pixmap = QPixmap()
        pixmap.loadFromData(icon_data)
        app_icon = QIcon(pixmap)
        qapp.setWindowIcon(app_icon)
        qapp.setStyle('Fusion')
        stylesheet = '\n            QPushButton[class="accent"] {\n                background-color: #3574F0;  /* A nice blue color */\n                color: white;\n                font-weight: bold;\n                border-radius: 4px;\n                padding: 5px;\n                min-width: 90px;\n            }\n            QPushButton[class="accent"]:hover {\n                background-color: #4E8AFC; /* A slightly lighter blue for hover */\n            }\n\n            QPushButton[class="secondary"] {\n                background-color: #e95767; /* Dark gray for secondary actions */\n                color: #dddddd;\n                font-weight: bold;\n                border-radius: 4px;\n                padding: 5px;\n                min-width: 90px;\n            }\n            QPushButton[class="secondary"]:hover {\n                background-color: #d64e5d; /* Slightly lighter on hover */\n                color: white;\n            }\n\n            QPushButton[class="ROIButton"] {\n                background-color: #f0f0f0; /* Light gray */\n                border-radius: 4px;\n                padding: 5px;\n                min-width: 150px;\n            }\n            QPushButton[class="ROIButton"]:hover {\n                background-color: #e0e0e0; /* Darker on hover */\n            }\n            \n            /* Style specifically for the TEXT inside the CUSTOM HELP button */\n            QPushButton[class="ROIButton"] QLabel {\n                color: #005A9C; /* A professional dark blue for readability */\n                font-weight: bold;\n                background-color: transparent; /* Ensure label background is clear */\n                border: none;\n            }\n        '
        qapp.setStyleSheet(stylesheet)
        app = ScunetWindow()
        app.show()
        sys.exit(qapp.exec())
    except Exception as e:
        print(f'Error initializing application: {str(e)}')
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
