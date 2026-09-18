#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Panagiotis Stefos
# Project and support: https://github.com/MadeOfNebulas/Aster
"""
Siril Star Glow — interactive enhancement for linear or stretched star layers.

Designed for Siril 1.4.x and sirilpy.
The currently loaded image should be a linear or nonlinear stars-only image.

Features
--------
- Reads the current Siril image directly (16-bit integer or 32-bit float).
- Non-linear mode preserves the established v1.2.0 detection and Screen workflow.
- Linear mode temporarily applies Rational LogD and uses the established effects.
- Reverse Rational LogD returns the pushed result to the linear domain.
- Optional processed starless background for preview-only recomposition.
- Blur choices: Gaussian, Box, Disk, Multi-scale Gaussian.
- Glow radius, selected-star diameter, glow strength and glow gamma controls.
- Optional red mask overlay.
- Pushes the full-resolution result directly back into Siril.

License: GPL-3.0-or-later
"""

from __future__ import annotations

import sys
import traceback
from dataclasses import dataclass

import numpy as np
import sirilpy as s

s.ensure_installed("scipy")
s.ensure_installed("PyQt6")
s.ensure_installed("astropy")

from scipy import ndimage as ndi
from scipy import signal
from astropy.io import fits
from PyQt6.QtCore import QEvent, QPoint, QRectF, QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPainter, QPen, QPixmap, QPolygon
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)


APP_TITLE = "Aster"
# Aster 1.3.0 — linear and non-linear star-mask support
PREVIEW_MAX = 1600
EMITTER_CACHE_MAX_ENTRIES = 8
_EMITTER_CACHE = {}
CATALOG_CACHE_MAX_ENTRIES = 4
_CATALOG_CACHE = {}


@dataclass
class ImageScale:
    black: float
    white: float
    original_dtype: np.dtype


@dataclass
class StarCatalog:
    """Shared detection result used by glow, primary and secondary spikes."""

    source_rgb: np.ndarray
    lum: np.ndarray
    detection_lum: np.ndarray
    labels: np.ndarray
    diameters: np.ndarray
    peaks: np.ndarray
    component_slices: list
    background: float
    linear_mode: bool = False
    snr: np.ndarray | None = None
    fluxes: np.ndarray | None = None


def build_star_catalog(
    rgb: np.ndarray,
    black_point: float,
    white_point: float,
) -> StarCatalog:
    """Detect components once on a temporary levels-adjusted luminance copy."""
    black_point = float(np.clip(black_point, 0.0, 0.99))
    white_point = float(np.clip(white_point, black_point + 0.01, 1.0))
    key = (
        id(rgb),
        rgb.shape,
        round(black_point, 4),
        round(white_point, 4),
    )
    cached = _CATALOG_CACHE.get(key)
    if cached is not None:
        return cached

    lum = luminance(rgb)
    detection_lum = np.clip(
        (lum - black_point) / (white_point - black_point),
        0.0,
        1.0,
    ).astype(np.float32)
    background = float(np.median(lum))

    # A small post-levels floor makes both points meaningful: black clips the
    # faint field and white controls how quickly surviving cores rise above
    # the detector floor. No dilation is used, so nearby halos are not joined.
    candidates = detection_lum > 0.05
    candidates = ndi.binary_opening(
        candidates, structure=np.ones((2, 2), dtype=bool)
    )
    labels, count = ndi.label(candidates)
    if count == 0:
        diameters = np.zeros(1, dtype=np.float32)
        peaks = np.zeros(1, dtype=np.float32)
        component_slices = []
    else:
        areas = np.bincount(labels.ravel())
        diameters = (2.0 * np.sqrt(areas / np.pi)).astype(np.float32)
        component_ids = np.arange(1, count + 1)
        peaks = np.zeros(count + 1, dtype=np.float32)
        peaks[1:] = ndi.maximum(lum, labels, component_ids)
        component_slices = ndi.find_objects(labels)

    catalog = StarCatalog(
        source_rgb=rgb,
        lum=lum,
        detection_lum=detection_lum,
        labels=labels,
        diameters=diameters,
        peaks=peaks,
        component_slices=component_slices,
        background=background,
    )
    if len(_CATALOG_CACHE) >= CATALOG_CACHE_MAX_ENTRIES:
        _CATALOG_CACHE.clear()
    _CATALOG_CACHE[key] = catalog
    return catalog


def build_linear_star_catalog(rgb: np.ndarray) -> StarCatalog:
    """Detect and deblend stars from untouched linear pixels."""
    key = ("linear", id(rgb), rgb.shape)
    cached = _CATALOG_CACHE.get(key)
    if cached is not None:
        return cached

    lum = luminance(rgb)
    background = float(np.median(lum))
    low_sample = lum[lum <= np.percentile(lum, 70.0)]
    if low_sample.size:
        low_median = float(np.median(low_sample))
        mad = float(np.median(np.abs(low_sample - low_median)))
    else:
        mad = 0.0
    sigma = 1.4826 * mad
    peak_signal = max(float(np.max(lum)) - background, 0.0)
    sigma = max(sigma, peak_signal * 1e-6, 1e-8)
    threshold = background + max(4.0 * sigma, peak_signal * 1e-5)
    candidates = lum > threshold

    initial_labels, initial_count = ndi.label(candidates)
    if initial_count == 0:
        labels = np.zeros_like(initial_labels, dtype=np.int32)
        count = 0
    else:
        # Split touching stars at distinct local maxima. Work per bounding box
        # so dense fields do not require repeated full-image allocations.
        labels = np.zeros_like(initial_labels, dtype=np.int32)
        next_id = 1
        for component_id, component_slice in enumerate(
            ndi.find_objects(initial_labels), start=1
        ):
            if component_slice is None:
                continue
            local_labels = initial_labels[component_slice]
            component = local_labels == component_id
            local_lum = lum[component_slice]
            component_peak = float(np.max(local_lum[component]))
            peak_floor = background + max(
                5.0 * sigma, 0.18 * (component_peak - background)
            )
            maxima = (
                component
                & (local_lum == ndi.maximum_filter(local_lum, size=3))
                & (local_lum >= peak_floor)
            )
            peak_labels, peak_count = ndi.label(maxima)
            if peak_count <= 1:
                labels_crop = labels[component_slice]
                labels_crop[component] = next_id
                next_id += 1
                continue

            peak_centres = np.asarray(
                ndi.center_of_mass(
                    local_lum,
                    peak_labels,
                    np.arange(1, peak_count + 1),
                ),
                dtype=np.float32,
            )
            yy, xx = np.nonzero(component)
            distances = (
                np.square(yy[:, None] - peak_centres[None, :, 0])
                + np.square(xx[:, None] - peak_centres[None, :, 1])
            )
            owners = np.argmin(distances, axis=1)
            labels_crop = labels[component_slice]
            labels_crop[yy, xx] = next_id + owners
            next_id += peak_count
        count = next_id - 1

    if count == 0:
        diameters = np.zeros(1, dtype=np.float32)
        peaks = np.zeros(1, dtype=np.float32)
        fluxes = np.zeros(1, dtype=np.float32)
        snr = np.zeros(1, dtype=np.float32)
        component_slices = []
    else:
        component_ids = np.arange(1, count + 1)
        peaks = np.zeros(count + 1, dtype=np.float32)
        peaks[1:] = ndi.maximum(lum, labels, component_ids)
        fluxes = np.zeros(count + 1, dtype=np.float32)
        fluxes[1:] = ndi.sum(
            np.clip(lum - background, 0.0, None), labels, component_ids
        )
        snr = np.zeros(count + 1, dtype=np.float32)
        snr[1:] = np.clip((peaks[1:] - background) / sigma, 0.0, None)
        component_slices = ndi.find_objects(labels)
        diameters = np.zeros(count + 1, dtype=np.float32)
        for component_id, component_slice in enumerate(
            component_slices, start=1
        ):
            if component_slice is None:
                continue
            component = labels[component_slice] == component_id
            local_lum = lum[component_slice]
            # Apparent linear diameter is the equivalent diameter of the
            # deblended above-noise footprint. This keeps the existing Aster
            # diameter control meaningful while flux/SNR independently govern
            # whether a weak object is eligible for an effect.
            apparent_area = np.count_nonzero(component)
            diameters[component_id] = 2.0 * np.sqrt(apparent_area / np.pi)

    detection_lum = np.clip(
        (lum - background) / max(peak_signal, 1e-8), 0.0, 1.0
    ).astype(np.float32)
    catalog = StarCatalog(
        source_rgb=rgb,
        lum=lum,
        detection_lum=detection_lum,
        labels=labels,
        diameters=diameters,
        peaks=peaks,
        component_slices=component_slices,
        background=background,
        linear_mode=True,
        snr=snr,
        fluxes=fluxes,
    )
    if len(_CATALOG_CACHE) >= CATALOG_CACHE_MAX_ENTRIES:
        _CATALOG_CACHE.clear()
    _CATALOG_CACHE[key] = catalog
    return catalog


def component_selection(
    catalog: StarCatalog,
    min_diameter: float,
    feather: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-component and per-pixel smooth diameter weights."""
    half_feather = max(float(feather), 0.1) * 0.5
    low = max(0.0, float(min_diameter) - half_feather)
    high = max(low + 1e-6, float(min_diameter) + half_feather)
    t = np.clip((catalog.diameters - low) / (high - low), 0.0, 1.0)
    weights = (t * t * (3.0 - 2.0 * t)).astype(np.float32)
    if catalog.linear_mode and catalog.snr is not None:
        # A second soft gate makes physically weak detections fade instead of
        # receiving the same spikes as high-SNR stars of similar diameter.
        signal_t = np.clip((catalog.snr - 3.0) / 5.0, 0.0, 1.0)
        signal_weight = signal_t * signal_t * (3.0 - 2.0 * signal_t)
        weights *= signal_weight.astype(np.float32)
    if weights.size:
        weights[0] = 0.0
    selection = weights[catalog.labels].astype(np.float32)
    selection = ndi.maximum_filter(selection, size=3, mode="nearest")
    return weights, selection


def canonical_hwc(data: np.ndarray) -> tuple[np.ndarray, str]:
    """Return image in H×W×C form and remember the original layout."""
    arr = np.asarray(data)

    if arr.ndim == 2:
        return arr[..., None], "HW"

    if arr.ndim != 3:
        raise ValueError(f"Unsupported image shape: {arr.shape}")

    # Siril normally exposes H×W×C. This fallback also accepts C×H×W.
    if arr.shape[-1] in (1, 3, 4):
        return arr, "HWC"
    if arr.shape[0] in (1, 3, 4):
        return np.moveaxis(arr, 0, -1), "CHW"

    raise ValueError(f"Cannot determine channel axis for image shape: {arr.shape}")


def restore_layout(data_hwc: np.ndarray, layout: str) -> np.ndarray:
    if layout == "HW":
        return data_hwc[..., 0]
    if layout == "CHW":
        return np.moveaxis(data_hwc, -1, 0)
    return data_hwc


def apply_orientation(data: np.ndarray, mode: str) -> np.ndarray:
    """Apply a self-inverse orientation mapping for preview and Siril I/O."""
    if mode == "Flip vertical":
        return np.flip(data, axis=0).copy()
    if mode == "Flip horizontal":
        return np.flip(data, axis=1).copy()
    if mode == "Rotate 180°":
        return np.flip(data, axis=(0, 1)).copy()
    return np.array(data, copy=True)


def normalize_image(data_hwc: np.ndarray) -> tuple[np.ndarray, ImageScale]:
    """
    Convert linearly to float32 in Siril's 0..1 working range.

    No histogram stretch, percentile rescaling, gamma, or automatic black-point
    adjustment is applied. Float images are expected to use Siril's normal 0..1
    pixel range; 16-bit integer images are divided by their dtype maximum.
    """
    arr = data_hwc.astype(np.float32, copy=False)
    dtype = data_hwc.dtype

    if not np.isfinite(arr).any():
        raise ValueError("The loaded image contains no finite pixels.")

    if np.issubdtype(dtype, np.integer):
        black = 0.0
        white = float(np.iinfo(dtype).max)
    else:
        black = 0.0
        white = 1.0

    norm = np.nan_to_num((arr - black) / (white - black), nan=0.0, posinf=1.0, neginf=0.0)
    norm = np.clip(norm, 0.0, 1.0)
    return norm.astype(np.float32), ImageScale(black, white, dtype)


def denormalize_image(norm: np.ndarray, scale: ImageScale) -> np.ndarray:
    arr = np.clip(norm, 0.0, 1.0) * (scale.white - scale.black) + scale.black
    if np.issubdtype(scale.original_dtype, np.integer):
        info = np.iinfo(scale.original_dtype)
        arr = np.clip(np.rint(arr), info.min, info.max)
    return arr.astype(scale.original_dtype)


def rational_logd_preview(rgb: np.ndarray, logd: float) -> np.ndarray:
    """VeraLux-style bounded Rational LogD curve."""
    x = np.clip(rgb, 0.0, 1.0).astype(np.float32, copy=False)
    logd = float(np.clip(logd, 1.0, 21.0))
    stretch_factor = np.clip((logd - 1.0) / 2.0, 0.0, 12.0)
    k = float(np.power(3.0, stretch_factor))
    denominator = (k - 1.0) * x + 1.0
    return np.clip((k * x) / denominator, 0.0, 1.0).astype(np.float32)


def inverse_rational_logd(rgb: np.ndarray, logd: float) -> np.ndarray:
    """Exact inverse of the neutral-profile Rational LogD curve."""
    y = np.clip(rgb, 0.0, 1.0).astype(np.float32, copy=False)
    logd = float(np.clip(logd, 1.0, 21.0))
    stretch_factor = np.clip((logd - 1.0) / 2.0, 0.0, 12.0)
    k = float(np.power(3.0, stretch_factor))
    denominator = k - (k - 1.0) * y
    return np.clip(
        y / np.maximum(denominator, 1e-12), 0.0, 1.0
    ).astype(np.float32)


def luminance(rgb: np.ndarray) -> np.ndarray:
    if rgb.shape[2] == 1:
        return rgb[..., 0]
    return (
        0.2126 * rgb[..., 0]
        + 0.7152 * rgb[..., 1]
        + 0.0722 * rgb[..., 2]
    ).astype(np.float32)


def disk_kernel(radius: int) -> np.ndarray:
    radius = max(1, int(radius))
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    kernel = (xx * xx + yy * yy <= radius * radius).astype(np.float32)
    kernel /= max(float(kernel.sum()), 1.0)
    return kernel


def moffat_kernel(radius: float) -> np.ndarray:
    """Photographic Moffat PSF with softer wings than a Gaussian."""
    alpha = max(float(radius), 0.5)
    extent = max(1, int(round(alpha * 3.0)))
    yy, xx = np.ogrid[-extent : extent + 1, -extent : extent + 1]
    kernel = np.power(1.0 + (xx * xx + yy * yy) / (alpha * alpha), -2.5)
    kernel = kernel.astype(np.float32)
    kernel /= max(float(kernel.sum()), 1e-12)
    return kernel


def blur_mask(mask: np.ndarray, radius: float, mode: str) -> np.ndarray:
    radius = max(0.1, float(radius))

    if mode == "Gaussian":
        return ndi.gaussian_filter(mask, sigma=radius, mode="reflect")

    if mode == "Box":
        size = max(1, int(round(radius * 2.0 + 1.0)))
        return ndi.uniform_filter(mask, size=size, mode="reflect")

    if mode == "Disk":
        kernel = disk_kernel(max(1, int(round(radius))))
        return ndi.convolve(mask, kernel, mode="reflect")

    if mode == "Triangle":
        size = max(1, int(round(radius + 1.0)))
        first = ndi.uniform_filter(mask, size=size, mode="reflect")
        return ndi.uniform_filter(first, size=size, mode="reflect")

    if mode == "Moffat":
        kernel = moffat_kernel(radius)
        pad_y = kernel.shape[0] // 2
        pad_x = kernel.shape[1] // 2
        padded = np.pad(mask, ((pad_y, pad_y), (pad_x, pad_x)), mode="reflect")
        convolved = signal.fftconvolve(padded, kernel, mode="same")
        return convolved[
            pad_y : pad_y + mask.shape[0],
            pad_x : pad_x + mask.shape[1],
        ].astype(np.float32)

    # A broader, more photographic halo: two Gaussian scales mixed together.
    tight = ndi.gaussian_filter(mask, sigma=radius * 0.65, mode="reflect")
    broad = ndi.gaussian_filter(mask, sigma=radius * 1.8, mode="reflect")
    return 0.65 * tight + 0.35 * broad


def select_stars(
    rgb: np.ndarray,
    min_diameter: float,
    feather: float,
    catalog: StarCatalog,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Detect bright connected objects and weight them smoothly by equivalent
    diameter. Stars around the cutoff fade continuously instead of switching
    abruptly on or off.
    """
    lum = catalog.lum
    _, selection_weight = component_selection(
        catalog, min_diameter, feather
    )

    # Keep actual star intensity so bright stars naturally produce stronger glow.
    intensity = np.clip(
        (lum - catalog.background) / max(1.0 - catalog.background, 1e-6),
        0.0,
        1.0,
    )
    intensity_mask = intensity * selection_weight

    return intensity_mask.astype(np.float32), selection_weight


def make_glow(
    base_rgb: np.ndarray,
    min_diameter: float,
    feather: float,
    radius: float,
    strength: float,
    gamma: float,
    blur_mode: str,
    blend: float,
    catalog: StarCatalog,
    exclusion_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    star_mask, selection_weight = select_stars(
        base_rgb, min_diameter, feather, catalog
    )
    if exclusion_mask is not None:
        allowed = 1.0 - np.clip(exclusion_mask, 0.0, 1.0)
        star_mask *= allowed
        selection_weight *= allowed
    blurred = blur_mask(star_mask, radius, blur_mode)

    peak = float(blurred.max())
    if peak > 0:
        blurred /= peak

    # Gamma below 1 spreads and lifts the faint halo; above 1 tightens it.
    glow_alpha = np.power(np.clip(blurred, 0.0, 1.0), max(gamma, 0.05))
    glow_alpha = np.clip(glow_alpha * strength, 0.0, 1.0)

    if base_rgb.shape[2] == 1:
        glow_rgb = glow_alpha[..., None]
    else:
        # Preserve a little of the star colour while keeping the halo smooth.
        weighted = base_rgb * star_mask[..., None]
        colour = ndi.gaussian_filter(
            weighted,
            sigma=(max(radius, 0.5), max(radius, 0.5), 0.0),
            mode="reflect",
        )
        colour_sum = colour.sum(axis=2, keepdims=True)
        colour = np.divide(
            colour,
            colour_sum,
            out=np.full_like(colour, 1.0 / 3.0),
            where=colour_sum > 1e-6,
        )
        colour *= 3.0
        glow_rgb = np.clip(colour * glow_alpha[..., None], 0.0, 1.0)

    # Screen blend: 1 - (1 - base) * (1 - glow)
    result = 1.0 - (1.0 - base_rgb) * (1.0 - glow_rgb)
    blend = float(np.clip(blend, 0.01, 1.0))
    result = base_rgb + (result - base_rgb) * blend
    if exclusion_mask is not None:
        excluded = np.clip(exclusion_mask, 0.0, 1.0)[..., None]
        result = result * (1.0 - excluded) + base_rgb * excluded
    return np.clip(result, 0.0, 1.0), selection_weight


def make_glow_linear(
    base_rgb: np.ndarray,
    min_diameter: float,
    feather: float,
    radius: float,
    strength: float,
    gamma: float,
    blur_mode: str,
    blend: float,
    catalog: StarCatalog,
    exclusion_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate additive glow while keeping the starmask genuinely linear."""
    star_mask, selection_weight = select_stars(
        base_rgb, min_diameter, feather, catalog
    )
    if exclusion_mask is not None:
        allowed = 1.0 - np.clip(exclusion_mask, 0.0, 1.0)
        star_mask *= allowed
        selection_weight *= allowed
    blurred = np.clip(blur_mask(star_mask, radius, blur_mode), 0.0, 1.0)
    # Gamma shapes the synthetic halo, but no display stretch or peak
    # normalization is baked into the returned linear image.
    glow_signal = np.power(blurred, max(gamma, 0.05)) * strength

    if base_rgb.shape[2] == 1:
        glow_rgb = glow_signal[..., None]
    else:
        weighted = base_rgb * selection_weight[..., None]
        colour = ndi.gaussian_filter(
            weighted,
            sigma=(max(radius, 0.5), max(radius, 0.5), 0.0),
            mode="reflect",
        )
        colour_sum = colour.sum(axis=2, keepdims=True)
        colour = np.divide(
            colour,
            colour_sum,
            out=np.full_like(colour, 1.0 / 3.0),
            where=colour_sum > 1e-9,
        )
        glow_rgb = np.clip(colour * 3.0 * glow_signal[..., None], 0.0, 1.0)

    blend = float(np.clip(blend, 0.01, 1.0))
    result = np.clip(base_rgb + glow_rgb * blend, 0.0, 1.0)
    if exclusion_mask is not None:
        excluded = np.clip(exclusion_mask, 0.0, 1.0)[..., None]
        result = result * (1.0 - excluded) + base_rgb * excluded
    return result.astype(np.float32), selection_weight


def _detect_star_emitters_uncached(
    rgb: np.ndarray,
    min_diameter: float,
    feather: float,
    catalog: StarCatalog,
) -> tuple[
    list[tuple[float, float, float, np.ndarray, float, float]],
    np.ndarray,
]:
    """
    Return one point emitter per selected star.

    Crucially, the emitter is independent of the star-core diameter. A large
    star therefore produces a brighter line, never a thicker line.
    """
    lum = catalog.lum
    background = catalog.background
    labels = catalog.labels
    diameters = catalog.diameters
    weights, selection_weight = component_selection(
        catalog, min_diameter, feather
    )
    selected_ids = np.flatnonzero(weights > 1e-4)
    selected_ids = selected_ids[selected_ids != 0]
    if selected_ids.size == 0:
        return [], selection_weight

    peaks = catalog.peaks[selected_ids]
    component_slices = catalog.component_slices
    emitters = []
    for component_id, peak in zip(selected_ids, peaks):
        component_slice = component_slices[int(component_id) - 1]
        if component_slice is None:
            continue

        # Expand the exact component bounding box so local morphology has the
        # same border context as the former full-image implementation.
        padding = 2
        y0 = max(0, component_slice[0].start - padding)
        y1 = min(labels.shape[0], component_slice[0].stop + padding)
        x0 = max(0, component_slice[1].start - padding)
        x1 = min(labels.shape[1], component_slice[1].stop + padding)
        local_slice = (slice(y0, y1), slice(x0, x1))

        labels_crop = labels[local_slice]
        component = labels_crop == component_id
        lum_crop = lum[local_slice]

        # Temporarily boost contrast until the stellar core becomes a solid
        # ball with a clear border. The spike origin is the geometric centre
        # of that ball, not the brightness-weighted centre of its halo.
        normalized = np.clip(
            (lum_crop - background) / max(float(peak) - background, 1e-6),
            0.0,
            1.0,
        )
        # sigmoid(normalized, midpoint=0.65) >= 0.5 is mathematically
        # equivalent to normalized >= 0.65, without allocating another array.
        core_ball = component & (normalized >= 0.65)
        core_ball = ndi.binary_closing(
            core_ball,
            structure=np.ones((3, 3), dtype=bool),
        )
        core_ball = ndi.binary_fill_holes(core_ball) & component

        # If noise created more than one island, keep only the largest compact
        # region, which represents the actual stellar core.
        ball_labels, ball_count = ndi.label(core_ball)
        if ball_count > 1:
            ball_areas = np.bincount(ball_labels.ravel())
            ball_areas[0] = 0
            core_ball = ball_labels == int(np.argmax(ball_areas))
        if not np.any(core_ball):
            core_ball = component

        ball_area = int(np.count_nonzero(core_ball))
        if ball_area == 0:
            continue
        core_y, core_x = np.nonzero(core_ball)
        cy = float(np.mean(core_y)) + y0
        cx = float(np.mean(core_x)) + x0
        iy = int(np.clip(round(cy), 0, rgb.shape[0] - 1))
        ix = int(np.clip(round(cx), 0, rgb.shape[1] - 1))
        brightness = (
            np.clip(
                (float(peak) - background) / max(1.0 - background, 1e-6),
                0.0,
                1.0,
            )
            * float(weights[component_id])
        )
        rgb_crop = rgb[local_slice]
        colour_sample = (
            component
            & (lum_crop >= background + 0.15 * (float(peak) - background))
            & (lum_crop <= background + 0.85 * (float(peak) - background))
        )
        if np.count_nonzero(colour_sample) < 3:
            colour_sample = component
        sample_weights = np.clip(
            lum_crop[colour_sample] - background, 1e-6, None
        )
        sampled_colour = np.average(
            rgb_crop[colour_sample],
            axis=0,
            weights=sample_weights,
        ).astype(np.float32)
        colour_peak = max(float(np.max(sampled_colour)), 1e-6)
        colour = np.clip(
            sampled_colour / colour_peak * brightness, 0.0, 1.0
        )
        emitters.append(
            (
                cy,
                cx,
                brightness,
                colour,
                float(diameters[component_id]),
                float(weights[component_id]),
            )
        )
    return emitters, selection_weight


def detect_star_emitters(
    rgb: np.ndarray,
    min_diameter: float,
    feather: float,
    catalog: StarCatalog,
) -> tuple[
    list[tuple[float, float, float, np.ndarray, float, float]],
    np.ndarray,
]:
    """Reuse one bounding-box emitter analysis for every spike selection."""
    key = (
        id(rgb),
        rgb.shape,
        id(catalog),
    )
    analysis_floor = max(
        0.0, float(min_diameter) - max(float(feather), 0.1) * 0.5
    )
    cache_entry = _EMITTER_CACHE.get(key)
    if cache_entry is None or cache_entry[0] > analysis_floor:
        cached = _detect_star_emitters_uncached(
            rgb, analysis_floor, 0.1, catalog
        )
        if len(_EMITTER_CACHE) >= EMITTER_CACHE_MAX_ENTRIES:
            _EMITTER_CACHE.clear()
        _EMITTER_CACHE[key] = (analysis_floor, cached)
    else:
        cached = cache_entry[1]
    base_emitters, _unused_selection = cached
    _, selection_weight = component_selection(
        catalog, min_diameter, feather
    )
    half_feather = max(float(feather), 0.1) * 0.5
    low = max(0.0, float(min_diameter) - half_feather)
    high = max(low + 1e-6, float(min_diameter) + half_feather)
    emitters = []
    for cy, cx, brightness, colour, diameter, _base_weight in base_emitters:
        t = float(np.clip((diameter - low) / (high - low), 0.0, 1.0))
        weight = t * t * (3.0 - 2.0 * t)
        if weight <= 1e-4:
            continue
        emitters.append(
            (
                cy,
                cx,
                brightness * weight,
                colour * weight,
                diameter,
                weight,
            )
        )
    # The caller may apply an exclusion mask in-place.
    return emitters, selection_weight.copy()


def connected_line_pixels(
    x0: int,
    y0: int,
    x1: int,
    y1: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return every pixel of an 8-connected Bresenham line."""
    points_x = []
    points_y = []
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    error = dx + dy
    while True:
        points_x.append(x0)
        points_y.append(y0)
        if x0 == x1 and y0 == y1:
            break
        doubled_error = 2 * error
        if doubled_error >= dy:
            error += dy
            x0 += sx
        if doubled_error <= dx:
            error += dx
            y0 += sy
    return np.asarray(points_x, dtype=int), np.asarray(points_y, dtype=int)


def draw_point_emitter_spikes(
    shape: tuple[int, int, int],
    emitters: list[
        tuple[float, float, float, np.ndarray, float, float]
    ],
    length: float,
    width: float,
    angle_degrees: float,
    spike_type: str = "Newtonian",
    spectral: bool = False,
    spectral_strength: float = 0.55,
    spectral_position: float = 0.28,
    spectral_spread: float = 0.18,
    spectral_saturation: float = 0.80,
    spectral_smoothness: float = 0.60,
    per_arm_variation: float = 0.10,
    linear_photometry: bool = False,
    progressive_linear_profile: bool = False,
) -> np.ndarray:
    """Draw four crisp, one-pixel lines from each point emitter."""
    height, image_width, channels = shape
    canvas = np.zeros(shape, dtype=np.float32)
    length = max(float(length), 2.0)
    width = max(float(width), 0.0)
    base_angle = np.deg2rad(float(angle_degrees))
    largest_diameter = max(
        (emitter[4] for emitter in emitters),
        default=1.0,
    )
    largest_flux_proxy = max(
        (emitter[2] * emitter[4] * emitter[4] for emitter in emitters),
        default=1.0,
    )
    if spike_type == "JWST":
        # Six long mirror spikes at 60-degree intervals, positioned at
        # roughly 12/2/4/6/8/10 o'clock, plus two shorter horizontal strut
        # spikes at 3 and 9 o'clock.
        arm_specs = [
            (np.pi / 6.0 + arm * np.pi / 3.0, 1.0, 1.0, True)
            for arm in range(6)
        ]
        arm_specs.extend(
            [
                (0.0, 0.42, 0.45, False),
                (np.pi, 0.42, 0.45, False),
            ]
        )
    else:
        arm_specs = [
            (arm * np.pi / 2.0, 1.0, 1.0, True)
            for arm in range(4)
        ]

    for cy, cx, brightness, colour, diameter, selection_factor in emitters:
        diameter_ratio = np.clip(
            diameter / max(largest_diameter, 1e-6),
            0.0,
            1.0,
        )
        # The length control is the maximum produced by the largest selected
        # star. Smaller stars scale directly from their measured diameter.
        if linear_photometry:
            flux_ratio = np.clip(
                (brightness * diameter * diameter)
                / max(largest_flux_proxy, 1e-9),
                0.0,
                1.0,
            )
            size_factor = max(
                0.04,
                float(
                    np.power(diameter_ratio, 0.35)
                    * np.power(flux_ratio, 0.65)
                ),
            )
        else:
            size_factor = max(0.08, float(np.power(diameter_ratio, 0.85)))
        feather_length = np.power(
            np.clip(selection_factor, 0.0, 1.0),
            0.35,
        )
        emitter_length = max(
            2.0,
            length * size_factor * feather_length,
        )
        source_colour = (
            np.array([brightness], dtype=np.float32)
            if channels == 1
            else np.asarray(colour[:channels], dtype=np.float32)
        )
        spectral_star_factor = float(
            np.power(diameter_ratio, 1.20)
            * np.power(np.clip(selection_factor, 0.0, 1.0), 0.50)
        )
        for arm, (
            theta_offset,
            arm_length_scale,
            arm_intensity_scale,
            arm_has_spectrum,
        ) in enumerate(arm_specs):
            theta = base_angle + theta_offset
            arm_length = max(2.0, emitter_length * arm_length_scale)
            # The user-selected length remains the full useful spike length in
            # both modes. Draw a short extra section only for the terminal
            # fade, rather than consuming part of the length control.
            rendered_arm_length = arm_length * 1.18
            end_x = int(round(cx + np.cos(theta) * rendered_arm_length))
            end_y = int(round(cy + np.sin(theta) * rendered_arm_length))
            xx, yy = connected_line_pixels(
                int(round(cx)),
                int(round(cy)),
                end_x,
                end_y,
            )
            valid = (
                (xx >= 0)
                & (xx < image_width)
                & (yy >= 0)
                & (yy < height)
            )
            if not np.any(valid):
                continue
            distances = np.hypot(xx - cx, yy - cy).astype(np.float32)
            normalized_distance = np.clip(
                distances / max(arm_length, 1e-6), 0.0, None
            )
            # Use the same optical radial profile for linear and non-linear
            # data: a bright compact inner spike followed by a weak
            # power-law diffraction tail. Linear mode still differs only in
            # its photometry, compositing and later inverse LogD conversion.
            inner = 0.90 * np.exp(-normalized_distance / 0.21)
            tail = 0.10 * np.power(
                1.0 + normalized_distance / 0.15, -1.28
            )
            # Hold the radial profile through the complete requested length,
            # then taper across the additional 18 percent.
            edge_taper = np.clip(
                (1.18 - normalized_distance) / 0.18, 0.0, 1.0
            )
            edge_taper = edge_taper * edge_taper * (
                3.0 - 2.0 * edge_taper
            )
            falloff = (inner + tail) * edge_taper

            # Keep the section crossing the stellar core optically solid in
            # both input modes. A broad quintic transition avoids a visible
            # hard cross on smaller stars while blending continuously into
            # the mode-specific outer-spike falloff.
            solid_radius = max(0.75, float(diameter) * 0.30)
            solid_transition = max(5.0, float(diameter) * 1.20)
            solid_mix = np.clip(
                1.0
                - (distances - solid_radius) / solid_transition,
                0.0,
                1.0,
            )
            solid_mix = solid_mix**3 * (
                10.0 - 15.0 * solid_mix + 6.0 * solid_mix**2
            )
            falloff = falloff * (1.0 - solid_mix) + solid_mix
            falloff *= arm_intensity_scale
            spectral_mix = None
            spectral_rgb = None
            if spectral and arm_has_spectrum and channels >= 3:
                position = np.clip(
                    distances / max(arm_length, 1e-6), 0.0, 1.0
                )
                variation_wave = np.sin(
                    (arm + 1) * 1.618
                    + cx * 0.013
                    + cy * 0.017
                )
                local_position = np.clip(
                    spectral_position
                    + variation_wave * per_arm_variation * 0.08,
                    0.05,
                    0.95,
                )
                local_strength = np.clip(
                    spectral_strength
                    * (1.0 + variation_wave * per_arm_variation * 0.25),
                    0.0,
                    1.0,
                )
                sigma = max(
                    spectral_spread
                    * (0.22 + 0.78 * spectral_smoothness)
                    * 0.5,
                    0.005,
                )
                spectral_mix = (
                    local_strength
                    * spectral_star_factor
                    * np.exp(
                        -0.5
                        * np.square((position - local_position) / sigma)
                    )
                )
                phase = np.clip(
                    (
                        position
                        - (local_position - spectral_spread * 0.5)
                    )
                    / max(spectral_spread, 1e-6),
                    0.0,
                    1.0,
                )
                rainbow = np.stack(
                    (
                        # Blue near the core, red toward the outer spike.
                        np.clip(3.0 * phase - 1.5, 0.0, 1.0),
                        np.clip(1.0 - 3.0 * np.abs(phase - 0.5), 0.0, 1.0),
                        np.clip(1.5 - 3.0 * phase, 0.0, 1.0),
                    ),
                    axis=1,
                ).astype(np.float32)
                hardness = 4.0 - 3.3 * spectral_smoothness
                rainbow = np.power(np.clip(rainbow, 0.0, 1.0), hardness)
                spectral_rgb = (
                    (1.0 - spectral_saturation)
                    + spectral_saturation * rainbow
                )
            for channel in range(channels):
                channel_value = np.full_like(
                    falloff, source_colour[channel]
                )
                if solid_mix is not None:
                    # Neutralize only the solid inner section toward the
                    # brightest sampled core channel. This makes the centre
                    # visually dense without whitening the coloured tail.
                    core_level = float(np.max(source_colour))
                    channel_value = (
                        channel_value * (1.0 - solid_mix)
                        + core_level * solid_mix
                    )
                if spectral_mix is not None and channel < 3:
                    spectral_value = brightness * (
                        0.20 + 0.80 * spectral_rgb[:, channel]
                    )
                    channel_value = (
                        channel_value * (1.0 - spectral_mix)
                        + spectral_value * spectral_mix
                    )
                np.maximum.at(
                    canvas[..., channel],
                    (yy[valid], xx[valid]),
                    falloff[valid] * channel_value[valid],
                )

    # Width is a fixed optical line width, entirely independent of star size.
    if width > 0.05:
        for channel in range(channels):
            original_peak = float(canvas[..., channel].max())
            widened = ndi.gaussian_filter(
                canvas[..., channel],
                sigma=width,
                mode="constant",
            )
            widened_peak = float(widened.max())
            if original_peak > 0 and widened_peak > 0:
                widened *= original_peak / widened_peak
            canvas[..., channel] = widened
    return np.clip(canvas, 0.0, 1.0)


def make_newtonian_spikes(
    source_rgb: np.ndarray,
    composite_rgb: np.ndarray,
    min_diameter: float,
    feather: float,
    strength: float,
    length: float,
    width: float,
    angle: float,
    blend: float,
    spike_type: str,
    spectral: bool,
    spectral_strength: float,
    spectral_position: float,
    spectral_spread: float,
    spectral_saturation: float,
    spectral_smoothness: float,
    per_arm_variation: float,
    catalog: StarCatalog,
    exclusion_mask: np.ndarray | None = None,
    linear_output: bool = False,
    progressive_linear_profile: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    emitters, selection_weight = detect_star_emitters(
        source_rgb, min_diameter, feather, catalog
    )
    if exclusion_mask is not None:
        allowed = 1.0 - np.clip(exclusion_mask, 0.0, 1.0)
        selection_weight *= allowed
        emitters = [
            emitter
            for emitter in emitters
            if allowed[
                int(np.clip(round(emitter[0]), 0, allowed.shape[0] - 1)),
                int(np.clip(round(emitter[1]), 0, allowed.shape[1] - 1)),
            ]
            > 0.5
        ]
    if not emitters:
        return composite_rgb.copy(), selection_weight
    spike_rgb = draw_point_emitter_spikes(
        source_rgb.shape,
        emitters,
        length,
        width,
        angle,
        spike_type,
        spectral,
        spectral_strength,
        spectral_position,
        spectral_spread,
        spectral_saturation,
        spectral_smoothness,
        per_arm_variation,
        linear_output,
        progressive_linear_profile,
    )
    spike_rgb = np.clip(spike_rgb * strength, 0.0, 1.0)

    blend = float(np.clip(blend, 0.01, 1.0))
    if linear_output:
        result = np.clip(composite_rgb + spike_rgb * blend, 0.0, 1.0)
    else:
        result = 1.0 - (1.0 - composite_rgb) * (1.0 - spike_rgb)
        result = composite_rgb + (result - composite_rgb) * blend
    # Protect only the exact centre pixel. The four lines still run underneath
    # and visibly meet the stellar core without creating a large artificial gap.
    for cy, cx, *_rest in emitters:
        iy = int(np.clip(round(cy), 0, result.shape[0] - 1))
        ix = int(np.clip(round(cx), 0, result.shape[1] - 1))
        result[iy, ix] = composite_rgb[iy, ix]
    if exclusion_mask is not None:
        excluded = np.clip(exclusion_mask, 0.0, 1.0)[..., None]
        result = result * (1.0 - excluded) + composite_rgb * excluded
    return np.clip(result, 0.0, 1.0), selection_weight


def make_secondary_spikes(
    source_rgb: np.ndarray,
    composite_rgb: np.ndarray,
    min_diameter: float,
    feather: float,
    strength: float,
    length: float,
    width: float,
    angle: float,
    blend: float,
    catalog: StarCatalog,
    exclusion_mask: np.ndarray | None = None,
    linear_output: bool = False,
    progressive_linear_profile: bool = False,
) -> np.ndarray:
    """Add short, soft secondary spikes at a 45-degree offset."""
    emitters, _selection_weight = detect_star_emitters(
        source_rgb, min_diameter, feather, catalog
    )
    # Secondary selection is governed by the same measured star diameter as
    # primary spikes. Do not apply a second brightness gate that could reject
    # a large star while accepting a smaller but more saturated one.
    gated_emitters = list(emitters)

    if exclusion_mask is not None:
        allowed = 1.0 - np.clip(exclusion_mask, 0.0, 1.0)
        gated_emitters = [
            emitter
            for emitter in gated_emitters
            if allowed[
                int(np.clip(round(emitter[0]), 0, allowed.shape[0] - 1)),
                int(np.clip(round(emitter[1]), 0, allowed.shape[1] - 1)),
            ]
            > 0.5
        ]
    if not gated_emitters:
        return composite_rgb.copy()

    secondary_rgb = draw_point_emitter_spikes(
        source_rgb.shape,
        gated_emitters,
        length,
        width,
        angle + 45.0,
        linear_photometry=linear_output,
        progressive_linear_profile=progressive_linear_profile,
    )
    secondary_rgb = np.clip(secondary_rgb * strength, 0.0, 1.0)
    blend = float(np.clip(blend, 0.01, 1.0))
    if linear_output:
        result = np.clip(composite_rgb + secondary_rgb * blend, 0.0, 1.0)
    else:
        result = 1.0 - (1.0 - composite_rgb) * (1.0 - secondary_rgb)
        result = composite_rgb + (result - composite_rgb) * blend

    for cy, cx, *_rest in gated_emitters:
        iy = int(np.clip(round(cy), 0, result.shape[0] - 1))
        ix = int(np.clip(round(cx), 0, result.shape[1] - 1))
        result[iy, ix] = composite_rgb[iy, ix]
    if exclusion_mask is not None:
        excluded = np.clip(exclusion_mask, 0.0, 1.0)[..., None]
        result = result * (1.0 - excluded) + composite_rgb * excluded
    return np.clip(result, 0.0, 1.0)


def make_effects(
    base_rgb: np.ndarray,
    detection_black: float,
    detection_white: float,
    glow_enabled: bool,
    glow_parameters: dict,
    spikes_enabled: bool,
    spike_parameters: dict,
    secondary_enabled: bool,
    secondary_parameters: dict,
    exclusion_mask: np.ndarray | None = None,
    mask_glow: bool = True,
    mask_spikes: bool = True,
    progress_callback=None,
    linear_mode: bool = False,
    progressive_linear_spikes: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if progress_callback is not None:
        progress_callback("Detecting stars…", 8)
    catalog = (
        build_linear_star_catalog(base_rgb)
        if linear_mode
        else build_star_catalog(base_rgb, detection_black, detection_white)
    )
    if progress_callback is not None:
        progress_callback("Star detection complete", 25)
    result = base_rgb.copy()
    glow_selection = np.zeros(base_rgb.shape[:2], dtype=np.float32)
    if glow_enabled:
        if progress_callback is not None:
            progress_callback("Creating star glow…", 32)
        glow_function = make_glow_linear if linear_mode else make_glow
        result, glow_selection = glow_function(
            base_rgb,
            catalog=catalog,
            exclusion_mask=exclusion_mask if mask_glow else None,
            **glow_parameters,
        )
    if progress_callback is not None:
        progress_callback("Rendering primary spikes…", 52)
    spike_selection = np.zeros(base_rgb.shape[:2], dtype=np.float32)
    if spikes_enabled:
        result, spike_selection = make_newtonian_spikes(
            base_rgb,
            result,
            catalog=catalog,
            exclusion_mask=exclusion_mask if mask_spikes else None,
            linear_output=linear_mode,
            progressive_linear_profile=progressive_linear_spikes,
            **spike_parameters,
        )
        if (
            secondary_enabled
            and spike_parameters.get("spike_type") == "Newtonian"
        ):
            if progress_callback is not None:
                progress_callback("Rendering secondary spikes…", 70)
            result = make_secondary_spikes(
                base_rgb,
                result,
                catalog=catalog,
                exclusion_mask=exclusion_mask if mask_spikes else None,
                linear_output=linear_mode,
                progressive_linear_profile=progressive_linear_spikes,
                **secondary_parameters,
            )
    if progress_callback is not None:
        progress_callback("Compositing result…", 84)
    detection_mask = (catalog.labels > 0).astype(np.float32)
    return result, glow_selection, spike_selection, detection_mask


def resize_mask(mask: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray | None:
    """Nearest-neighbour resize without softening hand-drawn exclusion edges."""
    if mask is None:
        return None
    target_h, target_w = shape
    source_h, source_w = mask.shape
    yy = np.minimum(
        (np.arange(target_h) * source_h / max(target_h, 1)).astype(int),
        source_h - 1,
    )
    xx = np.minimum(
        (np.arange(target_w) * source_w / max(target_w, 1)).astype(int),
        source_w - 1,
    )
    return mask[np.ix_(yy, xx)].astype(np.float32)


def resize_preview(rgb: np.ndarray, max_side: int = PREVIEW_MAX) -> np.ndarray:
    h, w = rgb.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale >= 1.0:
        return rgb.copy()
    return ndi.zoom(rgb, (scale, scale, 1), order=1, prefilter=False)


def to_qpixmap(rgb: np.ndarray) -> QPixmap:
    if rgb.shape[2] == 1:
        rgb = np.repeat(rgb, 3, axis=2)
    rgb8 = np.ascontiguousarray(np.clip(rgb * 255.0, 0, 255).astype(np.uint8))
    h, w, _ = rgb8.shape
    image = QImage(rgb8.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(image)


class SlideToggle(QCheckBox):
    """Neutral two-position slide switch with a moving handle."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedSize(54, 28)

    def sizeHint(self) -> QSize:
        return QSize(54, 28)

    def hitButton(self, position: QPoint) -> bool:
        """Make every visible part of the switch reliably clickable."""
        return self.rect().contains(position)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        track = QRectF(1.0, 4.0, 52.0, 20.0)
        painter.setPen(QPen(QColor(135, 135, 135), 1.0))
        painter.setBrush(QColor(92, 92, 92))
        painter.drawRoundedRect(track, 10.0, 10.0)
        knob_x = 31.0 if self.isChecked() else 3.0
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(225, 225, 225))
        painter.drawEllipse(QRectF(knob_x, 5.0, 18.0, 18.0))
        painter.end()


class NumericControl(QWidget):
    """Slider plus an editable value box with native up/down step buttons."""

    valueChanged = pyqtSignal(float)

    def __init__(
        self,
        minimum: float,
        maximum: float,
        step: float,
        value: float,
        decimals: int = 2,
        suffix: str = "",
    ):
        super().__init__()
        self.scale = 10 ** int(decimals)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.spin = QDoubleSpinBox()
        self.spin.setDecimals(decimals)
        self.spin.setKeyboardTracking(False)
        self.spin.setMinimumWidth(94)
        self.setRange(minimum, maximum)
        self.setSingleStep(step)
        self.setSuffix(suffix)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(self.slider, 1)
        layout.addWidget(self.spin)

        self.slider.valueChanged.connect(self._from_slider)
        self.spin.valueChanged.connect(self._from_spin)
        self.setValue(value)

    def setRange(self, minimum: float, maximum: float) -> None:
        self.spin.setRange(minimum, maximum)
        self.slider.setRange(
            int(round(minimum * self.scale)),
            int(round(maximum * self.scale)),
        )

    def setSingleStep(self, step: float) -> None:
        self.spin.setSingleStep(step)
        self.slider.setSingleStep(max(1, int(round(step * self.scale))))
        self.slider.setPageStep(max(1, int(round(step * self.scale * 5))))

    def setSuffix(self, suffix: str) -> None:
        self.spin.setSuffix(suffix)

    def setValue(self, value: float) -> None:
        self.spin.setValue(value)
        self.slider.setValue(int(round(value * self.scale)))

    def value(self) -> float:
        return float(self.spin.value())

    def _from_slider(self, value: int) -> None:
        numeric = value / self.scale
        self.spin.blockSignals(True)
        self.spin.setValue(numeric)
        self.spin.blockSignals(False)
        self.valueChanged.emit(float(numeric))

    def _from_spin(self, value: float) -> None:
        self.slider.blockSignals(True)
        self.slider.setValue(int(round(value * self.scale)))
        self.slider.blockSignals(False)
        self.valueChanged.emit(float(value))


class ZoomPreview(QScrollArea):
    """Scrollable viewer with wheel zoom, drag-to-pan, and exclusion lasso."""

    def __init__(self):
        super().__init__()
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("background: #111;")
        self.setWidget(self.image_label)
        self.setWidgetResizable(False)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(600, 500)
        self.setStyleSheet("background: #111; border: none;")
        self.viewport().installEventFilter(self)
        self.source_pixmap = QPixmap()
        self.zoom = 1.0
        self.fit_mode = True
        self.dragging = False
        self.drag_start = QPoint()
        self.h_start = 0
        self.v_start = 0
        self.lasso_mode = False
        self.lasso_drawing = False
        self.lasso_points: list[QPoint] = []
        self.exclusion_mask: np.ndarray | None = None
        self.mask_visible = False
        self.mask_changed_callback = None

    def set_source_pixmap(self, pixmap: QPixmap) -> None:
        if (
            self.exclusion_mask is not None
            and self.exclusion_mask.shape != (pixmap.height(), pixmap.width())
        ):
            self.exclusion_mask = None
        self.source_pixmap = pixmap
        self.render_pixmap()

    def set_lasso_mode(self, enabled: bool) -> None:
        self.lasso_mode = bool(enabled)
        self.lasso_drawing = False
        self.lasso_points = []
        self.viewport().setCursor(
            Qt.CursorShape.CrossCursor
            if self.lasso_mode
            else Qt.CursorShape.OpenHandCursor
        )
        self.render_pixmap()

    def clear_exclusion_mask(self) -> None:
        self.exclusion_mask = None
        self.lasso_points = []
        self.render_pixmap()
        if self.mask_changed_callback is not None:
            self.mask_changed_callback()

    def set_mask_visible(self, visible: bool) -> None:
        self.mask_visible = bool(visible)
        self.render_pixmap()

    def source_point(self, viewport_point: QPoint) -> QPoint:
        label_point = self.image_label.mapFrom(self.viewport(), viewport_point)
        x = int(
            np.clip(
                label_point.x() / max(self.zoom, 1e-6),
                0,
                self.source_pixmap.width() - 1,
            )
        )
        y = int(
            np.clip(
                label_point.y() / max(self.zoom, 1e-6),
                0,
                self.source_pixmap.height() - 1,
            )
        )
        return QPoint(x, y)

    def commit_lasso(self) -> None:
        if len(self.lasso_points) < 3 or self.source_pixmap.isNull():
            self.lasso_points = []
            self.render_pixmap()
            return
        image = QImage(
            self.source_pixmap.width(),
            self.source_pixmap.height(),
            QImage.Format.Format_Grayscale8,
        )
        image.fill(0)
        painter = QPainter(image)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(255, 255, 255))
        painter.drawPolygon(QPolygon(self.lasso_points))
        painter.end()

        ptr = image.bits()
        ptr.setsize(image.sizeInBytes())
        raster = np.frombuffer(ptr, dtype=np.uint8).reshape(
            image.height(), image.bytesPerLine()
        )[:, : image.width()].copy()
        new_mask = (raster > 0).astype(np.float32)
        self.exclusion_mask = (
            new_mask
            if self.exclusion_mask is None
            else np.maximum(self.exclusion_mask, new_mask)
        )
        self.lasso_points = []
        self.render_pixmap()
        if self.mask_changed_callback is not None:
            self.mask_changed_callback()

    def fit_to_viewer(self) -> None:
        self.fit_mode = True
        self.zoom = 1.0
        self.render_pixmap()

    def set_zoom(self, zoom: float) -> None:
        self.fit_mode = False
        self.zoom = float(np.clip(zoom, 0.1, 12.0))
        self.render_pixmap()

    def zoom_in(self) -> None:
        self.set_zoom(self.zoom * 1.25)

    def zoom_out(self) -> None:
        self.set_zoom(self.zoom / 1.25)

    def actual_size(self) -> None:
        self.set_zoom(1.0)

    def eventFilter(self, watched, event) -> bool:
        if watched is self.viewport():
            if event.type() == QEvent.Type.Wheel and not self.source_pixmap.isNull():
                steps = event.angleDelta().y() / 120.0
                if steps:
                    old_h = self.horizontalScrollBar().value()
                    old_v = self.verticalScrollBar().value()
                    old_zoom = self.zoom
                    self.fit_mode = False
                    self.zoom = float(np.clip(self.zoom * (1.2 ** steps), 0.1, 12.0))
                    ratio = self.zoom / max(old_zoom, 1e-6)
                    self.render_pixmap()
                    self.horizontalScrollBar().setValue(
                        int((old_h + event.position().x()) * ratio - event.position().x())
                    )
                    self.verticalScrollBar().setValue(
                        int((old_v + event.position().y()) * ratio - event.position().y())
                    )
                    return True
            elif (
                self.lasso_mode
                and event.type() == QEvent.Type.MouseButtonPress
                and event.button() == Qt.MouseButton.LeftButton
            ):
                self.lasso_drawing = True
                self.lasso_points = [self.source_point(event.position().toPoint())]
                return True
            elif (
                self.lasso_mode
                and event.type() == QEvent.Type.MouseMove
                and self.lasso_drawing
            ):
                point = self.source_point(event.position().toPoint())
                if not self.lasso_points or point != self.lasso_points[-1]:
                    self.lasso_points.append(point)
                    self.render_pixmap()
                return True
            elif (
                self.lasso_mode
                and event.type() == QEvent.Type.MouseButtonRelease
                and event.button() == Qt.MouseButton.LeftButton
            ):
                self.lasso_drawing = False
                self.commit_lasso()
                return True
            elif (
                not self.lasso_mode
                and event.type() == QEvent.Type.MouseButtonPress
                and event.button() == Qt.MouseButton.LeftButton
            ):
                self.dragging = True
                self.drag_start = event.position().toPoint()
                self.h_start = self.horizontalScrollBar().value()
                self.v_start = self.verticalScrollBar().value()
                self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
                return True
            elif event.type() == QEvent.Type.MouseMove and self.dragging:
                delta = event.position().toPoint() - self.drag_start
                self.horizontalScrollBar().setValue(self.h_start - delta.x())
                self.verticalScrollBar().setValue(self.v_start - delta.y())
                return True
            elif (
                event.type() == QEvent.Type.MouseButtonRelease
                and event.button() == Qt.MouseButton.LeftButton
            ):
                self.dragging = False
                self.viewport().setCursor(Qt.CursorShape.OpenHandCursor)
                return True
        return super().eventFilter(watched, event)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self.fit_mode:
            self.render_pixmap()

    def render_pixmap(self) -> None:
        if self.source_pixmap.isNull():
            return
        if self.fit_mode:
            available = self.viewport().size()
            self.zoom = min(
                available.width() / max(self.source_pixmap.width(), 1),
                available.height() / max(self.source_pixmap.height(), 1),
            )

        decorated = QPixmap(self.source_pixmap)
        painter = QPainter(decorated)
        if self.exclusion_mask is not None and self.mask_visible:
            rgba = np.zeros(
                (decorated.height(), decorated.width(), 4), dtype=np.uint8
            )
            rgba[..., 0] = 45
            rgba[..., 1] = 135
            rgba[..., 2] = 255
            rgba[..., 3] = (self.exclusion_mask * 105).astype(np.uint8)
            overlay = QImage(
                rgba.data,
                rgba.shape[1],
                rgba.shape[0],
                rgba.strides[0],
                QImage.Format.Format_RGBA8888,
            ).copy()
            painter.drawImage(0, 0, overlay)
        if len(self.lasso_points) > 1:
            painter.setPen(QPen(QColor(70, 170, 255), 2))
            painter.drawPolyline(QPolygon(self.lasso_points))
        painter.end()

        width = max(1, int(round(decorated.width() * self.zoom)))
        height = max(1, int(round(decorated.height() * self.zoom)))
        scaled = decorated.scaled(
            width,
            height,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled)
        self.image_label.resize(scaled.size())


class StarGlowWindow(QMainWindow):
    def __init__(self, siril: s.SirilInterface):
        super().__init__()
        self.siril = siril
        self.setWindowTitle(APP_TITLE)
        self.resize(1180, 780)
        self.setStyleSheet(
            "QWidget { font-family: 'Avenir Next', 'Inter', "
            "'Helvetica Neue', sans-serif; font-weight: 500; }"
            "QGroupBox { border: 1px solid #555; border-radius: 5px; "
            "margin-top: 10px; padding-top: 8px; }"
            "QGroupBox::title { color: #d6b45a; font-weight: 700; "
            "subcontrol-origin: margin; subcontrol-position: top left; "
            "left: 10px; padding: 0 5px; }"
            "QPushButton { background-color: #444; color: white; "
            "font-weight: bold; border: 1px solid #777; "
            "border-radius: 5px; padding: 5px 12px; min-height: 18px; }"
            "QPushButton:hover { background-color: #505050; "
            "border-color: #888; }"
            "QPushButton:pressed { background-color: #383838; }"
            "QPushButton:checked { background-color: #555; "
            "border-color: #aaa; }"
            "QPushButton:disabled { background-color: #363636; "
            "color: #777; border-color: #505050; }"
        )

        with self.siril.image_lock():
            image = self.siril.get_image()
            if image is None or image.data is None:
                raise RuntimeError("No image is currently loaded in Siril.")
            raw = np.array(image.data, copy=True)

        self.raw_hwc, self.layout = canonical_hwc(raw)
        self.base_native, self.scale = normalize_image(self.raw_hwc)

        # Ignore alpha if one appears; Siril science images are normally 1 or 3 channel.
        if self.base_native.shape[2] == 4:
            self.base_native = self.base_native[..., :3]

        self.orientation_combo = QComboBox()
        self.orientation_combo.addItems(
            ["No transform", "Flip vertical", "Flip horizontal", "Rotate 180°"]
        )
        self.orientation_combo.setCurrentText("Flip vertical")
        self.base_full = apply_orientation(
            self.base_native, self.orientation_combo.currentText()
        )

        self.preview_base = resize_preview(self.base_full)
        self.preview_working = self.preview_base
        self.preview_scale = self.preview_base.shape[1] / self.base_full.shape[1]
        self.starless_native: np.ndarray | None = None
        self.starless_preview: np.ndarray | None = None

        self.preview_label = ZoomPreview()
        self.preview_label.mask_changed_callback = self.schedule_preview
        self.preview_resolution_combo = QComboBox()
        self.preview_resolution_combo.addItem("Preview: 1200 px", 1200)
        self.preview_resolution_combo.addItem("Preview: 1600 px", 1600)
        self.preview_resolution_combo.addItem("Preview: 2400 px", 2400)
        self.preview_resolution_combo.addItem("Preview: 3200 px", 3200)
        self.preview_resolution_combo.addItem("Preview: Full resolution", 0)
        self.preview_resolution_combo.setCurrentIndex(
            self.preview_resolution_combo.findData(PREVIEW_MAX)
        )

        self.linear_mode_toggle = SlideToggle()
        self.linear_mode_toggle.setChecked(False)
        self.linear_mode_toggle.setToolTip(
            "Off: process an already stretched star mask exactly as in v1.2.0.\n"
            "On: temporarily apply Rational LogD, use the established Aster "
            "effects, then reverse LogD before Push to Siril."
        )
        self.non_linear_mode_label = QLabel("Non-Linear (Stretched)")
        self.linear_mode_label = QLabel("Linear (Unstretched)")
        self.rational_logd_spin = NumericControl(
            1.0, 21.0, 0.1, 9.0, 1
        )

        self.detection_black_spin = NumericControl(
            0.0, 0.99, 0.01, 0.20, 2
        )
        self.detection_white_spin = NumericControl(
            0.01, 1.0, 0.01, 0.90, 2
        )
        self.detection_mask_checkbox = QCheckBox(
            "Show common detection mask"
        )
        self.detection_mask_checkbox.setChecked(False)

        self.glow_enabled = QCheckBox("Enable star glow")
        self.glow_enabled.setChecked(False)

        self.blur_combo = QComboBox()
        self.blur_combo.addItems(
            ["Gaussian", "Multi-scale Gaussian", "Moffat", "Triangle", "Box", "Disk"]
        )
        self.blur_combo.setCurrentText("Moffat")

        self.radius_spin = NumericControl(0.5, 100.0, 0.5, 10.0, 1, " px")
        self.diameter_slider = NumericControl(1, 300, 1, 15, 0, " px")
        self.feather_spin = NumericControl(0.5, 40.0, 0.5, 20.0, 1, " px")
        self.strength_spin = NumericControl(0.0, 2.0, 0.05, 0.50, 2)
        self.gamma_spin = NumericControl(0.10, 3.00, 0.05, 0.70, 2)
        self.glow_blend_spin = NumericControl(1, 100, 1, 100, 0, "%")

        self.mask_checkbox = QCheckBox("Show selected stars in red")
        self.mask_checkbox.setChecked(False)

        self.spikes_enabled = QCheckBox("Enable star spikes")
        self.spikes_enabled.setChecked(False)

        self.spike_type_combo = QComboBox()
        self.spike_type_combo.addItems(["Newtonian", "JWST"])
        self.spike_type_combo.setCurrentText("Newtonian")

        self.spectral_checkbox = QCheckBox("Enable spectral diffraction")
        self.spectral_checkbox.setChecked(False)
        self.spectral_strength_spin = NumericControl(
            0, 100, 1, 100, 0, "%"
        )
        self.spectral_saturation_spin = NumericControl(
            0, 200, 1, 150, 0, "%"
        )

        self.spike_diameter_slider = NumericControl(1, 300, 1, 20, 0, " px")
        self.spike_feather_spin = NumericControl(0.5, 40.0, 0.5, 30.0, 1, " px")
        self.spike_strength_spin = NumericControl(0.0, 2.0, 0.05, 0.80, 2)
        self.spike_length_spin = NumericControl(2.0, 1000.0, 2.0, 400.0, 0, " px")
        self.spike_width_spin = NumericControl(0.0, 10.0, 0.10, 1.0, 2, " px")
        self.spike_angle_spin = NumericControl(0.0, 45.0, 1.0, 0.0, 1, "°")
        self.spike_blend_spin = NumericControl(1, 100, 1, 100, 0, "%")

        self.spike_mask_checkbox = QCheckBox("Show spike-selected stars in green")
        self.spike_mask_checkbox.setChecked(False)

        self.secondary_spikes_enabled = QCheckBox(
            "Enable secondary soft spikes (+45°)"
        )
        self.secondary_spikes_enabled.setChecked(False)
        self.secondary_diameter_spin = NumericControl(
            1, 300, 1, 20, 0, " px"
        )
        self.secondary_feather_spin = NumericControl(
            0.5, 80.0, 0.5, 10.0, 1, " px"
        )
        self.secondary_strength_spin = NumericControl(
            0.0, 2.0, 0.05, 0.40, 2
        )
        self.secondary_length_spin = NumericControl(
            2.0, 400.0, 2.0, 80.0, 0, " px"
        )
        self.secondary_width_spin = NumericControl(
            0.0, 20.0, 0.25, 4.0, 2, " px"
        )

        self.lasso_button = QPushButton("Draw exclusion lasso")
        self.lasso_button.setCheckable(True)
        self.lasso_button.setStyleSheet(
            "QPushButton { background-color: #2471a3; color: white; "
            "font-weight: bold; border: 1px solid #3498db; "
            "border-radius: 5px; padding: 5px 12px; }"
            "QPushButton:hover { background-color: #2e86c1; }"
            "QPushButton:checked { background-color: #1a5276; "
            "border-color: #5dade2; }"
        )
        self.clear_mask_button = QPushButton("Clear exclusion mask")
        self.clear_mask_button.setStyleSheet(
            "QPushButton { background-color: #a93226; color: white; "
            "font-weight: bold; border: 1px solid #e74c3c; "
            "border-radius: 5px; padding: 5px 12px; }"
            "QPushButton:hover { background-color: #c0392b; }"
            "QPushButton:pressed { background-color: #78281f; }"
        )
        self.mask_glow_checkbox = QCheckBox("Apply exclusion mask to glow")
        self.mask_glow_checkbox.setChecked(True)
        self.mask_spikes_checkbox = QCheckBox("Apply exclusion mask to star spikes")
        self.mask_spikes_checkbox.setChecked(True)
        self.show_mask_checkbox = QCheckBox("Show exclusion mask overlay")
        self.show_mask_checkbox.setChecked(False)

        self.before_button = QPushButton("View: Processed (After)")
        self.before_button.setCheckable(True)
        self.before_button.setChecked(False)

        self.load_starless_button = QPushButton("Load Starless Image")
        self.load_starless_button.setMinimumHeight(28)
        self.load_starless_button.setStyleSheet(
            "QPushButton { background: #444; color: white; font-weight: bold; "
            "border: 1px solid #777; border-radius: 5px; padding: 5px 12px; }"
            "QPushButton:hover { background: #505050; }"
        )
        self.load_starless_button.setToolTip(
            "Load a processed starless FITS or TIFF image for preview only."
        )
        self.show_starless_toggle = QCheckBox("Show Starless Background")
        self.show_starless_toggle.setChecked(False)
        self.show_starless_toggle.setEnabled(False)
        self.show_starless_toggle.setToolTip(
            "Show or hide the loaded starless image beneath the stars. "
            "This never changes the result pushed to Siril."
        )

        self.push_button = QPushButton("Push result to Siril")
        self.push_button.setMinimumHeight(42)
        self.push_button.setStyleSheet(
            "QPushButton { background-color: #238636; color: white; "
            "font-weight: bold; padding: 8px 16px; border-radius: 4px; }"
            "QPushButton:disabled { background-color: #5c6f61; }"
        )

        self.reset_button = QPushButton("Reset controls")
        self.zoom_out_button = QPushButton("−")
        self.fit_button = QPushButton("Fit")
        self.actual_size_button = QPushButton("1:1")
        self.zoom_in_button = QPushButton("+")
        for viewer_button in (
            self.zoom_out_button,
            self.fit_button,
            self.actual_size_button,
            self.zoom_in_button,
        ):
            viewer_button.setFixedWidth(48)
            viewer_button.setStyleSheet(
                "QPushButton { background: #444; color: white; font-weight: bold; "
                "border: 1px solid #666; border-radius: 4px; padding: 4px; }"
                "QPushButton:hover { background: #555; }"
            )
        self.close_button = QPushButton("Close")
        self.close_button.setMinimumHeight(42)
        self.close_button.setStyleSheet(
            "QPushButton { background-color: #c0392b; color: white; "
            "font-weight: bold; padding: 8px 16px; border-radius: 4px; }"
        )

        form = QFormLayout()
        form.addRow(self.glow_enabled)
        form.addRow("Blur type", self.blur_combo)
        form.addRow("Glow radius", self.radius_spin)

        form.addRow("Minimum star diameter", self.diameter_slider)
        form.addRow("Cutoff feather", self.feather_spin)

        form.addRow("Glow strength", self.strength_spin)
        form.addRow("Glow gamma", self.gamma_spin)
        form.addRow(self.mask_checkbox)
        form.addRow("Universal Blend", self.glow_blend_spin)

        controls = QGroupBox("Star Glow Controls")
        controls.setLayout(form)

        spike_form = QFormLayout()
        spike_form.addRow(self.spikes_enabled)
        spike_form.addRow("Spike type", self.spike_type_combo)
        spike_form.addRow("Minimum star diameter", self.spike_diameter_slider)
        spike_form.addRow("Cutoff feather", self.spike_feather_spin)
        spike_form.addRow("Spike strength", self.spike_strength_spin)
        spike_form.addRow("Spike length", self.spike_length_spin)
        spike_form.addRow("Spike width", self.spike_width_spin)
        spike_form.addRow("Spike angle", self.spike_angle_spin)
        spike_form.addRow(self.spike_mask_checkbox)
        spike_form.addRow("Universal Blend", self.spike_blend_spin)

        spectral_form = QFormLayout()
        spectral_form.addRow(self.spectral_checkbox)
        spectral_form.addRow(
            "Spectral strength", self.spectral_strength_spin
        )
        spectral_form.addRow(
            "Spectrum saturation", self.spectral_saturation_spin
        )
        spectral_controls = QGroupBox("Spectral Diffraction")
        spectral_controls.setLayout(spectral_form)
        spike_form.addRow(spectral_controls)

        secondary_form = QFormLayout()
        secondary_form.addRow(self.secondary_spikes_enabled)
        secondary_form.addRow(
            "Minimum star diameter", self.secondary_diameter_spin
        )
        secondary_form.addRow(
            "Cutoff feather", self.secondary_feather_spin
        )
        secondary_form.addRow(
            "Secondary strength", self.secondary_strength_spin
        )
        secondary_form.addRow(
            "Secondary length", self.secondary_length_spin
        )
        secondary_form.addRow(
            "Secondary width", self.secondary_width_spin
        )
        secondary_controls = QGroupBox("Secondary Spikes")
        secondary_controls.setLayout(secondary_form)
        spike_form.addRow(secondary_controls)

        spike_controls = QGroupBox("Star Spikes Controls")
        spike_controls.setLayout(spike_form)

        mask_form = QVBoxLayout()
        mask_buttons = QHBoxLayout()
        mask_buttons.addWidget(self.lasso_button)
        mask_buttons.addWidget(self.clear_mask_button)
        mask_form.addLayout(mask_buttons)
        mask_form.addWidget(self.mask_glow_checkbox)
        mask_form.addWidget(self.mask_spikes_checkbox)
        mask_form.addWidget(self.show_mask_checkbox)
        mask_help = QLabel(
            "Enable the lasso, then draw a closed region on the viewer. "
            "Excluded areas are blue when the mask overlay is visible."
        )
        mask_help.setWordWrap(True)
        mask_form.addWidget(mask_help)
        mask_controls = QGroupBox("Exclusion Mask")
        mask_controls.setLayout(mask_form)

        detection_form = QFormLayout()
        detection_form.addRow(
            "Detection black point", self.detection_black_spin
        )
        detection_form.addRow(
            "Detection white point", self.detection_white_spin
        )
        detection_form.addRow(self.detection_mask_checkbox)
        detection_help = QLabel(
            "Detection-only levels. They do not stretch or modify the result."
        )
        detection_help.setWordWrap(True)
        detection_form.addRow(detection_help)
        detection_controls = QGroupBox("Star Detection")
        detection_controls.setLayout(detection_form)

        input_mode_form = QFormLayout()
        mode_selector = QHBoxLayout()
        mode_selector.addWidget(self.non_linear_mode_label)
        mode_selector.addStretch(1)
        mode_selector.addWidget(self.linear_mode_toggle)
        mode_selector.addStretch(1)
        mode_selector.addWidget(self.linear_mode_label)
        input_mode_form.addRow(mode_selector)
        input_mode_form.addRow(
            "Rational Log D preview", self.rational_logd_spin
        )
        input_mode_controls = QGroupBox("Star Mask Input Mode")
        input_mode_controls.setLayout(input_mode_form)

        script_identity = QLabel(
            "<div style='font-size: 22px; font-weight: 700; "
            "color: #f1c40f;'>Aster</div>"
            "<div style='font-size: 14px; font-weight: 700; "
            "color: #e2d2a2;'>"
            "Star Cosmetic Enhancement</div>"
            "<div>v1.3.0 - Star Glow · Star Spikes</div>"
            "<div>Created by Panagiotis Stefos ©</div>"
        )
        script_identity.setWordWrap(True)
        script_identity.setTextFormat(Qt.TextFormat.RichText)
        script_identity.setStyleSheet(
            "QLabel { padding: 6px 4px 10px 4px; }"
        )

        input_note = QLabel(
            "Input requirement: stars-only image. Select Linear or Non-linear below."
        )
        input_note.setWordWrap(True)
        input_note.setStyleSheet(
            "QLabel { color: #d99a3d; font-weight: bold; padding: 4px; }"
        )

        starless_controls = QVBoxLayout()
        starless_controls.setContentsMargins(4, 2, 4, 6)
        starless_controls.addWidget(self.load_starless_button)
        starless_controls.addWidget(self.show_starless_toggle)

        left = QVBoxLayout()
        left.addWidget(script_identity)
        left.addWidget(input_note)
        left.addLayout(starless_controls)
        left.addWidget(input_mode_controls)
        left.addWidget(detection_controls)
        left.addWidget(controls)
        left.addWidget(spike_controls)
        left.addWidget(mask_controls)
        left.addWidget(self.reset_button)
        left.addStretch(1)
        left_widget = QWidget()
        left_widget.setLayout(left)
        left_widget.setMinimumWidth(450)
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setWidget(left_widget)
        left_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        left_scroll.setMinimumWidth(480)
        left_scroll.setMaximumWidth(540)
        left_scroll.setFrameShape(QFrame.Shape.NoFrame)

        viewer_top = QHBoxLayout()
        viewer_top.addWidget(self.zoom_out_button)
        viewer_top.addWidget(self.fit_button)
        viewer_top.addWidget(self.actual_size_button)
        viewer_top.addWidget(self.zoom_in_button)
        viewer_top.addWidget(self.preview_resolution_combo)
        viewer_top.addWidget(self.orientation_combo)
        compare_hint = QLabel("Hold Space Bar to Compare")
        compare_hint.setStyleSheet(
            "QLabel { color: #f1c40f; font-weight: bold; padding: 4px 8px; }"
        )
        viewer_top.addWidget(compare_hint)
        viewer_top.addStretch(1)

        actions = QHBoxLayout()
        actions.addStretch(1)
        actions.addWidget(self.close_button)
        actions.addWidget(self.push_button)

        self.progress_label = QLabel("Ready")
        self.progress_label.setVisible(False)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setVisible(False)

        viewer_column = QVBoxLayout()
        viewer_column.addLayout(viewer_top)
        viewer_column.addWidget(self.preview_label, 1)
        viewer_column.addWidget(self.progress_label)
        viewer_column.addWidget(self.progress_bar)
        viewer_column.addLayout(actions)

        root = QHBoxLayout()
        root.addWidget(left_scroll, 0)
        root.addLayout(viewer_column, 1)

        central = QWidget()
        central.setLayout(root)
        self.setCentralWidget(central)

        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(240)
        self.timer.timeout.connect(self.update_preview)

        self.update_input_mode_controls(rebuild=False)

        for widget in (
            self.blur_combo,
            self.linear_mode_toggle,
            self.detection_black_spin,
            self.detection_white_spin,
            self.detection_mask_checkbox,
            self.glow_enabled,
            self.radius_spin,
            self.diameter_slider,
            self.feather_spin,
            self.strength_spin,
            self.gamma_spin,
            self.glow_blend_spin,
            self.mask_checkbox,
            self.spike_type_combo,
            self.spikes_enabled,
            self.spectral_checkbox,
            self.spectral_strength_spin,
            self.spectral_saturation_spin,
            self.spike_diameter_slider,
            self.spike_feather_spin,
            self.spike_strength_spin,
            self.spike_length_spin,
            self.spike_width_spin,
            self.spike_angle_spin,
            self.spike_blend_spin,
            self.spike_mask_checkbox,
            self.secondary_spikes_enabled,
            self.secondary_diameter_spin,
            self.secondary_feather_spin,
            self.secondary_strength_spin,
            self.secondary_length_spin,
            self.secondary_width_spin,
            self.mask_glow_checkbox,
            self.mask_spikes_checkbox,
        ):
            if isinstance(widget, QComboBox):
                widget.currentIndexChanged.connect(self.schedule_preview)
            elif isinstance(widget, QCheckBox):
                widget.stateChanged.connect(self.schedule_preview)
            else:
                widget.valueChanged.connect(self.schedule_preview)

        self.linear_mode_toggle.stateChanged.connect(
            self.update_input_mode_controls
        )
        self.rational_logd_spin.valueChanged.connect(
            self.rebuild_preview_working
        )

        self.spikes_enabled.stateChanged.connect(self.update_spike_controls)
        self.spike_type_combo.currentIndexChanged.connect(
            self.update_spike_controls
        )
        self.spectral_checkbox.stateChanged.connect(
            self.update_spectral_controls
        )
        self.secondary_spikes_enabled.stateChanged.connect(
            self.update_secondary_spike_controls
        )
        self.glow_enabled.stateChanged.connect(self.update_glow_controls)
        self.lasso_button.toggled.connect(self.preview_label.set_lasso_mode)
        self.show_mask_checkbox.stateChanged.connect(
            self.preview_label.set_mask_visible
        )
        self.preview_resolution_combo.currentIndexChanged.connect(
            self.update_preview_resolution
        )
        self.orientation_combo.currentIndexChanged.connect(
            self.update_orientation
        )
        self.before_button.toggled.connect(self.update_before_after_button)
        self.load_starless_button.clicked.connect(self.load_starless_image)
        self.show_starless_toggle.stateChanged.connect(self.schedule_preview)
        self.clear_mask_button.clicked.connect(
            self.preview_label.clear_exclusion_mask
        )
        self.push_button.clicked.connect(self.push_to_siril)
        self.fit_button.clicked.connect(self.preview_label.fit_to_viewer)
        self.zoom_out_button.clicked.connect(self.preview_label.zoom_out)
        self.zoom_in_button.clicked.connect(self.preview_label.zoom_in)
        self.actual_size_button.clicked.connect(self.preview_label.actual_size)
        self.reset_button.clicked.connect(self.reset_controls)
        self.close_button.clicked.connect(self.close)

        self.update_glow_controls()
        self.update_spike_controls()
        self.update_secondary_spike_controls()
        self.rebuild_preview_working(schedule=False)
        self._space_before_active = False
        self._before_state_before_space = False
        QApplication.instance().installEventFilter(self)
        self.update_preview()

    def update_input_mode_controls(
        self, *_args, rebuild: bool = True
    ) -> None:
        linear = self.linear_mode_toggle.isChecked()
        self.rational_logd_spin.setEnabled(linear)
        # Detection runs on the temporary stretched working image in Linear
        # mode, so the established detection controls remain useful.
        self.detection_black_spin.setEnabled(True)
        self.detection_white_spin.setEnabled(True)
        active_style = (
            "QLabel { color: white; font-weight: bold; "
            "background: #555; border-radius: 4px; padding: 4px; }"
        )
        inactive_style = "QLabel { color: #999; padding: 4px; }"
        self.non_linear_mode_label.setStyleSheet(
            inactive_style if linear else active_style
        )
        self.linear_mode_label.setStyleSheet(
            active_style if linear else inactive_style
        )
        if rebuild:
            self.spike_strength_spin.setValue(1.50 if linear else 0.80)
            self.rebuild_preview_working()

    def rebuild_preview_working(
        self, *_args, schedule: bool = True
    ) -> None:
        """Build the image consumed by the unchanged non-linear effects."""
        if self.linear_mode_toggle.isChecked():
            self.preview_working = rational_logd_preview(
                self.preview_base, self.rational_logd_spin.value()
            )
        else:
            self.preview_working = self.preview_base
        _CATALOG_CACHE.clear()
        _EMITTER_CACHE.clear()
        if schedule:
            self.schedule_preview()

    def rebuild_starless_preview(self) -> None:
        if self.starless_native is None:
            self.starless_preview = None
            return
        oriented = apply_orientation(
            self.starless_native, self.orientation_combo.currentText()
        )
        requested = int(self.preview_resolution_combo.currentData())
        max_side = max(oriented.shape[:2]) if requested == 0 else requested
        self.starless_preview = resize_preview(oriented, max_side)

    def load_starless_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Processed Starless Image",
            "",
            "Astronomy images (*.fit *.fits *.fts *.tif *.tiff);;"
            "FITS images (*.fit *.fits *.fts);;TIFF images (*.tif *.tiff)",
        )
        if not path:
            return
        try:
            lower = path.lower()
            if lower.endswith((".fit", ".fits", ".fts")):
                data = np.asarray(fits.getdata(path))
            else:
                image = QImage(path)
                if image.isNull():
                    raise ValueError("The TIFF image could not be decoded.")
                image = image.convertToFormat(QImage.Format.Format_RGB888)
                ptr = image.bits()
                ptr.setsize(image.sizeInBytes())
                rows = np.frombuffer(ptr, dtype=np.uint8).reshape(
                    image.height(), image.bytesPerLine()
                )
                data = rows[:, : image.width() * 3].reshape(
                    image.height(), image.width(), 3
                ).copy()

            starless_hwc, _ = canonical_hwc(data)
            if starless_hwc.shape[2] == 4:
                starless_hwc = starless_hwc[..., :3]
            if starless_hwc.shape[:2] != self.base_native.shape[:2]:
                raise ValueError(
                    "The starless image must have the same width and height "
                    "as the stars-only image."
                )
            self.starless_native, _ = normalize_image(starless_hwc)
            self.rebuild_starless_preview()
            self.show_starless_toggle.setEnabled(True)
            self.show_starless_toggle.setChecked(True)
            self.load_starless_button.setText("Replace Starless Image")
            self.load_starless_button.setStyleSheet(
                "QPushButton { background: #238636; color: white; "
                "font-weight: bold; border: 1px solid #49a65d; "
                "border-radius: 5px; padding: 5px 12px; }"
                "QPushButton:hover { background: #2ea043; }"
            )
            self.schedule_preview()
        except Exception as exc:
            QMessageBox.critical(
                self,
                APP_TITLE,
                f"Could not load the starless image:\n\n{exc}",
            )

    def update_spike_controls(self, *_args) -> None:
        enabled = self.spikes_enabled.isChecked()
        for widget in (
            self.spectral_checkbox,
            self.spike_diameter_slider,
            self.spike_feather_spin,
            self.spike_strength_spin,
            self.spike_length_spin,
            self.spike_width_spin,
            self.spike_angle_spin,
            self.spike_blend_spin,
            self.spike_mask_checkbox,
            self.secondary_spikes_enabled,
        ):
            widget.setEnabled(enabled)
        self.secondary_spikes_enabled.setEnabled(
            enabled and self.spike_type_combo.currentText() == "Newtonian"
        )
        self.update_spectral_controls()
        self.update_secondary_spike_controls()
        self.schedule_preview()

    def update_spectral_controls(self, *_args) -> None:
        enabled = (
            self.spikes_enabled.isChecked()
            and self.spectral_checkbox.isChecked()
        )
        for widget in (
            self.spectral_strength_spin,
            self.spectral_saturation_spin,
        ):
            widget.setEnabled(enabled)
        self.schedule_preview()

    def update_secondary_spike_controls(self, *_args) -> None:
        enabled = (
            self.spikes_enabled.isChecked()
            and self.secondary_spikes_enabled.isChecked()
            and self.spike_type_combo.currentText() == "Newtonian"
        )
        for widget in (
            self.secondary_diameter_spin,
            self.secondary_feather_spin,
            self.secondary_strength_spin,
            self.secondary_length_spin,
            self.secondary_width_spin,
        ):
            widget.setEnabled(enabled)
        self.schedule_preview()

    def update_glow_controls(self, *_args) -> None:
        enabled = self.glow_enabled.isChecked()
        for widget in (
            self.blur_combo,
            self.radius_spin,
            self.diameter_slider,
            self.feather_spin,
            self.strength_spin,
            self.gamma_spin,
            self.glow_blend_spin,
            self.mask_checkbox,
        ):
            widget.setEnabled(enabled)
        self.schedule_preview()

    def schedule_preview(self, *_args) -> None:
        self.timer.start()

    def update_preview_resolution(self, *_args) -> None:
        requested = int(self.preview_resolution_combo.currentData())
        max_side = (
            max(self.base_full.shape[:2])
            if requested == 0
            else requested
        )
        old_mask = self.preview_label.exclusion_mask
        self.preview_base = resize_preview(self.base_full, max_side)
        self.preview_scale = (
            self.preview_base.shape[1] / self.base_full.shape[1]
        )
        self.preview_label.exclusion_mask = resize_mask(
            old_mask, self.preview_base.shape[:2]
        )
        self.rebuild_starless_preview()
        self.rebuild_preview_working()

    def update_orientation(self, *_args) -> None:
        self.base_full = apply_orientation(
            self.base_native, self.orientation_combo.currentText()
        )
        _CATALOG_CACHE.clear()
        _EMITTER_CACHE.clear()
        self.preview_label.clear_exclusion_mask()
        self.update_preview_resolution()

    def update_before_after_button(self, checked: bool) -> None:
        self.before_button.setText(
            "View: Original (Before)"
            if checked
            else "View: Processed (After)"
        )
        self.schedule_preview()

    def keyPressEvent(self, event) -> None:
        if (
            event.key() == Qt.Key.Key_Space
            and not event.isAutoRepeat()
            and not self._space_before_active
        ):
            self._space_before_active = True
            self._before_state_before_space = self.before_button.isChecked()
            self.before_button.setChecked(True)
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:
        if (
            event.key() == Qt.Key.Key_Space
            and not event.isAutoRepeat()
            and self._space_before_active
        ):
            self._space_before_active = False
            self.before_button.setChecked(
                self._before_state_before_space
            )
            event.accept()
            return
        super().keyReleaseEvent(event)

    def eventFilter(self, watched, event) -> bool:
        if event.type() == QEvent.Type.KeyPress:
            if (
                event.key() == Qt.Key.Key_Space
                and not event.isAutoRepeat()
                and not self._space_before_active
            ):
                self._space_before_active = True
                self._before_state_before_space = (
                    self.before_button.isChecked()
                )
                self.before_button.setChecked(True)
                self.timer.stop()
                self.update_preview()
                return True
        elif event.type() == QEvent.Type.KeyRelease:
            if (
                event.key() == Qt.Key.Key_Space
                and not event.isAutoRepeat()
                and self._space_before_active
            ):
                self._space_before_active = False
                self.before_button.setChecked(
                    self._before_state_before_space
                )
                self.timer.stop()
                self.update_preview()
                return True
        return super().eventFilter(watched, event)

    def detection_parameters(self) -> tuple[float, float]:
        black = self.detection_black_spin.value()
        white = max(self.detection_white_spin.value(), black + 0.01)
        return black, min(white, 1.0)

    def glow_parameters(self, preview: bool = False) -> dict:
        spatial_scale = self.preview_scale if preview else 1.0
        return {
            "min_diameter": max(1.0, self.diameter_slider.value() * spatial_scale),
            "feather": max(0.1, self.feather_spin.value() * spatial_scale),
            "radius": max(0.5, self.radius_spin.value() * spatial_scale),
            "strength": self.strength_spin.value(),
            "gamma": self.gamma_spin.value(),
            "blur_mode": self.blur_combo.currentText(),
            "blend": self.glow_blend_spin.value() / 100.0,
        }

    def spike_parameters(self, preview: bool = False) -> dict:
        spatial_scale = self.preview_scale if preview else 1.0
        return {
            "min_diameter": max(
                1.0, self.spike_diameter_slider.value() * spatial_scale
            ),
            "feather": max(
                0.1, self.spike_feather_spin.value() * spatial_scale
            ),
            "strength": self.spike_strength_spin.value(),
            "length": max(
                2.0, self.spike_length_spin.value() * spatial_scale
            ),
            "width": max(
                0.0, self.spike_width_spin.value() * spatial_scale
            ),
            "angle": self.spike_angle_spin.value(),
            "blend": self.spike_blend_spin.value() / 100.0,
            "spike_type": self.spike_type_combo.currentText(),
            "spectral": self.spectral_checkbox.isChecked(),
            "spectral_strength": (
                self.spectral_strength_spin.value() / 100.0
            ),
            # Fixed to the more natural checkbox-only look from the first
            # spectral prototype.
            "spectral_position": 0.28,
            "spectral_spread": 0.18,
            "spectral_saturation": (
                self.spectral_saturation_spin.value() / 100.0
            ),
            "spectral_smoothness": 0.91,
            "per_arm_variation": 0.04,
        }

    def secondary_spike_parameters(self, preview: bool = False) -> dict:
        spatial_scale = self.preview_scale if preview else 1.0
        return {
            "min_diameter": max(
                1.0, self.secondary_diameter_spin.value() * spatial_scale
            ),
            "feather": max(
                0.1, self.secondary_feather_spin.value() * spatial_scale
            ),
            "strength": self.secondary_strength_spin.value(),
            "length": max(
                2.0, self.secondary_length_spin.value() * spatial_scale
            ),
            "width": max(
                0.0, self.secondary_width_spin.value() * spatial_scale
            ),
            "angle": self.spike_angle_spin.value(),
            "blend": self.spike_blend_spin.value() / 100.0,
        }

    def update_preview(self) -> None:
        try:
            if self.before_button.isChecked():
                # In Linear mode the viewer intentionally shows the temporary
                # LogD working image, never the unstretched science data.
                shown = self.preview_working
            else:
                detection_black, detection_white = self.detection_parameters()
                (
                    result,
                    glow_selection,
                    spike_selection,
                    detection_mask,
                ) = make_effects(
                    self.preview_working,
                    detection_black,
                    detection_white,
                    self.glow_enabled.isChecked(),
                    self.glow_parameters(preview=True),
                    self.spikes_enabled.isChecked(),
                    self.spike_parameters(preview=True),
                    self.secondary_spikes_enabled.isChecked(),
                    self.secondary_spike_parameters(preview=True),
                    self.preview_label.exclusion_mask,
                    self.mask_glow_checkbox.isChecked(),
                    self.mask_spikes_checkbox.isChecked(),
                    # Always use the original v1.2-style effect engine.  Linear
                    # mode differs only by its temporary input/output mapping.
                    linear_mode=False,
                    progressive_linear_spikes=(
                        self.linear_mode_toggle.isChecked()
                    ),
                )
                shown = result

                if (
                    shown.shape[2] == 1
                    and (
                        self.detection_mask_checkbox.isChecked()
                        or
                        self.mask_checkbox.isChecked()
                        or (
                            self.spikes_enabled.isChecked()
                            and self.spike_mask_checkbox.isChecked()
                        )
                    )
                ):
                    shown = np.repeat(shown, 3, axis=2)

                if self.detection_mask_checkbox.isChecked():
                    shown = shown.copy()
                    alpha = np.clip(detection_mask * 0.72, 0.0, 0.72)
                    shown[..., 0] = (
                        shown[..., 0] * (1.0 - alpha) + alpha
                    )
                    if shown.shape[2] > 1:
                        shown[..., 1] = (
                            shown[..., 1] * (1.0 - alpha) + alpha
                        )
                        shown[..., 2] *= 1.0 - alpha

                if self.mask_checkbox.isChecked():
                    shown = shown.copy()
                    alpha = np.clip(glow_selection * 0.85, 0.0, 0.85)
                    shown[..., 0] = shown[..., 0] * (1.0 - alpha) + alpha
                    if shown.shape[2] > 1:
                        shown[..., 1] *= 1.0 - alpha
                        shown[..., 2] *= 1.0 - alpha

                if (
                    self.spikes_enabled.isChecked()
                    and self.spike_mask_checkbox.isChecked()
                ):
                    shown = shown.copy()
                    alpha = np.clip(spike_selection * 0.85, 0.0, 0.85)
                    shown[..., 0] *= 1.0 - alpha
                    if shown.shape[2] > 1:
                        shown[..., 1] = shown[..., 1] * (1.0 - alpha) + alpha
                        shown[..., 2] *= 1.0 - alpha

            if (
                self.show_starless_toggle.isChecked()
                and self.starless_preview is not None
            ):
                if self.starless_preview.shape[:2] != shown.shape[:2]:
                    raise ValueError(
                        "Starless preview dimensions no longer match the star mask."
                    )
                # Preview-only Screen recomposition: starless underneath,
                # current before/after star layer above it.
                shown = 1.0 - (
                    (1.0 - self.starless_preview) * (1.0 - shown)
                )

            pixmap = to_qpixmap(shown)
            self.preview_label.set_source_pixmap(pixmap)
        except Exception as exc:
            self.preview_label.image_label.setText(f"Preview error:\n{exc}")

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.schedule_preview()

    def reset_controls(self) -> None:
        self.linear_mode_toggle.setChecked(False)
        self.rational_logd_spin.setValue(9.0)
        self.detection_black_spin.setValue(0.20)
        self.detection_white_spin.setValue(0.90)
        self.detection_mask_checkbox.setChecked(False)
        self.blur_combo.setCurrentText("Moffat")
        self.glow_enabled.setChecked(False)
        self.radius_spin.setValue(10.0)
        self.diameter_slider.setValue(15)
        self.feather_spin.setValue(20.0)
        self.strength_spin.setValue(0.50)
        self.gamma_spin.setValue(0.70)
        self.glow_blend_spin.setValue(100)
        self.mask_checkbox.setChecked(False)
        self.spike_type_combo.setCurrentText("Newtonian")
        self.spikes_enabled.setChecked(False)
        self.spectral_checkbox.setChecked(False)
        self.spectral_strength_spin.setValue(100)
        self.spectral_saturation_spin.setValue(150)
        self.spike_diameter_slider.setValue(20)
        self.spike_feather_spin.setValue(30.0)
        self.spike_strength_spin.setValue(0.80)
        self.spike_length_spin.setValue(400.0)
        self.spike_width_spin.setValue(1.0)
        self.spike_angle_spin.setValue(0.0)
        self.spike_blend_spin.setValue(100)
        self.spike_mask_checkbox.setChecked(False)
        self.secondary_spikes_enabled.setChecked(False)
        self.secondary_diameter_spin.setValue(20)
        self.secondary_feather_spin.setValue(10.0)
        self.secondary_strength_spin.setValue(0.40)
        self.secondary_length_spin.setValue(80.0)
        self.secondary_width_spin.setValue(4.0)
        self.mask_glow_checkbox.setChecked(True)
        self.mask_spikes_checkbox.setChecked(True)
        self.show_mask_checkbox.setChecked(False)
        self.lasso_button.setChecked(False)
        self.preview_label.clear_exclusion_mask()
        self.before_button.setChecked(False)
        self.preview_resolution_combo.setCurrentIndex(
            self.preview_resolution_combo.findData(PREVIEW_MAX)
        )
        self.orientation_combo.setCurrentText("Flip vertical")
        self.schedule_preview()

    def set_push_progress(self, text: str, value: int) -> None:
        self.progress_label.setText(text)
        self.progress_label.setVisible(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(int(np.clip(value, 0, 100)))
        QApplication.processEvents()

    def push_to_siril(self) -> None:
        self.push_button.setEnabled(False)
        self.push_button.setText("Processing full resolution…")
        self.set_push_progress("Preparing full-resolution processing…", 0)

        try:
            detection_black, detection_white = self.detection_parameters()
            linear_mode = self.linear_mode_toggle.isChecked()
            processing_base = (
                rational_logd_preview(
                    self.base_full, self.rational_logd_spin.value()
                )
                if linear_mode
                else self.base_full
            )
            result, _, _, _ = make_effects(
                processing_base,
                detection_black,
                detection_white,
                self.glow_enabled.isChecked(),
                self.glow_parameters(preview=False),
                self.spikes_enabled.isChecked(),
                self.spike_parameters(preview=False),
                self.secondary_spikes_enabled.isChecked(),
                self.secondary_spike_parameters(preview=False),
                resize_mask(
                    self.preview_label.exclusion_mask,
                    self.base_full.shape[:2],
                ),
                self.mask_glow_checkbox.isChecked(),
                self.mask_spikes_checkbox.isChecked(),
                self.set_push_progress,
                # Preserve the established Aster glow and line-spike renderer.
                linear_mode=False,
                progressive_linear_spikes=linear_mode,
            )
            if linear_mode:
                self.set_push_progress("Reversing Rational LogD stretch…", 86)
                result = inverse_rational_logd(
                    result, self.rational_logd_spin.value()
                )
            self.set_push_progress("Preparing output pixels…", 88)
            result = apply_orientation(
                result, self.orientation_combo.currentText()
            )
            output_hwc = denormalize_image(result, self.scale)
            output = restore_layout(output_hwc, self.layout)

            self.set_push_progress("Transferring image to Siril…", 94)
            with self.siril.image_lock():
                current = self.siril.get_image()
                if current is None or current.data is None:
                    raise RuntimeError("The Siril image is no longer available.")
                if np.asarray(current.data).shape != output.shape:
                    raise RuntimeError(
                        "The image loaded in Siril changed size while Star Glow was open."
                    )
                undo_label = (
                    "Aster: Star Glow + Star Spikes"
                    if self.spikes_enabled.isChecked()
                    else "Aster: Star Glow"
                )
                self.siril.undo_save_state(undo_label)
                self.siril.set_image_pixeldata(np.ascontiguousarray(output))

            self.set_push_progress("Complete", 100)
            self.siril.log(
                "Star Glow: glow and optional spikes pushed successfully to Siril."
            )
            QMessageBox.information(
                self,
                APP_TITLE,
                "The full-resolution Screen composite was pushed to Siril.\n"
                "Use Save As in Siril to save it as FITS or TIFF.",
            )

            # Make repeated pushes non-cumulative unless the window is reopened.
        except Exception as exc:
            self.set_push_progress("Push failed", 0)
            self.siril.log(f"Star Glow error: {exc}")
            QMessageBox.critical(self, APP_TITLE, f"Could not push the result:\n\n{exc}")
        finally:
            self.push_button.setEnabled(True)
            self.push_button.setText("Push result to Siril")

def main() -> int:
    siril = s.SirilInterface()
    try:
        siril.connect()
        siril.cmd("requires", "1.4.0")

        app = QApplication.instance()
        owns_app = app is None
        if app is None:
            app = QApplication(sys.argv)

        window = StarGlowWindow(siril)
        window.show()
        window.raise_()
        window.activateWindow()

        if owns_app:
            return app.exec()

        # In case Siril already owns a Qt event loop.
        return 0

    except Exception as exc:
        message = f"{exc}\n\n{traceback.format_exc()}"
        try:
            siril.error_messagebox(APP_TITLE, message[:1000])
        except Exception:
            print(message, file=sys.stderr)
        return 1
    finally:
        # Do not disconnect while the GUI is still alive under an existing Qt loop.
        pass


if __name__ == "__main__":
    raise SystemExit(main())
