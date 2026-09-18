# Copyright (C) 2012-2026 team free-astro (see more in AUTHORS file)
# Copyright (C) 2026 StarNetAstro contributors
# Reference site is https://siril.org
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Maintained by StarNetAstro contributors for StarNet2 Siril integration support.
"""
StarNet star removal for Siril.

A self-contained wrapper around the third-party StarNet 2.5 (``starnet2``)
command-line tool. It removes stars from the currently loaded image or from a
loaded FITS sequence, and can optionally produce a star mask / star layer using
either StarNet's native outputs or a Siril-style descreen / subtraction.

The script offers a PyQt6 graphical interface when launched from the Scripts
menu, and a command-line interface for use with the ``pyscript`` command.

StarNet is a separate program written by Mikita (Nikita) Misiura
(https://starnetastro.com) and must be installed separately. Its location is
configured inside this script.

Usage (GUI):
    Launch from the Siril Scripts menu.

Usage (CLI, via the pyscript command):
    pyscript StarNet.py --linear --masks descreen
    pyscript StarNet.py --sequence --no-linear --stride 256

For FAQ and support guidance, see https://starnetastro.com/faq/.
Help requests for StarNet, DeepSNR, and Siril integration can be posted in:
https://www.cloudynights.com/topic/814885-starnet-v2-help-requests/

Original author: Adrian Knagg-Baugh
Current maintainer: StarNetAstro contributors
"""

# Version history
# 1.0.0 - Initial release: StarNet 2.5 support, PyQt6 GUI + CLI, single image
#         and FITS sequence processing, MTF autostretch for linear data, native
#         and Siril (descreen/subtract) star masks.
# 1.1.0 - Maintained by StarNetAstro: move to the StarNetAstro script category
#         and track script/product compatibility separately.

import sirilpy as s

# imagecodecs is required by tifffile to decode StarNet's LZW-compressed TIFF
# output (and to write compressed TIFFs).
s.ensure_installed("PyQt6", "numpy", "tifffile", "imagecodecs", "astropy")

import os
import re
import sys
import json
import time
import shutil
import warnings
import tempfile
import argparse
import threading
import subprocess
import base64
import platform
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import tifffile
from astropy.io import fits

from sirilpy import LogColor, SequenceType

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QIcon, QPixmap
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QGroupBox, QLabel, QLineEdit, QPushButton, QCheckBox, QComboBox, QSpinBox,
    QRadioButton, QButtonGroup, QProgressBar, QFileDialog, QMessageBox,
    QPlainTextEdit, QDialog,
)

SCRIPT_VERSION = "1.1.0"
SUPPORTED_STARNET2_MIN_VERSION = "2.5.0"
FEATURE_STARNET2_HIGHLIGHT_PROTECTION_MIN_VERSION = "2.5.3"
CONFIG_SUBDIR = "StarNet"
CONFIG_FILENAME = "settings.json"
STARNET_WEBSITE = "https://starnetastro.com/"
STARNET_DOWNLOAD_URL = "https://starnetastro.com/cli-tools/starnet/"
SUPPORT_URL = "https://www.cloudynights.com/topic/814885-starnet-v2-help-requests/"
LATEST_CLI_FEED_URL = "https://starnetastro.com/cli-tools/latest.json"
LATEST_CLI_FEED_SCHEMA = "starnetastro.cli-tools.latest.v1"
LATEST_CLI_CACHE_SECONDS = 24 * 60 * 60
LATEST_CLI_TIMEOUT_SECONDS = 1.0
ICON_B64 = (
    "LyogWFBNICovCnN0YXRpYyBjb25zdCBjaGFyICpTdGFyTmV0Mkljb25fWFBNW10gPSB7Ci8qIGNv"
    "bHVtbnMgcm93cyBjb2xvcnMgY2hhcnMtcGVyLXBpeGVsICovCiIyNCAyNCA1MSAxICIsCiIgIGMg"
    "YmxhY2siLAoiLiBjICMwMDAxMDEiLAoiWCBjICMwMTAxMDIiLAoibyBjICMwMTAxMDMiLAoiTyBj"
    "ICMwMTAyMDIiLAoiKyBjICMwMjAyMDQiLAoiQCBjICMwMjAyMDUiLAoiIyBjICMwMjAzMDUiLAoi"
    "JCBjICMwMjA0MDUiLAoiJSBjICMwMzA0MDUiLAoiJiBjICMwNDA1MDUiLAoiKiBjIGdyYXkyIiwK"
    "Ij0gYyAjMTUwNTA1IiwKIi0gYyAjMTYwNTA1IiwKIjsgYyAjMDUwNTE2IiwKIjogYyAjMjYwNDA0"
    "IiwKIj4gYyAjMzMwMDAwIiwKIiwgYyAjMzcwNDA0IiwKIjwgYyAjMDUwNDI2IiwKIjEgYyAjNDQw"
    "MDAwIiwKIjIgYyAjNDgwNDA0IiwKIjMgYyAjNTUwMDAwIiwKIjQgYyAjNTgwMzAzIiwKIjUgYyAj"
    "NjYwMDAwIiwKIjYgYyAjNjkwMzAzIiwKIjcgYyAjNzcwMDAwIiwKIjggYyAjN0EwMzAzIiwKIjkg"
    "YyAjMDUwNDQ4IiwKIjAgYyAjMDIwMDU1IiwKInEgYyAjMDUwMzU4IiwKIncgYyAjMDIwMDY2IiwK"
    "ImUgYyAjMDQwMjY5IiwKInIgYyAjMDUwMzY5IiwKInQgYyAjMDUwMzdBIiwKInkgYyAjOTkwMDAw"
    "IiwKInUgYyAjOUIwMjAyIiwKImkgYyAjQUMwMjAyIiwKInAgYyAjQkMwMTAxIiwKImEgYyAjQ0Mw"
    "MDAwIiwKInMgYyAjQ0QwMTAxIiwKImQgYyAjREQwMDAwIiwKImYgYyAjRUUwMDAwIiwKImcgYyBy"
    "ZWQiLAoiaCBjICMwNjAyOEEiLAoiaiBjICMwNjAyQUMiLAoiayBjICMwNjAxQkMiLAoibCBjICMw"
    "NjAxQ0QiLAoieiBjICMwNjAxREUiLAoieCBjICMwNjAwRUUiLAoiYyBjICMwNjAwRkYiLAoidiBj"
    "IE5vbmUiLAovKiBwaXhlbHMgKi8KInZ2dnZ2dnZ2dnZ2dnZ2dnZ2dnZ2dnZ2diIsCiJ2ICAgICAg"
    "ICAgICoqICAgICAgICAgIHYiLAoidiAgICAgICAgICAqKiAgICAgICAgICB2IiwKInYgICAxYWdn"
    "ZCAqKioqICAgICAgICAgdiIsCiJ2ICA+Z3kxMTUgKioqKisgICAgICAgIHYiLAoidiAgN2cgICAg"
    "KioqKioqICAgICAgICB2IiwKInYgIDNnNyAgICoqKioqKisgICAgICAgdiIsCiJ2ICAgeWdkNz0q"
    "KioqKioqICAgICAgIHYiLAoidiAgJCosdWdhPSoqKioqKioqKiokICB2IiwKInYqKioqKioyZzYq"
    "KioqKioqKioqKioqdiIsCiJ2KioqKioqKmc4KioqKioqKioqKioqKnYiLAoidiQqNjgyMmlmOioq"
    "KmNsKioqKmNxKiR2IiwKInYgIDRhZ2dwMioqKipjY3IqKipjciQgdiIsCiJ2ICAgKioqKioqKioq"
    "Y2x4OyoqY3IgIHYiLAoidiAgICAqKioqKioqKmN0eGoqKmN3ICB2IiwKInYgICAgKyoqKioqKipj"
    "cnFjOSpjdyAgdiIsCiJ2ICAgICAqKioqKioqY3Iqa3okY3cgIHYiLAoidiAgICAqKioqKioqKmNy"
    "KjxjdGN3ICB2IiwKInYgICAgKioqKioqKipjcioqaHh4dyAgdiIsCiJ2ICAgICoqKioqKioqY3Iq"
    "Kjt6Y3cgIHYiLAoidiAgICAqKioqKiorIGNxKioqcWMwICB2IiwKInYgICAkKioqKisgICAgKyoq"
    "KioqICAgdiIsCiJ2ICAgKioqJCAgICAgICAgKyoqKiAgIHYiLAoidnZ2dnZ2dnZ2dnZ2dnZ2dnZ2"
    "dnZ2dnZ2Igp9Owo="
)

# Auto-stretch constants, matched to Siril's defaults (see src/filters/mtf.c).
AS_DEFAULT_SHADOWS_CLIPPING = -2.80
AS_DEFAULT_TARGET_BACKGROUND = 0.25
MAD_NORM = 1.4826

# Mask / star-layer output kinds.
MASK_STARNET = "starnet-mask"      # StarNet native -m mask
MASK_UNSCREEN = "starnet-unscreen"  # StarNet native -n unscreen star layer
MASK_DESCREEN = "descreen"          # Siril-style descreen of starless from original
MASK_SUBTRACT = "subtract"          # original - starless
ALL_MASK_KINDS = (MASK_STARNET, MASK_UNSCREEN, MASK_DESCREEN, MASK_SUBTRACT)

MASK_LABELS = {
    MASK_STARNET: "Starmask",
    MASK_UNSCREEN: "Unscreen stars",
    MASK_DESCREEN: "Siril star layer (descreen)",
    MASK_SUBTRACT: "Siril star layer (subtraction)",
}

STRIDE_PRESET_LARGE = "large"
STRIDE_PRESET_STANDARD = "standard"
STRIDE_PRESET_SMALL = "small"
STRIDE_PRESETS = (
    (STRIDE_PRESET_LARGE, "Large", 384),
    (STRIDE_PRESET_STANDARD, "Standard", 256),
    (STRIDE_PRESET_SMALL, "Small", 128),
)
STRIDE_PRESET_VALUES = {key: value for key, _label, value in STRIDE_PRESETS}
STRIDE_TOOLTIP = (
    "Controls tile overlap during star removal."
)
STRIDE_HELP_TEXT = (
    "Stride selection guidance:\n"
    "Large: use for super-wide landscape images.\n"
    "Standard: use for telescope images.\n"
    "Small: slower; may help with extremely large stars."
)

# Filename prefixes used for outputs. Each output uses the same basename as the
# original image, with the matching prefix.
PREFIX_STARLESS = "starless_"
MASK_PREFIXES = {
    MASK_STARNET: "starnetmask_",
    MASK_UNSCREEN: "starnetdescreen_",
    MASK_DESCREEN: "descreen_mask_",
    MASK_SUBTRACT: "subtract_mask_",
}


def about_html():
    """Rich-text content for the About dialog."""
    return (
        "<h3>StarNet for Siril</h3>"
        f"<p>Script version {SCRIPT_VERSION}</p>"
        "<p>StarNet is a third-party neural-network application that removes "
        "stars from astronomical images, developed by Mikita (Nikita) Misiura. "
        "This script lets Siril hand an image (or sequence) to StarNet and bring "
        "back the starless result, with optional star-mask and star-layer "
        "outputs.</p>"
        f"<p><b>This script supports StarNet2 "
        f"{SUPPORTED_STARNET2_MIN_VERSION} or newer.</b> Earlier releases "
        "(StarNet++ v1 and v2.0) are not supported: the script detects them "
        "and asks you to upgrade.</p>"
        f"<p>Highlight protection controls require StarNet2 "
        f"{FEATURE_STARNET2_HIGHLIGHT_PROTECTION_MIN_VERSION} or newer.</p>"
        "<p>Siril loads the source image and provides pixel data to this "
        "script. The current StarNet2 command-line handoff uses temporary "
        "16-bit TIFF files, then the result is loaded back into Siril. The "
        "final Siril image follows the original image sample type unless "
        "Siril is configured to force 16-bit output.</p>"
        "<p>StarNet must be installed separately. Please make sure your StarNet "
        "2.5 installation runs correctly on its own before expecting it to work "
        "from this script. Install the StarNet2 CLI normally so "
        "<code>starnet2</code> is available on your system PATH or in the "
        "standard installer location. Set the executable location in this "
        "dialog only for nonstandard installations.</p>"
        "<p>If required, the path to the model weights file may also be specified "
        "however this is usually handled automatically by the Starnet 2.5 "
        "packaging.</p>"
        "<p>Siril's own built-in StarNet support handles older versions of Starnet "
        "(up to StarNet 2.0) and will remain available throughout the Siril 1.4.x "
        "series. The built-in interface will be retired in Siril 1.6, after which "
        "this script will be the only way to use StarNet with Siril and users will "
        "need to upgrade to Starnet 2.5.</p>"
        "<p>Help and bug reports: "
        f'<a href="{SUPPORT_URL}">'
        "StarNet/DeepSNR help requests on Cloudy Nights</a>.</p>"
        f'<p>Website: <a href="{STARNET_WEBSITE}">{STARNET_WEBSITE}</a></p>'
    )


# ---------------------------------------------------------------------------
# Image layout conversion
# ---------------------------------------------------------------------------
#
# Siril hands pixel data to sirilpy as a planar numpy array: shape (H, W) for a
# mono image, or (C, H, W) for a colour image, with dtype uint16 or float32.
# Internally we work in (H, W, C) float32 in the [0, 1] range, which is what the
# numpy maths and tifffile expect.

def siril_to_working(arr):
    """Convert a Siril pixel array to (H, W, C) float32 in [0, 1].

    Returns (working, meta) where meta records how to convert back.
    Float images whose peak exceeds 1.0 are rescaled to avoid clipping the
    highlights when written to an integer TIFF (mirrors the C StarNet tool).
    """
    is_uint16 = (arr.dtype == np.uint16)
    if arr.ndim == 2:
        mono = True
        hwc = arr[:, :, np.newaxis].astype(np.float32)
    elif arr.ndim == 3:
        mono = (arr.shape[0] == 1)
        hwc = np.transpose(arr, (1, 2, 0)).astype(np.float32)
    else:
        raise ValueError(f"Unsupported pixel data shape: {arr.shape}")

    rescaled = False
    if is_uint16:
        hwc /= 65535.0
    else:
        scale = 1.0
        peak = float(hwc.max()) if hwc.size else 1.0
        if peak > 1.0:
            scale = peak
            hwc /= scale
            rescaled = True
    hwc = np.clip(hwc, 0.0, 1.0)
    meta = {
        "mono": mono,
        "is_uint16": is_uint16,
        "rescaled": rescaled,
        "scale": 1.0 if is_uint16 else scale,
    }
    return hwc, meta


def working_to_siril(hwc, meta, force_uint16=False):
    """Convert an (H, W, C) float32 [0, 1] array back to Siril's layout/dtype.

    The output dtype matches the original image (meta['is_uint16']) unless
    force_uint16 is set (Siril's core.force_16bit preference), in which case it
    is always saved as 16-bit."""
    hwc = np.clip(hwc, 0.0, 1.0)
    if meta["mono"]:
        out = hwc[:, :, 0]
    else:
        out = np.transpose(hwc, (2, 0, 1))
    if meta["is_uint16"] or force_uint16:
        out = (out * 65535.0 + 0.5).astype(np.uint16)
    else:
        out = out * float(meta.get("scale", 1.0))
        out = out.astype(np.float32)
    return np.ascontiguousarray(out)


# ---------------------------------------------------------------------------
# TIFF I/O for the StarNet round-trip
# ---------------------------------------------------------------------------

def write_starnet_tiff(path, hwc):
    """Write an (H, W, C) float [0, 1] array as a 16-bit integer TIFF for StarNet."""
    a = hwc[:, :, 0] if hwc.shape[2] == 1 else hwc
    a = (np.clip(a, 0.0, 1.0) * 65535.0 + 0.5).astype(np.uint16)
    tifffile.imwrite(path, a, compression="lzw")


def read_image_tiff(path):
    """Read a TIFF written by StarNet, returning (H, W, C) float [0, 1]."""
    a = tifffile.imread(path)
    if a.dtype == np.uint8:
        a = a.astype(np.float32) / 255.0
    elif a.dtype == np.uint16:
        a = a.astype(np.float32) / 65535.0
    else:
        a = a.astype(np.float32)
    if a.ndim == 2:
        a = a[:, :, np.newaxis]
    return np.clip(a, 0.0, 1.0)


# ---------------------------------------------------------------------------
# MTF auto-stretch (faithful to Siril's unlinked autostretch)
# ---------------------------------------------------------------------------

def _mtf(x, m, lo, hi):
    """Midtones transfer function (scalar or ndarray x)."""
    rng = hi - lo
    xp = np.clip((x - lo) / rng, 0.0, 1.0) if rng > 0 else np.zeros_like(x)
    denom = (2.0 * m - 1.0) * xp - m
    return ((m - 1.0) * xp) / denom


def _mtf_scalar(x, m, lo, hi):
    if x <= lo:
        return 0.0
    if x >= hi:
        return 1.0
    xp = (x - lo) / (hi - lo)
    return ((m - 1.0) * xp) / ((2.0 * m - 1.0) * xp - m)


def compute_unlinked_mtf_params(hwc):
    """Per-channel (shadows, midtones, highlights), matching Siril's default
    unlinked autostretch (find_unlinked_midtones_balance_default)."""
    nb = hwc.shape[2]
    medians = [float(np.median(hwc[:, :, c])) for c in range(nb)]
    inverted = sum(1 for med in medians if med > 0.5)
    use_inverted = (inverted >= nb)

    params = []
    for c in range(nb):
        chan = hwc[:, :, c]
        median = medians[c]
        mad = float(np.median(np.abs(chan - median))) * MAD_NORM
        if mad == 0.0:
            mad = 0.001
        if not use_inverted:
            c0 = median + AS_DEFAULT_SHADOWS_CLIPPING * mad
            if c0 < 0.0:
                c0 = 0.0
            m2 = median - c0
            midtones = _mtf_scalar(m2, AS_DEFAULT_TARGET_BACKGROUND, 0.0, 1.0)
            params.append((c0, midtones, 1.0))
        else:
            c1 = median - AS_DEFAULT_SHADOWS_CLIPPING * mad
            if c1 > 1.0:
                c1 = 1.0
            m2 = c1 - median
            midtones = 1.0 - _mtf_scalar(m2, AS_DEFAULT_TARGET_BACKGROUND, 0.0, 1.0)
            params.append((0.0, midtones, c1))
    return params


def apply_unlinked_mtf(hwc, params):
    out = np.empty_like(hwc)
    for c, (lo, m, hi) in enumerate(params):
        out[:, :, c] = _mtf(hwc[:, :, c], m, lo, hi)
    return out


def apply_unlinked_pseudoinverse_mtf(hwc, params):
    """Approximate inverse of apply_unlinked_mtf, used to reverse the
    pre-stretch on the starless image (matches Siril's MTF_pseudoinverse)."""
    out = np.empty_like(hwc)
    for c, (shadows, midtones, highlights) in enumerate(params):
        y = hwc[:, :, c]
        a = (shadows + highlights) * midtones - shadows
        b = shadows * (1.0 - midtones)
        num = a * y + b
        den = (2.0 * midtones - 1.0) * y - midtones + 1.0
        out[:, :, c] = num / den
    return np.clip(out, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Siril-style star layer extraction
# ---------------------------------------------------------------------------

def descreen_mask(original, starless):
    """Star layer = descreen of starless from original: (o - s) / (1 - s),
    falling back to (o - s) where the denominator is ~0 (matches Siril)."""
    denom = 1.0 - starless
    sub = original - starless
    valid = np.abs(denom) >= 1e-6
    denom_safe = np.where(valid, denom, 1.0)
    screened = sub / denom_safe
    result = np.where(valid, screened, sub)
    return np.clip(result, 0.0, 1.0)


def subtract_mask(original, starless):
    """Star layer = original - starless (clamped to [0, 1])."""
    return np.clip(original - starless, 0.0, 1.0)


# ---------------------------------------------------------------------------
# FITS header / FILTER handling
# ---------------------------------------------------------------------------

def append_filter(current, tag):
    """Combine an existing FILTER value with a tag, matching Siril's
    update_filter_information append behaviour (e.g. 'Ha' -> 'Ha_Starless')."""
    current = (current or "").strip()
    if current and current != tag:
        return f"{current}_{tag}"[:70]
    return tag[:70]


def header_with_filter(original_header, filter_value):
    """Return a FITS header string based on original_header with the FILTER
    card updated (append semantics, like Siril's update_filter_information).

    Siril delivers and expects FITS headers as newline-separated card strings,
    so we parse and re-serialise with sep='\\n'. astropy verification warnings
    on Siril's (slightly non-standard) cards are suppressed."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if original_header:
                hdr = fits.Header.fromstring(original_header, sep="\n")
            else:
                hdr = fits.Header()
            hdr["FILTER"] = append_filter(str(hdr.get("FILTER", "")), filter_value)
            return hdr.tostring(sep="\n")
    except Exception:
        # Unparseable header: don't risk wiping metadata, keep it as-is.
        return original_header


# ---------------------------------------------------------------------------
# StarNet executable: version detection and invocation
# ---------------------------------------------------------------------------

_starnet_lock = threading.Lock()
_latest_cli_feed_cache = {"checked_at": 0.0, "payload": None}

# Names to look for on PATH when no explicit executable is configured.
STARNET_EXE_NAMES = ("starnet2", "starnet2.exe", "StarNet2", "StarNet2.exe")
STARNET_DEFAULT_EXE_PATHS = (
    r"C:\Program Files\StarNet2\bin\starnet2.exe",
    "/usr/local/bin/starnet2",
    "/usr/bin/starnet2",
)


class StarNetError(Exception):
    pass


def find_starnet_on_path():
    """Return the path to a StarNet 2 executable on PATH, or None."""
    for name in STARNET_EXE_NAMES:
        found = shutil.which(name)
        if found:
            return found
    return None


def find_starnet_installed_default():
    """Return a StarNet executable from a known installer path, or None."""
    for path in STARNET_DEFAULT_EXE_PATHS:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    return None


def resolve_exe(configured):
    """Resolve the StarNet executable to use.

    PATH and known CLI installer locations are preferred. A configured path is a
    last-resort fallback for nonstandard installations. Returns a path string,
    or None if nothing is found.
    """
    def configured_exe():
        if not configured:
            return None
        candidate = os.path.expanduser(configured.strip())
        if not candidate:
            return None
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
        return shutil.which(candidate)

    return find_starnet_on_path() or find_starnet_installed_default() or configured_exe()


def display_exe_path(exe):
    """Format an executable path for UI display without changing execution."""
    if not exe:
        return ""
    path = os.path.normpath(exe)
    root, ext = os.path.splitext(path)
    if ext.lower() == ".exe":
        return root
    return path


def parse_version_tuple(text):
    if not text:
        return None
    match = re.search(r"(\d+)\.(\d+)(?:\.(\d+))?", str(text))
    if not match:
        return None
    return (
        int(match.group(1)),
        int(match.group(2)),
        int(match.group(3) or 0),
    )


def format_version(version):
    return ".".join(str(part) for part in version)


def current_latest_cli_platform_key(system=None, machine=None):
    system = (system or platform.system()).lower()
    machine = (machine or platform.machine()).lower()
    if system.startswith("win"):
        return "windows-x64"
    if system == "darwin":
        if machine in ("arm64", "aarch64"):
            return "macos-arm64"
        return "macos-x64"
    if system == "linux":
        return "linux-x64"
    return None


def latest_cli_platform_label(platform_key):
    return {
        "windows-x64": "Windows",
        "linux-x64": "Linux",
        "macos-x64": "macOS Intel",
        "macos-arm64": "macOS Apple silicon",
    }.get(platform_key, platform_key or "this platform")


def fetch_latest_cli_feed(url=LATEST_CLI_FEED_URL):
    now = time.time()
    cached = _latest_cli_feed_cache.get("payload")
    checked_at = _latest_cli_feed_cache.get("checked_at", 0.0)
    if cached is not None and now - checked_at < LATEST_CLI_CACHE_SECONDS:
        return cached
    request = urllib.request.Request(
        url,
        headers={"User-Agent": f"StarNetAstro-Siril/{SCRIPT_VERSION}"},
    )
    with urllib.request.urlopen(request, timeout=LATEST_CLI_TIMEOUT_SECONDS) as response:
        payload = json.loads(response.read().decode("utf-8"))
    _latest_cli_feed_cache["checked_at"] = now
    _latest_cli_feed_cache["payload"] = payload
    return payload


def latest_cli_update_notice(product_key, installed_version, *,
                             platform_key=None, fetcher=None):
    if installed_version is None:
        return None
    platform_key = platform_key or current_latest_cli_platform_key()
    if not platform_key:
        return None
    try:
        payload = (fetcher or fetch_latest_cli_feed)()
    except (OSError, ValueError, urllib.error.URLError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("schema") != LATEST_CLI_FEED_SCHEMA:
        return None
    tool = payload.get("tools", {}).get(product_key)
    if not isinstance(tool, dict):
        return None
    latest = tool.get("latest", {}).get(platform_key)
    if not isinstance(latest, dict):
        return None
    latest_version = parse_version_tuple(latest.get("version"))
    if latest_version is None or latest_version <= installed_version:
        return None
    display_name = tool.get("display_name") or "StarNet2"
    download_url = tool.get("download_url") or STARNET_DOWNLOAD_URL
    return (
        f"Update available: {display_name} {format_version(latest_version)} "
        f"is available for {latest_cli_platform_label(platform_key)}.\n"
        f"Download: {download_url}"
    )


def missing_starnet_message(location_hint, configured=None):
    prefix = (
        f"StarNet executable not found or not executable: {configured.strip()}"
        if configured and configured.strip()
        else "StarNet ('starnet2') was not found in a standard CLI installer "
             "location or on PATH."
    )
    return (
        f"{prefix}\n"
        f"Download the StarNet2 CLI installer from {STARNET_DOWNLOAD_URL} "
        f"and run the installer.\n"
        f"Alternatively, use {location_hint} to select an existing executable."
    )


def detect_starnet_version_info(configured):
    """Probe the configured (or PATH-resolved) StarNet executable.

    Returns (status, message) where status is one of:
        'ok'       - StarNet 2.5 or newer, usable
        'upgrade'  - an older StarNet (v1 / v2.0) was found; user must upgrade
        'missing'  - no executable could be found
        'unknown'  - ran but the version could not be identified
    """
    exe = resolve_exe(configured)
    if not exe:
        if configured and configured.strip():
            return ("missing",
                    missing_starnet_message("Set paths", configured),
                    None)
        return ("missing",
                missing_starnet_message("Set paths"),
                None)

    exe_dir = os.path.dirname(os.path.abspath(exe))
    blobs = []
    with _starnet_lock:
        # --help carries the clean banner ("StarNet2 v2.5.1", "ONNX Runtime
        # backend"); --version uses TCLAP's "<prog>  version: 2.5.1" form; the
        # no-argument run catches the old "StarNet++ v2.0" banner.
        for extra in (["--help"], ["--version"], []):
            try:
                proc = subprocess.run(
                    [exe] + extra, cwd=exe_dir,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    timeout=30, text=True, errors="replace",
                )
                blobs.append(proc.stdout or "")
            except subprocess.TimeoutExpired:
                blobs.append("")
            except OSError as exc:
                return "missing", f"Could not run the StarNet executable: {exc}", exe
    blob = "\n".join(blobs)
    low = blob.lower()

    # Old "++" family (StarNet v1 and v2.0) - advise upgrade.
    if "starnet++" in low:
        if "v2.0" in low:
            return ("upgrade",
                    "StarNet++ v2.0 was detected. This script requires StarNet 2.5 "
                    "or newer - please upgrade.",
                    exe)
        return ("upgrade",
                "An old StarNet (v1) executable was detected. This script requires "
                "StarNet 2.5 or newer - please upgrade.",
                exe)

    # New ONNX StarNet2. The version appears either as "StarNet2 v2.5.1" (help
    # banner) or "starnet2  version: 2.5.1" (TCLAP --version).
    version = None
    for pattern in (
            r"starnet2[\s_]+v?(\d+)\.(\d+)(?:\.(\d+))?",
            r"version:\s*v?(\d+)\.(\d+)(?:\.(\d+))?"):
        match = re.search(pattern, low)
        if match:
            version = (
                int(match.group(1)),
                int(match.group(2)),
                int(match.group(3) or 0),
            )
            break

    if version is not None:
        major, minor, patch = version
        if (major, minor) >= (2, 5):
            return (
                "ok",
                f"StarNet {major}.{minor}.{patch} found\n"
                f"Path: {display_exe_path(exe)}",
                exe,
            )
        return ("upgrade",
                f"StarNet {major}.{minor}.{patch} was detected. This script "
                "requires StarNet 2.5 or newer - please upgrade.",
                exe)

    # No version number, but the ONNX StarNet2 identifies itself unambiguously
    # (and we have already excluded the "++" family above).
    if "onnx" in low and "starnet" in low:
        return (
            "ok",
            f"StarNet 2.5 (ONNX) found\nPath: {display_exe_path(exe)}",
            exe,
        )
    if "starnet2" in low:
        return "ok", f"StarNet 2 found\nPath: {display_exe_path(exe)}", exe

    return ("unknown",
            f"The executable at {exe} did not report a recognisable StarNet "
            "version. StarNet 2.5 or newer is required.",
            exe)


def detect_starnet_version(configured):
    status, message, _exe = detect_starnet_version_info(configured)
    return status, message


def build_starnet_argv(exe, in_tiff, out_tiff, *, stride=None, upsample=False,
                       weights=None, mask_tiff=None, unscreen_tiff=None,
                       protect_highlights=True):
    argv = [exe, "-i", in_tiff, "-o", out_tiff]
    if stride is not None:
        argv += ["-s", str(stride)]
    if upsample:
        argv.append("-u")
    if weights:
        argv += ["-w", weights]
    if mask_tiff:
        argv += ["-m", mask_tiff]
    if unscreen_tiff:
        argv += ["-n", unscreen_tiff]
    if not protect_highlights:
        argv.append("-d")
    return argv


def starnet_supports_disable_highlights(exe):
    exe_dir = os.path.dirname(os.path.abspath(exe))
    try:
        proc = subprocess.run(
            [exe, "--help"], cwd=exe_dir,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            timeout=30, text=True, errors="replace",
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    help_text = (proc.stdout or "").lower()
    return (
        "disable-highlights-protection" in help_text
        or (
            "highlight" in help_text
            and re.search(r"(^|\s)-d([,\s]|$)", help_text) is not None
        )
    )


def starnet_highlight_support_for_configured(configured):
    exe = resolve_exe(configured)
    if not exe:
        return False
    return starnet_supports_disable_highlights(exe)


def run_starnet(exe, in_tiff, out_tiff, *, stride=None, upsample=False,
                weights=None, mask_tiff=None, unscreen_tiff=None,
                protect_highlights=True, progress_cb=None, log_cb=None,
                cancel_cb=None):
    """Run StarNet on in_tiff, producing out_tiff (and optionally mask/unscreen
    TIFFs). Raises StarNetError on failure. Returns True, or 'cancelled'."""
    exe_dir = os.path.dirname(os.path.abspath(exe))
    argv = build_starnet_argv(
        exe, in_tiff, out_tiff,
        stride=stride, upsample=upsample, weights=weights,
        mask_tiff=mask_tiff, unscreen_tiff=unscreen_tiff,
        protect_highlights=protect_highlights,
    )

    pct_re = re.compile(rb"(\d{1,3}(?:\.\d+)?)\s*%")
    output_tail = []

    with _starnet_lock:
        try:
            proc = subprocess.Popen(
                argv, cwd=exe_dir,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            )
        except OSError as exc:
            raise StarNetError(f"Could not launch StarNet: {exc}") from exc

        def reader():
            fd = proc.stdout.fileno()
            pending = b""
            while True:
                try:
                    data = os.read(fd, 4096)
                except OSError:
                    break
                if not data:
                    break
                pending += data
                parts = re.split(rb"[\r\n]+", pending)
                pending = parts.pop()
                for token in parts + [pending]:
                    last = None
                    for m in pct_re.finditer(token):
                        last = m
                    if last and progress_cb:
                        try:
                            progress_cb(min(1.0, float(last.group(1)) / 100.0))
                        except Exception:
                            pass
                    # Progress lines (containing a percentage) drive the bar only;
                    # don't echo them to the log. Log other informational lines.
                    if last is None and log_cb and token in parts:
                        text = token.decode("utf-8", "replace").strip()
                        if text:
                            log_cb(text)
                    if last is None and token in parts:
                        text = token.decode("utf-8", "replace").strip()
                        if text:
                            output_tail.append(text)
                            del output_tail[:-20]
            if pending:
                text = pending.decode("utf-8", "replace").strip()
                if text and pct_re.search(pending) is None:
                    output_tail.append(text)
                    del output_tail[:-20]
                    if log_cb:
                        log_cb(text)

        thread = threading.Thread(target=reader, daemon=True)
        thread.start()

        cancelled = False
        while proc.poll() is None:
            if cancel_cb and cancel_cb():
                cancelled = True
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                break
            time.sleep(0.1)
        thread.join(timeout=2)

    if cancelled:
        return "cancelled"
    if proc.returncode != 0:
        details = "\n".join(output_tail[-10:])
        message = f"StarNet exited with status {proc.returncode}."
        if details:
            message += f"\nLast output:\n{details}"
        raise StarNetError(message)
    if not os.path.isfile(out_tiff):
        raise StarNetError("StarNet did not produce an output file.")
    missing = []
    if mask_tiff and not os.path.isfile(mask_tiff):
        missing.append("native star mask")
    if unscreen_tiff and not os.path.isfile(unscreen_tiff):
        missing.append("native unscreen star layer")
    if missing:
        raise StarNetError("StarNet did not produce requested output(s): "
                           + ", ".join(missing) + ".")
    return True


# ---------------------------------------------------------------------------
# Core processing: one image
# ---------------------------------------------------------------------------

class StarNetOptions:
    """Plain options container shared by the GUI and the CLI."""

    def __init__(self):
        self.exe = ""
        self.weights = ""
        self.linear = True
        self.stride_preset = STRIDE_PRESET_STANDARD
        self.custom_stride = False
        self.stride = 256
        self.upsample = False
        self.protect_highlights = True
        self.masks = []  # subset of ALL_MASK_KINDS

    def stride_value(self):
        if self.custom_stride:
            v = int(self.stride)
        else:
            v = STRIDE_PRESET_VALUES.get(
                self.stride_preset, STRIDE_PRESET_VALUES[STRIDE_PRESET_STANDARD])
        if v % 2:
            v += 1
        return max(2, min(512, v))

    def to_dict(self):
        return {
            "exe": self.exe, "weights": self.weights, "linear": self.linear,
            "stride_preset": self.stride_preset,
            "custom_stride": self.custom_stride, "stride": self.stride,
            "upsample": self.upsample,
            "protect_highlights": self.protect_highlights,
            "masks": list(self.masks),
        }

    @classmethod
    def from_dict(cls, d):
        o = cls()
        o.exe = d.get("exe", "")
        o.weights = d.get("weights", "")
        o.linear = bool(d.get("linear", True))
        has_stride_preset = "stride_preset" in d
        o.stride_preset = d.get("stride_preset", STRIDE_PRESET_STANDARD)
        if o.stride_preset not in STRIDE_PRESET_VALUES:
            o.stride_preset = STRIDE_PRESET_STANDARD
        o.custom_stride = bool(d.get("custom_stride", False))
        o.stride = int(d.get("stride", 256))
        if not has_stride_preset and o.custom_stride and o.stride == 256:
            o.custom_stride = False
            o.stride_preset = STRIDE_PRESET_STANDARD
        o.upsample = bool(d.get("upsample", False))
        o.protect_highlights = bool(d.get("protect_highlights", True))
        o.masks = [m for m in d.get("masks", []) if m in ALL_MASK_KINDS]
        return o


def process_array(arr, opts, force_16bit=False, progress_cb=None,
                  log_cb=None, cancel_cb=None):
    """Run the full StarNet pipeline on a single Siril pixel array.

    Returns a dict with key 'starless' (always) and one key per enabled mask
    kind, each holding a Siril-layout array. The output dtype matches the input
    unless force_16bit is set, in which case all outputs are 16-bit.
    Returns None if cancelled.
    """
    exe = resolve_exe(opts.exe)
    if not exe:
        raise StarNetError(missing_starnet_message("--exe"))
    if not opts.protect_highlights and not starnet_supports_disable_highlights(exe):
        raise StarNetError(
            "Disabling highlight protection requires StarNet2 "
            f"{FEATURE_STARNET2_HIGHLIGHT_PROTECTION_MIN_VERSION} or newer.")

    working, meta = siril_to_working(arr)
    if meta["rescaled"] and log_cb:
        log_cb("Pixel values exceed 1.0; rescaling to avoid clipping the highlights.")

    # Pre-stretch linear data so StarNet sees a stretched image. 'working' is now
    # the image in the exact domain StarNet operates in (stretched if linear).
    params = None
    if opts.linear:
        params = compute_unlinked_mtf_params(working)
        working = apply_unlinked_mtf(working, params)
        if log_cb:
            log_cb("Linear mode: applied MTF autostretch to the StarNet input.")

    def fix_mono(layer):
        return layer[:, :, :1] if (meta["mono"] and layer.shape[2] == 3) else layer

    def destretch(layer):
        if opts.linear and params is not None:
            return apply_unlinked_pseudoinverse_mtf(layer, params)
        return layer

    tmp = tempfile.mkdtemp(prefix="siril_starnet_")
    try:
        in_tiff = os.path.join(tmp, "input.tif")
        out_tiff = os.path.join(tmp, "starless.tif")
        mask_tiff = os.path.join(tmp, "mask.tif") if MASK_STARNET in opts.masks else None
        unscreen_tiff = os.path.join(tmp, "unscreen.tif") if MASK_UNSCREEN in opts.masks else None

        write_starnet_tiff(in_tiff, working)

        result = run_starnet(
            exe, in_tiff, out_tiff,
            stride=opts.stride_value(), upsample=opts.upsample,
            weights=(opts.weights or None),
            mask_tiff=mask_tiff, unscreen_tiff=unscreen_tiff,
            protect_highlights=opts.protect_highlights,
            progress_cb=progress_cb, log_cb=log_cb, cancel_cb=cancel_cb,
        )
        if result == "cancelled":
            return None

        starless = fix_mono(read_image_tiff(out_tiff))  # StarNet (stretched) domain

        # Build every star layer in the SAME domain StarNet worked in, so the
        # Siril descreen/subtraction match the background cancellation that
        # StarNet's own outputs achieve, then de-stretch all of them uniformly
        # (just like the starless) back to the original image domain.
        layers = {}
        for kind in opts.masks:
            if kind == MASK_STARNET and mask_tiff and os.path.isfile(mask_tiff):
                layers[kind] = fix_mono(read_image_tiff(mask_tiff))
            elif kind == MASK_UNSCREEN and unscreen_tiff and os.path.isfile(unscreen_tiff):
                layers[kind] = fix_mono(read_image_tiff(unscreen_tiff))
            elif kind == MASK_DESCREEN:
                layers[kind] = descreen_mask(working, starless)
            elif kind == MASK_SUBTRACT:
                layers[kind] = subtract_mask(working, starless)

        outputs = {"starless": working_to_siril(destretch(starless), meta, force_16bit)}
        for kind, layer in layers.items():
            outputs[kind] = working_to_siril(destretch(layer), meta, force_16bit)
        return outputs
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Siril-facing flows
# ---------------------------------------------------------------------------

def _get_config(siril, group, key, default=None):
    try:
        value = siril.get_siril_config(group, key)
        return default if value is None else value
    except Exception:
        return default


def _config_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("1", "true", "yes", "on"):
            return True
        if text in ("0", "false", "no", "off", ""):
            return False
    return default


def _force_16bit(siril):
    return _config_bool(_get_config(siril, "core", "force_16bit", False))


def _output_path(siril, wd, name_base):
    """Build the output FITS path honouring the preferred extension and the FITS
    compression preference (a compressed file gains a .fz suffix)."""
    ext = _get_config(siril, "core", "extension", ".fit") or ".fit"
    compressed = _config_bool(_get_config(siril, "compression", "enabled", False))
    filename = f"{name_base}{ext}"
    if compressed:
        filename += ".fz"
    return os.path.join(wd, filename)


def image_basename(filename):
    base = os.path.basename(filename or "") or "image"
    low = base.lower()
    for suffix in (
        ".fits.fz", ".fit.fz", ".fts.fz",
        ".xisf", ".fits", ".fit", ".fts", ".tiff", ".tif",
    ):
        if low.endswith(suffix):
            return base[:-len(suffix)] or "image"
    stem = os.path.splitext(base)[0]
    return stem or "image"


def process_single_image(siril, opts, progress_cb=None, log_cb=None,
                         cancel_cb=None):
    """Replace the loaded image with the starless result and save any enabled
    star masks / layers as FITS files in the working directory."""
    arr = siril.get_image_pixeldata()
    try:
        header = siril.get_image_fits_header() or ""
    except Exception:
        header = ""
    try:
        orig_filter = siril.get_image_keywords().filter
    except Exception:
        orig_filter = ""
    try:
        fit = siril.get_image(with_pixels=False)
        src_name = fit.filename if fit and fit.filename else "image"
    except Exception:
        src_name = "image"
    basename = image_basename(src_name)

    if min(arr.shape[-1], arr.shape[-2]) < 512 and log_cb:
        log_cb("Warning: images smaller than 512 px may fail in StarNet.")

    outputs = process_array(arr, opts, force_16bit=_force_16bit(siril),
                            progress_cb=progress_cb, log_cb=log_cb,
                            cancel_cb=cancel_cb)
    if outputs is None:
        return False

    # Replace the loaded image with the starless result.
    siril.undo_save_state(f"StarNet (starless), stride={opts.stride_value() or 'default'}")
    with siril.image_lock():
        siril.set_image_pixeldata(outputs["starless"])
    # Tag the loaded starless image's FILTER card. update_key acts on the loaded
    # image and must run outside image_lock (it needs the processing thread).
    try:
        value = append_filter(orig_filter, "Starless")
        siril.cmd("update_key", "FILTER", f'"{value}"' if " " in value else value)
    except Exception as exc:
        if log_cb:
            log_cb(f"Could not update FILTER metadata: {exc}")

    # Save the star mask / layer extras, using the original basename + each
    # prefix, and the preferred extension / compression.
    wd = siril.get_siril_wd()
    for kind in opts.masks:
        if kind not in outputs:
            continue
        filename = _output_path(siril, wd, f"{MASK_PREFIXES[kind]}{basename}")
        mask_header = header_with_filter(header, "Starmask")
        try:
            siril.save_image_file(outputs[kind], header=mask_header, filename=filename)
            if log_cb:
                log_cb(f"Saved {MASK_LABELS[kind]} to {os.path.basename(filename)}")
        except Exception as exc:
            if log_cb:
                log_cb(f"Could not save {MASK_LABELS[kind]}: {exc}")
    return True


def selected_sequence_indices(seq):
    imgparam = getattr(seq, "imgparam", None)
    selected = [
        i for i, param in enumerate(imgparam or [])
        if bool(getattr(param, "incl", False))
    ]
    if selected:
        return selected
    return list(range(seq.number))


def process_sequence(siril, opts, progress_cb=None, log_cb=None,
                     cancel_cb=None):
    """Process every selected frame of the loaded FITS sequence, building a new
    starless sequence (plus one new sequence per enabled star-mask kind)."""
    seq = siril.get_seq()
    if seq.type != SequenceType.SEQ_REGULAR:
        raise StarNetError(
            "Sequence processing requires a FITS sequence (SER / FITSEQ are not "
            "supported by this script). Convert the sequence to FITS first.")

    indices = selected_sequence_indices(seq)
    n = len(indices)
    total = seq.number
    force_16bit = _force_16bit(siril)
    mask_prefixes = {kind: MASK_PREFIXES[kind] for kind in opts.masks}
    if log_cb:
        if n == total:
            log_cb(f"Processing {n} frame(s) from sequence '{seq.seqname}'.")
        else:
            log_cb(f"Processing {n} selected frame(s) from sequence '{seq.seqname}'.")

    for pos, i in enumerate(indices):
        if cancel_cb and cancel_cb():
            return False
        if log_cb:
            log_cb(f"Frame {pos + 1}/{n} (sequence frame {i + 1}/{total})")

        def frame_progress(frac, base=pos):
            if progress_cb:
                progress_cb((base + frac) / n)

        arr = siril.get_seq_frame_pixeldata(i)
        outputs = process_array(arr, opts, force_16bit=force_16bit,
                                progress_cb=frame_progress, log_cb=None,
                                cancel_cb=cancel_cb)
        if outputs is None:
            return False

        siril.set_seq_frame_pixeldata(i, outputs["starless"], prefix=PREFIX_STARLESS)
        for kind, prefix in mask_prefixes.items():
            if kind in outputs:
                siril.set_seq_frame_pixeldata(i, outputs[kind], prefix=prefix)
        if progress_cb:
            progress_cb((pos + 1) / n)

    siril.create_new_seq(PREFIX_STARLESS)
    for prefix in mask_prefixes.values():
        siril.create_new_seq(prefix)
    if log_cb:
        log_cb("Sequence processing complete.")
    return True


# ---------------------------------------------------------------------------
# Settings persistence (self-contained, in the script's own config dir)
# ---------------------------------------------------------------------------

def settings_path(siril):
    cfg_dir = Path(siril.get_siril_configdir()) / CONFIG_SUBDIR
    cfg_dir.mkdir(parents=True, exist_ok=True)
    return cfg_dir / CONFIG_FILENAME


def load_options(siril):
    try:
        data = json.loads(settings_path(siril).read_text(encoding="utf-8"))
        return StarNetOptions.from_dict(data)
    except FileNotFoundError:
        return StarNetOptions()
    except Exception:
        return StarNetOptions()


def save_options(siril, opts):
    try:
        settings_path(siril).write_text(
            json.dumps(opts.to_dict(), indent=2), encoding="utf-8")
    except Exception as exc:
        siril.log(f"StarNet: could not save settings: {exc}", LogColor.SALMON)


# ---------------------------------------------------------------------------
# PyQt6 worker thread
# ---------------------------------------------------------------------------

class StarNetWorker(QThread):
    progress = pyqtSignal(float)
    message = pyqtSignal(str)
    finished_ok = pyqtSignal(bool, str)

    def __init__(self, siril, opts, sequence_mode):
        super().__init__()
        self.siril = siril
        self.opts = opts
        self.sequence_mode = sequence_mode
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def _cancelled(self):
        return self._cancel

    def run(self):
        try:
            if self.sequence_mode:
                ok = process_sequence(
                    self.siril, self.opts,
                    progress_cb=self.progress.emit, log_cb=self.message.emit,
                    cancel_cb=self._cancelled)
            else:
                ok = process_single_image(
                    self.siril, self.opts,
                    progress_cb=self.progress.emit, log_cb=self.message.emit,
                    cancel_cb=self._cancelled)
            if self._cancel:
                self.finished_ok.emit(False, "Cancelled.")
            elif ok:
                self.finished_ok.emit(True, "StarNet finished.")
            else:
                self.finished_ok.emit(False, "StarNet did not complete.")
        except Exception as exc:
            self.finished_ok.emit(False, str(exc))


class LatestCliVersionWorker(QThread):
    notice = pyqtSignal(str)

    def __init__(self, installed_version):
        super().__init__()
        self.installed_version = installed_version

    def run(self):
        notice = latest_cli_update_notice("starnet2", self.installed_version)
        if notice:
            self.notice.emit(notice)


def starnet_window_icon():
    pixmap = QPixmap()
    if pixmap.loadFromData(base64.b64decode(ICON_B64), "XPM"):
        return QIcon(pixmap)
    return QIcon()


# ---------------------------------------------------------------------------
# PyQt6 GUI
# ---------------------------------------------------------------------------

class StarNetGUI(QMainWindow):
    def __init__(self, siril):
        super().__init__()
        self.siril = siril
        self.opts = load_options(siril)
        self.worker = None
        self.update_worker = None
        self.has_image = False
        self.has_sequence = False
        try:
            self.has_image = siril.is_image_loaded()
        except Exception:
            pass
        try:
            self.has_sequence = siril.is_sequence_loaded()
        except Exception:
            pass

        self.setWindowTitle("StarNet star removal")
        self._icon = starnet_window_icon()
        self.setWindowIcon(self._icon)
        self._build_ui()
        self._load_into_ui()
        self._check_executable()

    # -- UI construction --------------------------------------------------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        self.exe_path = ""
        self.weights_path = ""
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        self.update_label = QLabel("")
        self.update_label.setWordWrap(True)
        self.update_label.setStyleSheet("color: #d99200;")
        layout.addWidget(self.update_label)
        status_line = QLabel("")
        status_line.setFixedHeight(1)
        status_line.setStyleSheet("border-top: 1px solid gray;")
        layout.addWidget(status_line)

        stride_layout = QGridLayout()
        stride_layout.setColumnStretch(1, 1)
        stride_help = QLabel(STRIDE_HELP_TEXT)
        stride_help.setWordWrap(True)
        stride_help.setStyleSheet("color: gray;")
        stride_layout.addWidget(stride_help, 0, 0, 1, 2)

        stride_label = QLabel("Stride:")
        stride_label.setToolTip(STRIDE_TOOLTIP)
        self.stride_combo = QComboBox()
        self.stride_combo.setToolTip(STRIDE_TOOLTIP)
        for key, label, _value in STRIDE_PRESETS:
            self.stride_combo.addItem(label, key)
        stride_layout.addWidget(stride_label, 1, 0)
        stride_layout.addWidget(self.stride_combo, 1, 1)

        self.stride_check = QCheckBox("Use custom stride")
        self.stride_check.setToolTip(
            "Use a custom numeric tile stride instead of the preset values.")
        self.stride_check.toggled.connect(self._sync_enabled)
        self.stride_spin = QSpinBox()
        self.stride_spin.setRange(2, 512)
        self.stride_spin.setSingleStep(2)
        self.stride_spin.setToolTip(
            "Custom tile stride. Must be an even number between 2 and 512.")
        stride_layout.addWidget(self.stride_check, 2, 0)
        stride_layout.addWidget(self.stride_spin, 2, 1)
        layout.addLayout(stride_layout)

        # Processing options
        opt_group = QGroupBox("Processing Options")
        opt_layout = QVBoxLayout(opt_group)

        self.upsample_check = QCheckBox("2x upsampling")
        self.upsample_check.setToolTip(
            "Use intermediate 2x upsampling. Slower and more memory-hungry, but "
            "can help with very tight stars.")
        opt_layout.addWidget(self.upsample_check)

        self.linear_check = QCheckBox("Linear data")
        self.linear_check.setToolTip(
            "Enable when the loaded image is unstretched (linear). An automatic "
            "stretch is applied before StarNet and reversed afterwards. Disable "
            "if the image has already been stretched.")
        opt_layout.addWidget(self.linear_check)

        self.protect_highlights_check = QCheckBox("Protect highlights")
        self.protect_highlights_check.setToolTip(
            "Preserve clipped or near-clipped highlights when assigning the "
            "starless result. Disabling this requires StarNet2 "
            f"{FEATURE_STARNET2_HIGHLIGHT_PROTECTION_MIN_VERSION} or newer.")
        opt_layout.addWidget(self.protect_highlights_check)
        layout.addWidget(opt_group)

        # Mask outputs
        mask_group = QGroupBox("Output Options")
        mask_layout = QVBoxLayout(mask_group)
        self.mask_checks = {}
        tooltips = {
            MASK_STARNET: "Save the star mask produced directly by StarNet.",
            MASK_UNSCREEN: "Save the star layer (unscreened) produced directly by StarNet.",
            MASK_DESCREEN: "Build a star layer in Siril by descreening the starless image from the original.",
            MASK_SUBTRACT: "Build a star layer in Siril by subtracting the starless image from the original.",
        }
        for kind in ALL_MASK_KINDS:
            cb = QCheckBox(MASK_LABELS[kind])
            cb.setToolTip(tooltips[kind])
            mask_layout.addWidget(cb)
            self.mask_checks[kind] = cb
        layout.addWidget(mask_group)

        # Target
        target_group = QGroupBox("Apply to")
        target_layout = QHBoxLayout(target_group)
        self.target_image = QRadioButton("Loaded image")
        self.target_seq = QRadioButton("Loaded sequence")
        self.target_image.setToolTip("Process the single image currently loaded in Siril.")
        self.target_seq.setToolTip("Process every selected frame of the loaded FITS sequence.")
        self.target_group = QButtonGroup(self)
        self.target_group.addButton(self.target_image)
        self.target_group.addButton(self.target_seq)
        target_layout.addWidget(self.target_image)
        target_layout.addWidget(self.target_seq)
        self.target_image.setEnabled(self.has_image)
        self.target_seq.setEnabled(self.has_sequence)
        if self.has_sequence and not self.has_image:
            self.target_seq.setChecked(True)
        else:
            self.target_image.setChecked(True)
        layout.addWidget(target_group)

        # Progress + log
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        layout.addWidget(self.progress_bar)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(500)
        self.log_view.setFixedHeight(120)
        layout.addWidget(self.log_view)

        # Buttons
        btn_row = QHBoxLayout()
        about_btn = QPushButton("About")
        about_btn.setToolTip("About StarNet and this script.")
        about_btn.clicked.connect(self._show_about)
        advanced_btn = QPushButton("Set paths")
        advanced_btn.setToolTip(
            "Optional executable and weights paths for nonstandard installs.")
        advanced_btn.clicked.connect(self._show_advanced_paths)
        self.run_btn = QPushButton("Run StarNet")
        self.run_btn.clicked.connect(self._on_run)
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.close)
        btn_row.addWidget(about_btn)
        btn_row.addWidget(advanced_btn)
        btn_row.addStretch()
        btn_row.addWidget(self.run_btn)
        btn_row.addWidget(self.close_btn)
        layout.addLayout(btn_row)

    # -- settings <-> widgets --------------------------------------------
    def _load_into_ui(self):
        self.exe_path = self.opts.exe
        self.weights_path = self.opts.weights
        self.linear_check.setChecked(self.opts.linear)
        preset_index = self.stride_combo.findData(self.opts.stride_preset)
        if preset_index < 0:
            preset_index = self.stride_combo.findData(STRIDE_PRESET_STANDARD)
        self.stride_combo.setCurrentIndex(preset_index)
        self.stride_check.setChecked(self.opts.custom_stride)
        self.stride_spin.setValue(self.opts.stride)
        self.upsample_check.setChecked(self.opts.upsample)
        self.protect_highlights_check.setChecked(self.opts.protect_highlights)
        for kind, cb in self.mask_checks.items():
            cb.setChecked(kind in self.opts.masks)
        self._sync_enabled()

    def _collect_options(self):
        self.opts.exe = self.exe_path.strip()
        self.opts.weights = self.weights_path.strip()
        self.opts.linear = self.linear_check.isChecked()
        self.opts.stride_preset = self.stride_combo.currentData()
        self.opts.custom_stride = self.stride_check.isChecked()
        self.opts.stride = self.stride_spin.value()
        self.opts.upsample = self.upsample_check.isChecked()
        self.opts.protect_highlights = self.protect_highlights_check.isChecked()
        self.opts.masks = [k for k, cb in self.mask_checks.items() if cb.isChecked()]
        return self.opts

    def _sync_enabled(self):
        self.stride_combo.setEnabled(not self.stride_check.isChecked())
        self.stride_spin.setEnabled(self.stride_check.isChecked())

    def _sync_highlight_control(self, supports_disable):
        if not hasattr(self, "protect_highlights_check"):
            return
        if supports_disable:
            self.protect_highlights_check.setEnabled(True)
            self.protect_highlights_check.setToolTip(
                "Preserve clipped or near-clipped highlights when assigning the "
                "starless result.")
        else:
            self.protect_highlights_check.setChecked(True)
            self.protect_highlights_check.setEnabled(False)
            self.protect_highlights_check.setToolTip(
                "This StarNet2 executable does not support disabling highlight "
                "protection. StarNet2 "
                f"{FEATURE_STARNET2_HIGHLIGHT_PROTECTION_MIN_VERSION} or newer "
                "is required.")

    # -- about ------------------------------------------------------------
    def _show_about(self):
        dlg = QDialog(self)
        dlg.setWindowIcon(self._icon)
        dlg.setWindowTitle("About StarNet")
        lay = QVBoxLayout(dlg)
        label = QLabel(about_html())
        label.setTextFormat(Qt.TextFormat.RichText)
        label.setWordWrap(True)
        label.setOpenExternalLinks(True)
        lay.addWidget(label)
        row = QHBoxLayout()
        row.addStretch()
        close = QPushButton("Close")
        close.clicked.connect(dlg.accept)
        row.addWidget(close)
        lay.addLayout(row)
        dlg.exec()

    # -- executable handling ---------------------------------------------
    def _browse_exe(self, target=None):
        path, _ = QFileDialog.getOpenFileName(self, "Select the StarNet executable")
        if path:
            if target is not None:
                target.setText(path)
            else:
                self.exe_path = path
                self._check_executable()

    def _browse_weights(self, target=None):
        path, _ = QFileDialog.getOpenFileName(self, "Select the StarNet weights file")
        if path:
            if target is not None:
                target.setText(path)
            else:
                self.weights_path = path

    def _show_advanced_paths(self):
        dlg = QDialog(self)
        dlg.setWindowIcon(self._icon)
        dlg.setWindowTitle("Set paths")
        dlg.resize(900, 220)
        lay = QVBoxLayout(dlg)
        form = QGridLayout()
        form.setColumnStretch(1, 1)

        exe_edit = QLineEdit()
        exe_edit.setMinimumWidth(560)
        exe_edit.setText(self.exe_path or getattr(self, "_resolved_exe_path", ""))
        exe_edit.setPlaceholderText(
            "Normally leave blank; use only for nonstandard installs")
        exe_edit.setToolTip(
            "Optional path to a nonstandard StarNet 2.5 executable. Normally "
            "leave blank so the script uses PATH or the standard CLI installer "
            "location.")
        exe_browse = QPushButton("Browse...")
        exe_browse.clicked.connect(lambda: self._browse_exe(exe_edit))
        form.addWidget(QLabel("Executable:"), 0, 0)
        form.addWidget(exe_edit, 0, 1)
        form.addWidget(exe_browse, 0, 2)

        weights_edit = QLineEdit()
        weights_edit.setMinimumWidth(560)
        weights_edit.setText(self.weights_path)
        weights_edit.setPlaceholderText(
            "Optional - leave blank to use the bundled weights")
        weights_edit.setToolTip(
            "Optional path to an alternative model weights file. Leave blank to "
            "use the weights bundled with StarNet.")
        weights_browse = QPushButton("Browse...")
        weights_browse.clicked.connect(lambda: self._browse_weights(weights_edit))
        weights_clear = QPushButton("Clear")
        weights_clear.clicked.connect(lambda: weights_edit.clear())
        form.addWidget(QLabel("Weights:"), 1, 0)
        form.addWidget(weights_edit, 1, 1)
        wbox = QHBoxLayout()
        wbox.addWidget(weights_browse)
        wbox.addWidget(weights_clear)
        wcell = QWidget()
        wcell.setLayout(wbox)
        form.addWidget(wcell, 1, 2)
        lay.addLayout(form)

        row = QHBoxLayout()
        row.addStretch()
        ok_btn = QPushButton("OK")
        cancel_btn = QPushButton("Cancel")
        ok_btn.clicked.connect(dlg.accept)
        cancel_btn.clicked.connect(dlg.reject)
        row.addWidget(ok_btn)
        row.addWidget(cancel_btn)
        lay.addLayout(row)

        if dlg.exec():
            self.exe_path = exe_edit.text().strip()
            self.weights_path = weights_edit.text().strip()
            self._check_executable()

    def _check_executable(self):
        exe = self.exe_path.strip()
        status, message, resolved_exe = detect_starnet_version_info(exe)
        self._resolved_exe_path = resolved_exe or ""
        colors = {"ok": "green", "upgrade": "darkorange",
                  "missing": "red", "unknown": "darkorange"}
        self.status_label.setText(message)
        self.status_label.setStyleSheet(f"color: {colors.get(status, 'black')};")
        self.update_label.setText("")
        self._exe_ok = (status == "ok")
        self._sync_highlight_control(
            self._exe_ok and starnet_highlight_support_for_configured(exe))
        if self._exe_ok:
            self._start_latest_cli_check(parse_version_tuple(message))
        if hasattr(self, "run_btn"):
            self.run_btn.setEnabled(self._exe_ok)
        return self._exe_ok

    def _start_latest_cli_check(self, installed_version):
        if installed_version is None:
            return
        self.update_worker = LatestCliVersionWorker(installed_version)
        self.update_worker.notice.connect(self._show_latest_cli_notice)
        self.update_worker.start()

    def _show_latest_cli_notice(self, notice):
        self.update_label.setText(notice)

    # -- run --------------------------------------------------------------
    def _on_run(self):
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.run_btn.setEnabled(False)
            return

        opts = self._collect_options()
        if not self._check_executable():
            return
        sequence_mode = self.target_seq.isChecked()
        if sequence_mode and not self.has_sequence:
            QMessageBox.warning(self, "No sequence", "No sequence is loaded.")
            return
        if not sequence_mode and not self.has_image:
            QMessageBox.warning(self, "No image", "No image is loaded.")
            return

        save_options(self.siril, opts)
        self.progress_bar.setValue(0)
        self.run_btn.setText("Cancel")

        self.worker = StarNetWorker(self.siril, opts, sequence_mode)
        self.worker.progress.connect(self._on_progress)
        self.worker.message.connect(self._on_message)
        self.worker.finished_ok.connect(self._on_finished)
        self.worker.start()

    def _on_progress(self, frac):
        self.progress_bar.setValue(int(max(0.0, min(1.0, frac)) * 100))
        try:
            self.siril.update_progress("StarNet", max(0.0, min(1.0, frac)))
        except Exception:
            pass

    def _on_message(self, text):
        self.log_view.appendPlainText(text)

    def _on_finished(self, ok, message):
        self.run_btn.setText("Run StarNet")
        self.run_btn.setEnabled(True)
        self._on_message(message)
        try:
            self.siril.reset_progress()
        except Exception:
            pass
        if ok:
            self.progress_bar.setValue(100)
        else:
            QMessageBox.warning(self, "StarNet", message)

    def closeEvent(self, event):
        try:
            self._collect_options()
            save_options(self.siril, self.opts)
        except Exception:
            pass
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait(5000)
        try:
            self.siril.disconnect()
        except Exception:
            pass
        event.accept()


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------

def build_arg_parser():
    def stride_arg(value):
        try:
            stride = int(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("stride must be an integer") from exc
        if stride < 2 or stride > 512 or stride % 2:
            raise argparse.ArgumentTypeError(
                "stride must be an even integer between 2 and 512")
        return stride

    p = argparse.ArgumentParser(
        prog="StarNet.py",
        description="Remove stars from the loaded image or sequence using StarNet 2.5.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--exe", help="Optional last-resort path to a nonstandard "
                   "StarNet 2.5 executable. Normally leave unset so PATH or the "
                   "standard CLI installer location is used.")
    p.add_argument("--weights", help="Optional path to a model weights file.")
    stretch = p.add_mutually_exclusive_group()
    stretch.add_argument("--linear", dest="linear", action="store_true",
                         help="Treat the image as linear and autostretch before StarNet (default).")
    stretch.add_argument("--no-linear", dest="linear", action="store_false",
                         help="Treat the image as already stretched.")
    p.set_defaults(linear=None)
    p.add_argument("--stride", type=stride_arg, help="Custom tile stride (even, 2..512).")
    upsample = p.add_mutually_exclusive_group()
    upsample.add_argument("--upsample", dest="upsample", action="store_true",
                          help="Use intermediate 2x upsampling.")
    upsample.add_argument("--no-upsample", dest="upsample", action="store_false",
                          help="Disable intermediate 2x upsampling.")
    p.set_defaults(upsample=None)
    highlights = p.add_mutually_exclusive_group()
    highlights.add_argument("--protect-highlights", dest="protect_highlights",
                            action="store_true",
                            help="Enable StarNet2 highlight protection.")
    highlights.add_argument("--disable-highlights-protection", dest="protect_highlights",
                            action="store_false",
                            help="Disable StarNet2 highlight protection. Requires StarNet2 "
                            f"{FEATURE_STARNET2_HIGHLIGHT_PROTECTION_MIN_VERSION} or newer.")
    p.set_defaults(protect_highlights=None)
    p.add_argument("--masks", help="Comma-separated star outputs: "
                   + ", ".join(ALL_MASK_KINDS) + ".")
    p.add_argument("--sequence", action="store_true",
                   help="Process the loaded FITS sequence instead of the loaded image.")
    return p


def run_cli(siril, argv):
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    opts = load_options(siril)
    if args.exe:
        opts.exe = args.exe
    if args.weights is not None:
        opts.weights = args.weights
    if args.linear is not None:
        opts.linear = args.linear
    if args.stride is not None:
        opts.custom_stride = True
        opts.stride = args.stride
    if args.upsample is not None:
        opts.upsample = args.upsample
    if args.protect_highlights is not None:
        opts.protect_highlights = args.protect_highlights
    if args.masks is not None:
        kinds = [m.strip() for m in args.masks.split(",") if m.strip()]
        bad = [m for m in kinds if m not in ALL_MASK_KINDS]
        if bad:
            siril.log(f"StarNet: unknown mask kind(s): {', '.join(bad)}", LogColor.RED)
            return 1
        opts.masks = kinds

    status, message = detect_starnet_version(opts.exe)
    siril.log(f"StarNet: {message}", LogColor.GREEN if status == "ok" else LogColor.SALMON)
    if status != "ok":
        return 1
    notice = latest_cli_update_notice("starnet2", parse_version_tuple(message))
    if notice:
        siril.log(f"StarNet: {notice}", LogColor.SALMON)

    def progress_cb(frac):
        try:
            siril.update_progress("StarNet", max(0.0, min(1.0, frac)))
        except Exception:
            pass

    def log_cb(text):
        siril.log(f"StarNet: {text}", LogColor.DEFAULT)

    try:
        if args.sequence:
            if not siril.is_sequence_loaded():
                siril.log("StarNet: no sequence is loaded.", LogColor.RED)
                return 1
            ok = process_sequence(siril, opts, progress_cb=progress_cb,
                                  log_cb=log_cb)
        else:
            if not siril.is_image_loaded():
                siril.log("StarNet: no image is loaded.", LogColor.RED)
                return 1
            ok = process_single_image(siril, opts, progress_cb=progress_cb,
                                      log_cb=log_cb)
    except StarNetError as exc:
        siril.log(f"StarNet: {exc}", LogColor.RED)
        return 1
    finally:
        try:
            siril.reset_progress()
        except Exception:
            pass
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    cli_argv = [a for a in sys.argv[1:] if a not in ("--", )]

    siril = s.SirilInterface()
    try:
        siril.connect()
    except s.SirilConnectionError:
        print("StarNet: could not connect to Siril.", file=sys.stderr)
        sys.exit(1)

    try:
        siril.cmd("requires", "1.4.0")
    except Exception as exc:
        siril.log(f"StarNet: {exc}", LogColor.RED)
        siril.disconnect()
        sys.exit(1)

    # CLI mode: any arguments, or invoked from the command line rather than the
    # Scripts menu.
    if cli_argv:
        try:
            rc = run_cli(siril, cli_argv)
        finally:
            siril.disconnect()
        sys.exit(rc)

    # GUI mode.
    app = QApplication.instance() or QApplication(sys.argv)
    gui = StarNetGUI(siril)
    gui.show()
    app.exec()


if __name__ == "__main__":
    main()

