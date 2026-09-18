"""
Svenesis Annotate Image
Script Version: 1.1.0
=====================================

Author: Svenesis-Siril-Scripts project.
Contact and support: See repository README and Siril forum / scripts repository.

This script reads the current plate-solved image from Siril, queries online
catalogs for objects in the field of view, and renders configurable annotations
(markers, labels, coordinate grid, info box, compass, legend) onto an exportable
PNG/TIFF/JPEG image. Inspired by PixInsight's AnnotateImage script.

All catalog data comes from live online queries -- no hardcoded object data.

Data Sources:
- VizieR VII/118 (NGC 2000.0): NGC, IC, and Messier objects
- VizieR VII/20 (Sharpless 1959): HII regions
- VizieR VII/220A (Barnard 1927): Dark nebulae
- VizieR V/50 (Yale BSC): Named bright stars
- SIMBAD: Supplementary objects (UGC, Abell, Arp, Hickson, Markarian,
  vdB, PGC, MCG, etc.) plus common name resolution via TAP

Object Types (12 categories, selectable via checkboxes):
- Galaxies (gold), Nebulae (red), Planetary Nebulae (green)
- Open Clusters (blue), Globular Clusters (orange), Named Stars (white)
- Reflection Nebulae (light red), Supernova Remnants (magenta)
- Dark Nebulae (grey), HII Regions (red-pink)
- Asterisms (pale blue), Quasars (violet)

Features:
- Object-type selection UI (not catalog selection) -- user controls what
  types of objects appear, catalogs are queried automatically
- Parallel catalog queries via ThreadPoolExecutor for fast annotation
- Thread-safe siril coordinate access via locking
- Common names from SIMBAD (e.g. "Andromeda Galaxy", "Eyes")
  with automatic filtering of catalog-like names (FAUST, IRAS, etc.)
- Object size rendered as scaled ellipses from catalog angular size
- Smart label collision avoidance (32 candidates, spatial grid scoring)
- Configurable magnitude limit (dark nebulae bypass this)
- RA/DEC coordinate grid with auto-scaled spacing
- Info box (center coords, FOV, pixel scale, rotation, object count)
- N/E compass rose with WCS-derived orientation
- Color legend (auto-generated, only shows present types)
- Leader lines connecting labels to objects
- Color-by-type toggle (gold/red/green/blue/orange/etc.)
- Deduplication across catalogs (by name + spatial proximity)
- Large mosaic support (display downscaling, DPI capping, memory mgmt)
- Configurable font size, marker size, DPI
- Output as PNG, TIFF, or JPEG
- Dark-themed PyQt6 GUI with two-column checkbox layout
- Persistent settings via QSettings
- Progress feedback during annotation

Run from Siril via Processing -> Scripts. Place AnnotateImage.py inside a folder
named Utility in one of Siril's Script Storage Directories (Preferences -> Scripts).

(c) 2025-2026
SPDX-License-Identifier: GPL-3.0-or-later

# SPDX-License-Identifier: GPL-3.0-or-later
# Script Name: Svenesis Annotate Image
# Script Version: 1.1.0
# Siril Version: 1.4.0
# Python Module Version: 1.0.0
# Script Category: processing
# Script Description: Annotates a plate-solved image with catalog objects
#   (NGC, IC, Messier, Sharpless, Barnard, bright stars, SIMBAD supplements),
#   coordinate grid, compass, info box, and legend. Exports as PNG/TIFF/JPEG.
#   All data from live VizieR/SIMBAD queries. Requires a plate-solved image.
# Script Author: Sven Ramuschkat

CHANGELOG:
1.1.0 - Performance and correctness update
      - Parallel catalog queries (VizieR + SIMBAD run concurrently)
      - Thread-safe siril coordinate access via locking
      - Parallel SIMBAD tiling for wide-field images
      - Removed "Extended Catalogs" checkbox -- SIMBAD always queried
      - Common name quality filter (discards catalog-like names)
      - IC naming fix (handles all NGC 2000.0 IC prefix formats)
      - Spatial grid deduplication with precise distance check
      - Spatial grid label collision avoidance (O(1) per candidate)
      - Pre-compiled regex patterns for hot paths
      - First-char dispatch for SIMBAD prefix filtering
      - Pre-resolved column names in SIMBAD result loop
      - Display data downscaling for large mosaics
      - Memory cleanup after rendering (gc.collect)
      - Fixed type defaults (Ast/QSO now correctly default to off)
      - Suppressed vstack MergeConflictWarnings
      - Two-column checkbox layout for Display and Extras groups
      - Consistent checkbox spacing and alignment across all groups
      - Updated help dialog to match current features

1.0.0 - Initial release
      - Plate-solved image annotation with catalog objects
      - Dynamic VizieR catalog queries (NGC/IC, Sharpless, Barnard, bright stars)
      - Color-coded markers and labels by object type
      - Coordinate grid overlay with RA/DEC labels
      - Info box with center, FOV, scale, rotation
      - Compass rose overlay (N/E arrows)
      - Label collision avoidance
      - Magnitude limit filtering
      - PNG/TIFF/JPEG export with configurable DPI
      - Dark-themed PyQt6 GUI
      - Persistent settings
"""
from __future__ import annotations

import sys
import os
import re
import gc
import traceback
import math
import datetime
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

import sirilpy as s
from sirilpy import NoImageError

try:
    from sirilpy.exceptions import SirilError, SirilConnectionError
except ImportError:
    class SirilError(Exception):
        pass
    class SirilConnectionError(Exception):
        pass

s.ensure_installed("numpy", "PyQt6", "matplotlib", "astropy")

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
    QWidget, QLabel, QPushButton, QMessageBox, QGroupBox,
    QCheckBox, QSlider, QSpinBox, QDoubleSpinBox,
    QSizePolicy, QDialog, QTextEdit, QTabWidget, QScrollArea,
    QProgressBar, QComboBox, QRadioButton, QButtonGroup,
    QLineEdit, QFileDialog, QGridLayout,
)
from PyQt6.QtCore import Qt, QSettings, QUrl
from PyQt6.QtGui import (
    QPainter, QColor, QPen, QFont, QLinearGradient,
    QShortcut, QKeySequence, QDesktopServices, QImageReader,
)

# Remove Qt's default image allocation limit (256 MB) so large mosaics can be
# saved/loaded without "Rejecting image as it exceeds the allocation limit".
QImageReader.setAllocationLimit(0)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle, FancyArrowPatch
import matplotlib.patheffects as pe
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

from astropy.wcs import WCS
from astropy.io import fits as astropy_fits
from astropy.coordinates import SkyCoord, Angle
import astropy.units as u
from astroquery.vizier import Vizier
from astroquery.simbad import Simbad

VERSION = "1.1.0"

# Pre-compiled regex patterns for hot paths
_RE_WHITESPACE = re.compile(r'\s+')
_RE_IC_PREFIX = re.compile(r'^I\s*(\d+)')

# Settings keys
SETTINGS_ORG = "Svenesis"
SETTINGS_APP = "AnnotateImage"

# Layout constants
LEFT_PANEL_WIDTH = 360


def _safe_radec2pix(siril, ra: float, dec: float,
                    lock: threading.Lock | None = None):
    """Thread-safe wrapper for siril.radec2pix()."""
    if lock:
        with lock:
            return siril.radec2pix(ra, dec)
    return siril.radec2pix(ra, dec)


def _safe_pix2radec(siril, x: float, y: float,
                    lock: threading.Lock | None = None):
    """Thread-safe wrapper for siril.pix2radec()."""
    if lock:
        with lock:
            return siril.pix2radec(x, y)
    return siril.pix2radec(x, y)


# ------------------------------------------------------------------------------
# STYLING
# ------------------------------------------------------------------------------

def _nofocus(w: QWidget | None) -> None:
    if w is not None:
        w.setFocusPolicy(Qt.FocusPolicy.NoFocus)


DARK_STYLESHEET = """
QWidget{background-color:#2b2b2b;color:#e0e0e0;font-size:10pt}

QToolTip{background-color:#333333;color:#ffffff;border:1px solid #88aaff}

QGroupBox{border:1px solid #444444;margin-top:5px;font-weight:bold;border-radius:4px;padding-top:12px}
QGroupBox::title{subcontrol-origin:margin;left:8px;padding:0 3px;color:#88aaff}

QLabel{color:#cccccc}

QCheckBox{color:#cccccc;spacing:5px}
QCheckBox::indicator{width:14px;height:14px;border:1px solid #666666;background:#3c3c3c;border-radius:3px}
QCheckBox::indicator:checked{background:#285299;border:1px solid #88aaff;image:none}

QSlider::groove:horizontal{background:#3c3c3c;height:6px;border-radius:3px}
QSlider::handle:horizontal{background:#88aaff;width:14px;margin:-4px 0;border-radius:7px}
QSlider::sub-page:horizontal{background:#285299;border-radius:3px}

QSpinBox,QDoubleSpinBox{background-color:#3c3c3c;color:#e0e0e0;border:1px solid #666666;border-radius:4px;padding:4px;min-width:60px}
QSpinBox:focus,QDoubleSpinBox:focus{border-color:#88aaff}

QLineEdit{background-color:#3c3c3c;color:#e0e0e0;border:1px solid #666666;border-radius:4px;padding:4px}
QLineEdit:focus{border-color:#88aaff}

QComboBox{background-color:#3c3c3c;color:#e0e0e0;border:1px solid #666666;border-radius:4px;padding:4px;min-width:60px}
QComboBox:focus{border-color:#88aaff}
QComboBox::drop-down{border:none}
QComboBox QAbstractItemView{background-color:#3c3c3c;color:#e0e0e0;selection-background-color:#285299}

QRadioButton{color:#cccccc;spacing:5px}
QRadioButton::indicator{width:14px;height:14px;border:1px solid #666666;background:#3c3c3c;border-radius:7px}
QRadioButton::indicator:checked{background:#285299;border:2px solid #88aaff}

QPushButton{background-color:#444444;color:#dddddd;border:1px solid #666666;border-radius:4px;padding:6px;font-weight:bold}
QPushButton:hover{background-color:#555555;border-color:#777777}
QPushButton#CoffeeButton{background-color:#FFDD00;color:#000000;border:1px solid #ccb100;font-weight:bold}
QPushButton#CloseButton{background-color:#553333;color:#ffaaaa;border:1px solid #884444}
QPushButton#CloseButton:hover{background-color:#664444}
QPushButton#AnnotateButton{background-color:#335533;color:#aaffaa;border:1px solid #448844}
QPushButton#AnnotateButton:hover{background-color:#446644}

QProgressBar{background-color:#3c3c3c;border:1px solid #555555;border-radius:3px;text-align:center;color:#e0e0e0;font-size:9pt}
QProgressBar::chunk{background-color:#285299;border-radius:2px}

QTabWidget::pane{border:1px solid #444444;border-radius:4px}
QTabBar::tab{background-color:#333333;color:#bbbbbb;padding:6px 14px;margin-right:2px;border-top-left-radius:4px;border-top-right-radius:4px}
QTabBar::tab:selected{background-color:#2b2b2b;color:#88aaff;border-bottom:2px solid #88aaff}
QTabBar::tab:hover{background-color:#3c3c3c}

QScrollArea{border:none}
QScrollBar:vertical{background:#2b2b2b;width:10px;border:none}
QScrollBar::handle:vertical{background:#555555;border-radius:4px;min-height:20px}
QScrollBar::handle:vertical:hover{background:#666666}
QScrollBar::add-line:vertical,QScrollBar::sub-line:vertical{height:0}
"""


# ------------------------------------------------------------------------------
# COLOR SCHEME PER OBJECT TYPE
# ------------------------------------------------------------------------------

DEFAULT_COLORS = {
    "Gal":   "#FFD700",  # Gold — Galaxies
    "Neb":   "#FF4444",  # Red — Emission nebulae
    "RN":    "#FF8888",  # Light red — Reflection nebulae
    "PN":    "#44FF44",  # Green — Planetary nebulae
    "OC":    "#44AAFF",  # Light blue — Open clusters
    "GC":    "#FF8800",  # Orange — Globular clusters
    "SNR":   "#FF44FF",  # Magenta — Supernova remnants
    "DN":    "#888888",  # Grey — Dark nebulae
    "Star":  "#FFFFFF",  # White — Named stars
    "HII":   "#FF6666",  # Red-pink — HII regions
    "Ast":   "#AADDFF",  # Pale blue — Asterisms
    "QSO":   "#DDAAFF",  # Violet — Quasars
    "Other": "#CCCCCC",  # Light grey — Other
}

OBJECT_TYPE_LABELS = {
    "Gal": "Galaxy", "Neb": "Emission Nebula", "RN": "Reflection Nebula",
    "PN": "Planetary Nebula", "OC": "Open Cluster", "GC": "Globular Cluster",
    "SNR": "Supernova Remnant", "DN": "Dark Nebula", "Star": "Named Star",
    "HII": "HII Region", "Ast": "Asterism", "QSO": "Quasar", "Other": "Other",
}


# ------------------------------------------------------------------------------
# DYNAMIC CATALOG QUERIES (VizieR / SIMBAD)
# ------------------------------------------------------------------------------
# All catalog data is fetched dynamically from online databases.
# No hardcoded object positions — queries VizieR catalogs at runtime.
# Supported catalogs:
#   NGC/IC/Messier: VII/118 (NGC 2000.0)
#   Sharpless HII:  VII/20
#   Barnard dark:   VII/220A
#   Lynds dark:     VII/7A
#   Bright stars:   V/50 (Yale BSC)
# SIMBAD is used as a supplementary source for additional objects.


# NGC 2000.0 object type code → our internal type code
NGC2000_TYPE_MAP: dict[str, str] = {
    "Gx": "Gal",    # Galaxy
    "Gb": "GC",     # Globular cluster
    "OC": "OC",     # Open cluster
    "Pl": "PN",     # Planetary nebula
    "Nb": "Neb",    # Bright emission or reflection nebula
    "C+N": "OC",    # Cluster associated with nebulosity
    "Kt": "Neb",    # Knot in external galaxy
    "Ast": "Ast",    # Asterism / star group
    "*": "Star",    # Single star
    "D*": "Star",   # Double star
    "***": "Star",  # Triple star
    "?": "Other",   # Uncertain type
    "PD": "Other",  # Plate defect
    "-": "Other",   # Nonexistent
    "": "Other",    # Unknown type
}

def resolve_common_names(objects: list[dict], log_func=None) -> None:
    """
    Resolve common names for all objects via SIMBAD TAP query.
    Queries the SIMBAD 'ident' table in batch to find identifiers
    starting with 'NAME ' (= common/popular names).
    Updates objects in-place, setting the 'common_name' field.
    """
    if not objects:
        return

    # Collect names that need resolving (skip objects that already have a common name)
    to_resolve = [obj for obj in objects if not obj.get("common_name")]
    if not to_resolve:
        return

    try:
        # Build a single TAP query to resolve all names at once.
        # The ident table maps object IDs (oidref) to all known identifiers.
        # We join it with itself: one side matches our catalog names,
        # the other side finds "NAME ..." identifiers (common names).
        #
        # Process in batches to avoid overly long SQL queries.
        BATCH_SIZE = 80
        name_map: dict[str, str] = {}

        # Normalize whitespace: "NGC  5473" → "NGC 5473"
        def _normalize(n: str) -> str:
            return _RE_WHITESPACE.sub(' ', n).strip()

        # Build mapping: normalized name → original object indices
        all_names_raw = [obj["name"] for obj in to_resolve]
        all_names = [_normalize(n) for n in all_names_raw]
        # Also normalize object names in-place so they display cleanly
        for i, obj in enumerate(to_resolve):
            obj["name"] = all_names[i]

        n_batches = (len(all_names) + BATCH_SIZE - 1) // BATCH_SIZE

        if log_func:
            log_func(f"SIMBAD names: resolving {len(all_names)} objects "
                      f"in {n_batches} batch(es) via TAP query")
            # Show first few names being queried
            sample = all_names[:5]
            log_func(f"  sample names: {sample}")

        for batch_start in range(0, len(all_names), BATCH_SIZE):
            batch = all_names[batch_start:batch_start + BATCH_SIZE]
            batch_num = batch_start // BATCH_SIZE + 1
            # Escape single quotes in names
            escaped = [n.replace("'", "''") for n in batch]
            in_clause = ", ".join(f"'{n}'" for n in escaped)

            query = (
                "SELECT id1.id AS catalog_id, id2.id AS common_name "
                "FROM ident AS id1 "
                "JOIN ident AS id2 ON id1.oidref = id2.oidref "
                f"WHERE id1.id IN ({in_clause}) "
                "AND id2.id LIKE 'NAME %'"
            )

            try:
                result = Simbad.query_tap(query)
                batch_found = 0
                if result is not None and len(result) > 0:
                    if log_func and len(result) > 0:
                        log_func(f"  TAP returned {len(result)} rows, first: {dict(result[0])}")
                    for row in result:
                        cat_id = _normalize(str(row["catalog_id"]).strip())
                        common = str(row["common_name"]).strip()
                        # Remove the "NAME " prefix
                        if common.upper().startswith("NAME "):
                            common = common[5:]
                        # Discard catalog-like names entirely — they are not
                        # useful common names (e.g. "FAUST V051", "IRAS 12345")
                        _JUNK_PREFIXES = (
                            "FAUST", "IRAS", "2MASS", "SDSS", "WISE",
                            "GALEX", "XMM", "ROSAT", "NVSS", "FIRST",
                            "UGCA", "MCG", "PGC", "UGC", "KUG",
                            "MRK", "ARK", "CGCG", "ZWG", "LEDA",
                            "1RXS", "2E", "RX J", "LBQS", "QSO",
                        )
                        first_word = common.split()[0].upper() if common else ""
                        if any(first_word.startswith(p.upper()) for p in _JUNK_PREFIXES):
                            continue  # skip this name entirely
                        # Prefer the longest (most descriptive) common name
                        if cat_id not in name_map or len(common) > len(name_map[cat_id]):
                            if cat_id not in name_map:
                                batch_found += 1
                            name_map[cat_id] = common
                if log_func:
                    log_func(f"SIMBAD names: batch {batch_num}/{n_batches}: "
                              f"{batch_found} common names found for {len(batch)} objects")
            except Exception as batch_err:
                if log_func:
                    log_func(f"SIMBAD names: batch {batch_num}/{n_batches} failed: {batch_err}")
                continue

        # Apply resolved names to objects
        resolved = 0
        for obj in to_resolve:
            common = name_map.get(obj["name"], "")
            if common:
                obj["common_name"] = common
                resolved += 1

        if log_func:
            log_func(f"SIMBAD names: resolved {resolved}/{len(to_resolve)} common names total")
            # Show some examples
            examples = [(obj["name"], obj["common_name"])
                        for obj in to_resolve if obj.get("common_name")][:5]
            for cat_name, common in examples:
                log_func(f"  {cat_name} \u2192 {common}")

    except Exception as e:
        if log_func:
            log_func(f"SIMBAD name resolution failed: {e}")
            import traceback
            log_func(f"  traceback: {traceback.format_exc().splitlines()[-1]}")


def query_vizier_ngc_ic(
    center_ra: float, center_dec: float, radius_deg: float,
    mag_limit: float, siril, log_func, img_width: int, img_height: int,
    siril_lock: threading.Lock | None = None,
) -> list[dict]:
    """
    Query VizieR VII/118 (NGC 2000.0) for NGC/IC/Messier objects in the FOV.
    Returns list of object dicts with pixel coordinates.
    """
    try:
        vizier = Vizier(
            columns=["**", "+_RAJ2000", "+_DEJ2000"],
            row_limit=5000,
        )

        center = SkyCoord(center_ra, center_dec, unit="deg")
        if log_func:
            log_func(f"VizieR NGC/IC: querying VII/118 radius={radius_deg:.2f}\u00b0 "
                      f"around RA={center_ra:.4f} DEC={center_dec:.4f}, mag<={mag_limit:.1f}")
        results = vizier.query_region(
            center, radius=radius_deg * u.deg, catalog="VII/118/ngc2000",
        )
        if not results or len(results) == 0:
            if log_func:
                log_func("VizieR NGC/IC: no results returned from server")
            return []

        table = results[0]
        if log_func:
            log_func(f"VizieR NGC/IC: {len(table)} objects returned from server")
            log_func(f"VizieR NGC/IC: columns = {table.colnames}")

        objects = []
        skipped_mag = 0
        skipped_pix = 0
        skipped_margin = 0
        skipped_err = 0
        first_err_logged = False
        margin = 30
        for row in table:
            try:
                # Name: VizieR returns just the number (e.g. "5457"), need prefix
                raw_name = str(row["Name"]).strip()
                # Detect NGC vs IC from the catalog (Name column format)
                # NGC 2000.0 uses "I" prefix for IC objects: "I 3280", "I0123", "I3280"
                ic_match = _RE_IC_PREFIX.match(raw_name)
                if ic_match:
                    name = f"IC {ic_match.group(1)}"
                else:
                    name = f"NGC {raw_name}"

                # Coordinates from VizieR computed columns (J2000, degrees)
                ra_deg = float(row["_RAJ2000"])
                dec_deg = float(row["_DEJ2000"])

                # Magnitude — try multiple possible column names
                mag = 0.0
                for mag_col in ("Mag", "mag", "Vmag"):
                    if mag_col in table.colnames and not np.ma.is_masked(row[mag_col]):
                        mag = float(row[mag_col])
                        break
                mag_known = mag > 0.0
                if mag_known and mag > mag_limit:
                    skipped_mag += 1
                    continue

                # Size — try multiple possible column names
                size = 0.0
                for size_col in ("size", "Diam", "diam", "MajAxis", "Size"):
                    if size_col in table.colnames and not np.ma.is_masked(row[size_col]):
                        size = float(row[size_col])
                        break

                # Type mapping
                raw_type = ""
                for type_col in ("Type", "type", "OType"):
                    if type_col in table.colnames and not np.ma.is_masked(row[type_col]):
                        raw_type = str(row[type_col]).strip()
                        break
                obj_type = NGC2000_TYPE_MAP.get(raw_type, "Other")

                # Messier designation — try multiple column names
                messier = ""
                for m_col in ("Messier", "messier", "Mno"):
                    if m_col in table.colnames and not np.ma.is_masked(row[m_col]):
                        m = str(row[m_col]).strip()
                        if m:
                            messier = f"M{m}"
                            break

                # Display name: prefer Messier, then NGC/IC
                display_name = messier if messier else name

                # Convert to pixel coordinates (thread-safe)
                result_pix = _safe_radec2pix(siril, ra_deg, dec_deg, siril_lock)
                if result_pix is None:
                    skipped_pix += 1
                    continue
                x, y = float(result_pix[0]), float(result_pix[1])
                if not (margin < x < img_width - margin and margin < y < img_height - margin):
                    skipped_margin += 1
                    continue

                objects.append({
                    "name": display_name,
                    "ra": ra_deg,
                    "dec": dec_deg,
                    "type": obj_type,
                    "mag": mag,
                    "size_arcmin": size,
                    "common_name": "",  # resolved later via SIMBAD TAP
                    "pixel_x": x,
                    "pixel_y": y,
                })
            except Exception as row_err:
                skipped_err += 1
                if not first_err_logged and log_func:
                    first_err_logged = True
                    log_func(f"VizieR NGC/IC: first row error: {row_err}")
                    try:
                        log_func(f"  row data: {dict(zip(table.colnames, [row[c] for c in table.colnames]))}")
                    except Exception:
                        pass
                continue

        if log_func:
            log_func(f"VizieR NGC/IC: {len(objects)} in FOV "
                      f"(skipped: {skipped_mag} too faint, {skipped_pix} pix2radec fail, "
                      f"{skipped_margin} outside margin, {skipped_err} errors)")
        return objects

    except Exception as e:
        if log_func:
            log_func(f"VizieR NGC/IC query failed: {e}")
            import traceback
            log_func(f"  traceback: {traceback.format_exc().splitlines()[-1]}")
        return []


def query_vizier_sharpless(
    center_ra: float, center_dec: float, radius_deg: float,
    siril, log_func, img_width: int, img_height: int,
    siril_lock: threading.Lock | None = None,
) -> list[dict]:
    """
    Query VizieR VII/20 (Sharpless 1959) for HII regions in the FOV.
    """
    try:
        vizier = Vizier(
            columns=["**", "+_RAJ2000", "+_DEJ2000"],
            row_limit=5000,
        )
        center = SkyCoord(center_ra, center_dec, unit="deg")
        if log_func:
            log_func(f"VizieR Sharpless: querying VII/20 radius={radius_deg:.2f}\u00b0")
        results = vizier.query_region(
            center, radius=radius_deg * u.deg, catalog="VII/20",
        )
        if not results or len(results) == 0:
            if log_func:
                log_func("VizieR Sharpless: no results returned from server")
            return []

        table = results[0]
        if log_func:
            log_func(f"VizieR Sharpless: {len(table)} objects returned, columns={table.colnames}")

        objects = []
        skipped_pix = 0
        skipped_margin = 0
        skipped_err = 0
        first_err_logged = False
        margin = 30
        for row in table:
            try:
                # Find the Sharpless number column
                num = ""
                for sh_col in ("Sh2", "Sh 2", "SH2"):
                    if sh_col in table.colnames and not np.ma.is_masked(row[sh_col]):
                        num = str(row[sh_col]).strip()
                        break
                if not num:
                    num = str(row[table.colnames[0]]).strip()
                name = f"Sh2-{num}"

                ra_deg = float(row["_RAJ2000"])
                dec_deg = float(row["_DEJ2000"])

                size = 0.0
                for size_col in ("Diam", "diam", "Dia"):
                    if size_col in table.colnames and not np.ma.is_masked(row[size_col]):
                        size = float(row[size_col])
                        break

                result_pix = _safe_radec2pix(siril, ra_deg, dec_deg, siril_lock)
                if result_pix is None:
                    skipped_pix += 1
                    continue
                x, y = float(result_pix[0]), float(result_pix[1])
                if not (margin < x < img_width - margin and margin < y < img_height - margin):
                    skipped_margin += 1
                    continue

                objects.append({
                    "name": name,
                    "ra": ra_deg,
                    "dec": dec_deg,
                    "type": "HII",
                    "mag": 0.0,
                    "size_arcmin": size,
                    "common_name": "",  # resolved later via SIMBAD TAP
                    "pixel_x": x,
                    "pixel_y": y,
                })
            except Exception as row_err:
                skipped_err += 1
                if not first_err_logged and log_func:
                    first_err_logged = True
                    log_func(f"VizieR Sharpless: first row error: {row_err}")
                    try:
                        log_func(f"  row data: {dict(zip(table.colnames, [row[c] for c in table.colnames]))}")
                    except Exception:
                        pass
                continue

        if log_func:
            log_func(f"VizieR Sharpless: {len(objects)} in FOV "
                      f"(skipped: {skipped_pix} pix fail, {skipped_margin} outside, {skipped_err} errors)")
        return objects

    except Exception as e:
        if log_func:
            log_func(f"VizieR Sharpless query failed: {e}")
            import traceback
            log_func(f"  traceback: {traceback.format_exc().splitlines()[-1]}")
        return []


def query_vizier_barnard(
    center_ra: float, center_dec: float, radius_deg: float,
    siril, log_func, img_width: int, img_height: int,
    siril_lock: threading.Lock | None = None,
) -> list[dict]:
    """
    Query VizieR VII/220A (Barnard 1927) for dark nebulae in the FOV.
    """
    try:
        vizier = Vizier(
            columns=["**", "+_RAJ2000", "+_DEJ2000"],
            row_limit=5000,
        )
        center = SkyCoord(center_ra, center_dec, unit="deg")
        if log_func:
            log_func(f"VizieR Barnard: querying VII/220A radius={radius_deg:.2f}\u00b0")
        results = vizier.query_region(
            center, radius=radius_deg * u.deg, catalog="VII/220A",
        )

        objects = []
        margin = 30

        if results and len(results) > 0:
            table = results[0]
            if log_func:
                log_func(f"VizieR Barnard: {len(table)} objects returned, columns={table.colnames}")
            skipped_err = 0
            first_err_logged = False
            for row in table:
                try:
                    # Find the Barnard number column
                    num = ""
                    for b_col in ("Barn", "Bern", "No"):
                        if b_col in table.colnames and not np.ma.is_masked(row[b_col]):
                            num = str(row[b_col]).strip()
                            break
                    if not num:
                        num = str(row[table.colnames[0]]).strip()
                    name = f"B{num}"

                    ra_deg = float(row["_RAJ2000"])
                    dec_deg = float(row["_DEJ2000"])

                    size = 0.0
                    for size_col in ("Diam", "diam", "Dia"):
                        if size_col in table.colnames and not np.ma.is_masked(row[size_col]):
                            size = float(row[size_col])
                            break

                    result_pix = _safe_radec2pix(siril, ra_deg, dec_deg, siril_lock)
                    if result_pix is None:
                        continue
                    x, y = float(result_pix[0]), float(result_pix[1])
                    if not (margin < x < img_width - margin and margin < y < img_height - margin):
                        continue

                    objects.append({
                        "name": name,
                        "ra": ra_deg,
                        "dec": dec_deg,
                        "type": "DN",
                        "mag": 0.0,
                        "size_arcmin": size,
                        "common_name": "",  # resolved later via SIMBAD TAP
                        "pixel_x": x,
                        "pixel_y": y,
                    })
                except Exception as row_err:
                    skipped_err += 1
                    if not first_err_logged and log_func:
                        first_err_logged = True
                        log_func(f"VizieR Barnard: first row error: {row_err}")
                        try:
                            log_func(f"  row data: {dict(zip(table.colnames, [row[c] for c in table.colnames]))}")
                        except Exception:
                            pass
                    continue

        if log_func:
            log_func(f"VizieR Barnard: {len(objects)} in FOV")
        return objects

    except Exception as e:
        if log_func:
            log_func(f"VizieR Barnard query failed: {e}")
        return []


def query_vizier_bright_stars(
    center_ra: float, center_dec: float, radius_deg: float,
    mag_limit: float, siril, log_func, img_width: int, img_height: int,
    siril_lock: threading.Lock | None = None,
) -> list[dict]:
    """
    Query VizieR V/50 (Yale Bright Star Catalogue) for named stars in the FOV.
    """
    try:
        vizier = Vizier(
            columns=["**", "+_RAJ2000", "+_DEJ2000"],
            row_limit=5000,
            column_filters={"Vmag": f"<{mag_limit}"},
        )
        center = SkyCoord(center_ra, center_dec, unit="deg")
        if log_func:
            log_func(f"VizieR BSC: querying V/50 radius={radius_deg:.2f}\u00b0, Vmag<{mag_limit:.1f}")
        results = vizier.query_region(
            center, radius=radius_deg * u.deg, catalog="V/50/catalog",
        )
        if not results or len(results) == 0:
            if log_func:
                log_func("VizieR BSC: no results returned from server")
            return []

        table = results[0]
        if log_func:
            log_func(f"VizieR BSC: {len(table)} stars returned, columns={table.colnames}")

        objects = []
        skipped_pix = 0
        skipped_margin = 0
        skipped_err = 0
        first_err_logged = False
        margin = 30
        for row in table:
            try:
                # Find HR number
                hr = 0
                for hr_col in ("HR", "hr"):
                    if hr_col in table.colnames and not np.ma.is_masked(row[hr_col]):
                        hr = int(row[hr_col])
                        break

                ra_deg = float(row["_RAJ2000"])
                dec_deg = float(row["_DEJ2000"])

                mag = 0.0
                for mag_col in ("Vmag", "mag", "Mag"):
                    if mag_col in table.colnames and not np.ma.is_masked(row[mag_col]):
                        mag = float(row[mag_col])
                        break
                if mag > mag_limit:
                    continue

                # Name: use VizieR Name column, fall back to HR number
                viz_name = ""
                for name_col in ("Name", "name"):
                    if name_col in table.colnames and not np.ma.is_masked(row[name_col]):
                        viz_name = str(row[name_col]).strip()
                        break
                display_name = viz_name if viz_name else f"HR {hr}"

                result_pix = _safe_radec2pix(siril, ra_deg, dec_deg, siril_lock)
                if result_pix is None:
                    skipped_pix += 1
                    continue
                x, y = float(result_pix[0]), float(result_pix[1])
                if not (margin < x < img_width - margin and margin < y < img_height - margin):
                    skipped_margin += 1
                    continue

                objects.append({
                    "name": display_name,
                    "ra": ra_deg,
                    "dec": dec_deg,
                    "type": "Star",
                    "mag": mag,
                    "size_arcmin": 0.0,
                    "common_name": "",  # resolved later via SIMBAD TAP
                    "pixel_x": x,
                    "pixel_y": y,
                })
            except Exception as row_err:
                skipped_err += 1
                if not first_err_logged and log_func:
                    first_err_logged = True
                    log_func(f"VizieR BSC: first row error: {row_err}")
                    try:
                        log_func(f"  row data: {dict(zip(table.colnames, [row[c] for c in table.colnames]))}")
                    except Exception:
                        pass
                continue

        if log_func:
            log_func(f"VizieR BSC: {len(objects)} stars in FOV "
                      f"(skipped: {skipped_pix} pix fail, {skipped_margin} outside, {skipped_err} errors)")
        return objects

    except Exception as e:
        if log_func:
            log_func(f"VizieR BSC query failed: {e}")
            import traceback
            log_func(f"  traceback: {traceback.format_exc().splitlines()[-1]}")
        return []


# ------------------------------------------------------------------------------
# COORDINATE UTILITIES
# ------------------------------------------------------------------------------

def degrees_to_hms(deg: float) -> str:
    """Convert degrees to RA hours:minutes:seconds string."""
    h = deg / 15.0
    hours = int(h)
    m = (h - hours) * 60
    minutes = int(m)
    seconds = (m - minutes) * 60
    return f"{hours:02d}h {minutes:02d}m {seconds:04.1f}s"


def degrees_to_dms(deg: float) -> str:
    """Convert degrees to DEC degrees:arcmin:arcsec string."""
    sign = "+" if deg >= 0 else "-"
    deg = abs(deg)
    d = int(deg)
    m = (deg - d) * 60
    minutes = int(m)
    seconds = (m - minutes) * 60
    return f"{sign}{d:02d}\u00b0 {minutes:02d}' {seconds:04.1f}\""


def choose_grid_step(fov_degrees: float) -> float:
    """Choose an appropriate grid line spacing in degrees for the given FOV."""
    steps = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 30.0, 45.0]
    target_lines = 5  # aim for ~5 grid lines across the FOV
    ideal_step = fov_degrees / target_lines
    for step in steps:
        if step >= ideal_step:
            return step
    return steps[-1]


# ------------------------------------------------------------------------------
# LABEL COLLISION AVOIDANCE
# ------------------------------------------------------------------------------

def resolve_label_collisions(
    objects: list[dict],
    img_width: int = 0,
    img_height: int = 0,
    min_distance_px: int = 80,
) -> list[dict]:
    """
    Assign label offsets to avoid overlapping labels.
    Uses a greedy algorithm with multiple placement positions.
    Clamps labels to stay within image bounds.
    """
    # Generate placement candidates at multiple distances
    placements = []
    for dist in (25, 50, 90, 140):
        placements.extend([
            (dist, dist),      # top-right
            (-dist, dist),     # top-left
            (dist, -dist),     # bottom-right
            (-dist, -dist),    # bottom-left
            (dist, 0),         # right
            (-dist, 0),        # left
            (0, dist),         # above
            (0, -dist),        # below
        ])
    # Spatial grid for O(1) collision checks instead of O(n) per candidate
    _cell = max(min_distance_px, 1)
    _placed_grid: dict[tuple[int, int], int] = {}  # grid cell → count

    def _add_to_grid(px: float, py: float) -> None:
        gx, gy = int(px) // _cell, int(py) // _cell
        _placed_grid[(gx, gy)] = _placed_grid.get((gx, gy), 0) + 1

    def _count_collisions(px: float, py: float) -> int:
        gx, gy = int(px) // _cell, int(py) // _cell
        total = 0
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                total += _placed_grid.get((gx + dx, gy + dy), 0)
        return total

    for obj in objects:
        x, y = obj["pixel_x"], obj["pixel_y"]
        best_offset = placements[0]
        best_score = float("inf")

        # Estimate this label's pixel width based on text length
        name = obj.get("name", "")
        common = obj.get("common_name", "")
        label_text = f"{name} ({common})" if common else name
        est_label_width = len(label_text) * 8

        for offset_x, offset_y in placements:
            lx = x + offset_x
            ly = y + offset_y

            # --- Edge penalty ---
            edge_penalty = 0
            if img_width > 0:
                if offset_x >= 0:
                    remaining = img_width - lx
                    if remaining < est_label_width:
                        edge_penalty += 5000
                else:
                    if lx < est_label_width * 0.3:
                        edge_penalty += 5000
            if img_height > 0:
                if ly < 40 or ly > img_height - 40:
                    edge_penalty += 3000

            # --- Positional bias ---
            side_bias = 0
            if img_width > 0:
                frac_x = x / img_width
                if frac_x > 0.35 and offset_x > 0:
                    side_bias += int(frac_x * 150)
                elif frac_x < 0.25 and offset_x < 0:
                    side_bias += int((1.0 - frac_x) * 80)

            # --- Collision penalty (O(1) grid lookup) ---
            collision_count = _count_collisions(lx, ly)

            dist_penalty = (abs(offset_x) + abs(offset_y)) * 0.3
            score = collision_count * 300 + edge_penalty + side_bias + dist_penalty
            if score < best_score:
                best_score = score
                best_offset = (offset_x, offset_y)

        obj["label_offset_x"] = best_offset[0]
        obj["label_offset_y"] = best_offset[1]
        _add_to_grid(x + best_offset[0], y + best_offset[1])

    return objects


# ------------------------------------------------------------------------------
# RENDERING ENGINE (matplotlib)
# ------------------------------------------------------------------------------

def render_annotated_image(
    image_data: np.ndarray,
    objects: list[dict],
    wcs: WCS | None,
    config: dict,
    output_path: str,
    progress_callback=None,
) -> str:
    """
    Render the annotated image and save to disk.
    Returns the output file path.
    """
    dpi = config.get("dpi", 150)
    height, width = image_data.shape[:2]

    # Cap output resolution for very large mosaics to avoid multi-GB files,
    # excessive render times, and out-of-memory errors.
    # Target: output pixel count <= 8k x 8k (~256 Mpx RGBA buffer).
    MAX_OUTPUT_DIM = 8000
    output_w = width
    output_h = height
    if output_w > MAX_OUTPUT_DIM or output_h > MAX_OUTPUT_DIM:
        scale_factor = min(MAX_OUTPUT_DIM / output_w, MAX_OUTPUT_DIM / output_h)
        dpi = max(72, int(dpi * scale_factor))
        print(f"[AnnotateImage] Large image detected ({width}x{height}), "
              f"reducing output DPI to {dpi} to keep file size manageable")

    # For very large images, downscale the pixel data for matplotlib display.
    # Annotations are vector overlays and render at full quality regardless.
    # This avoids matplotlib duplicating the full array in its Agg backend.
    MAX_DISPLAY_DIM = 6000
    display_data = image_data
    if width > MAX_DISPLAY_DIM or height > MAX_DISPLAY_DIM:
        ds = max(width / MAX_DISPLAY_DIM, height / MAX_DISPLAY_DIM)
        step = max(2, int(ds))
        display_data = image_data[::step, ::step]
        print(f"[AnnotateImage] Downscaling display data by {step}x for rendering "
              f"({display_data.shape[1]}x{display_data.shape[0]})")
        del image_data  # free the full-res copy

    fig_w = width / dpi
    fig_h = height / dpi

    fig = Figure(figsize=(fig_w, fig_h), dpi=dpi, facecolor="black")
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_axes([0, 0, 1, 1])

    # Background image
    # sirilpy preview data is in display order (row 0 = top of image),
    # while radec2pix returns FITS coordinates (y=0 = bottom).
    # Use origin='upper' so the image displays correctly, then
    # flip y-coordinates of annotations: y_display = height - y_fits.
    ax.imshow(display_data, origin="upper", aspect="equal", interpolation="bilinear",
              extent=[0, width, height, 0])
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis("off")

    # Flip y-coordinates from FITS (bottom-up) to display (top-down)
    for obj in objects:
        obj["pixel_y"] = height - obj["pixel_y"]

    if progress_callback:
        progress_callback(20, "Rendering object annotations...")

    # Compute pixel scale (arcsec/pixel)
    pixel_scale = config.get("pixel_scale_arcsec", 1.0)

    # Render object annotations
    total = len(objects)
    for idx, obj in enumerate(objects):
        _render_object(ax, obj, config, pixel_scale)
        if progress_callback and total > 0 and idx % max(1, total // 5) == 0:
            pct = 20 + int(40 * (idx + 1) / total)
            progress_callback(pct, f"Annotating {idx + 1}/{total} objects...")

    if progress_callback:
        progress_callback(65, "Rendering overlays...")

    # Coordinate grid
    if config.get("show_grid", False) and wcs is not None:
        _render_grid(ax, wcs, width, height, config)

    if progress_callback:
        progress_callback(75, "Rendering info box...")

    # Info box
    if config.get("show_info_box", True) and wcs is not None:
        _render_info_box(ax, wcs, width, height, config, len(objects))

    # Compass
    if config.get("show_compass", False) and wcs is not None:
        _render_compass(ax, wcs, width, height, config)

    # Color legend
    if config.get("show_legend", True) and len(objects) > 0:
        _render_legend(ax, width, height, objects, config)

    if progress_callback:
        progress_callback(85, "Saving output file...")

    # Save and free memory
    fig.savefig(output_path, dpi=dpi, facecolor="black", edgecolor="none")
    plt.close(fig)
    del fig, canvas, display_data

    if progress_callback:
        progress_callback(100, "Done!")

    return output_path


def _render_object(ax, obj: dict, config: dict, pixel_scale: float) -> None:
    """Render marker + label for a single catalog object."""
    x, y = obj["pixel_x"], obj["pixel_y"]
    obj_type = obj.get("type", "Other")
    colors = config.get("colors", DEFAULT_COLORS)

    if config.get("color_by_type", True):
        color = colors.get(obj_type, colors.get("Other", "#CCCCCC"))
    else:
        color = "#CCCCCC"  # Uniform color when type coloring is disabled

    font_size = config.get("font_size", 10)
    marker_size = config.get("marker_size", 20)
    line_width = config.get("line_width", 1.5)

    # Marker: ellipse for extended objects, crosshair for point-like
    if config.get("show_ellipses", True) and obj.get("size_arcmin", 0) > 0 and pixel_scale > 0:
        size_px = obj["size_arcmin"] * 60.0 / pixel_scale
        size_px = max(size_px, marker_size * 0.6)
        # Use thinner lines for large ellipses so they don't dominate
        ell_lw = line_width if size_px < 100 else line_width * 0.7
        ell_alpha = 0.8 if size_px < 200 else 0.5
        ellipse = Ellipse(
            (x, y), size_px, size_px,
            fill=False, edgecolor=color, linewidth=ell_lw, alpha=ell_alpha,
        )
        ax.add_patch(ellipse)
    else:
        # Crosshair marker
        half = marker_size * 0.3
        ax.plot([x - half, x + half], [y, y], "-", color=color,
                linewidth=line_width, alpha=0.8)
        ax.plot([x, x], [y - half, y + half], "-", color=color,
                linewidth=line_width, alpha=0.8)

    # Label
    label_parts = []
    name = obj["name"]
    common = obj.get("common_name", "")
    if common and config.get("show_common_names", True):
        label_parts.append(f"{name} ({common})")
    else:
        label_parts.append(name)

    if config.get("show_magnitude", False) and obj.get("mag", 0) != 0:
        label_parts[0] += f"  {obj['mag']:.1f}m"

    if config.get("show_type_label", False):
        type_label = OBJECT_TYPE_LABELS.get(obj_type, obj_type)
        label_parts[0] += f"  [{type_label}]"

    label = label_parts[0]
    offset_x = obj.get("label_offset_x", 15)
    offset_y = obj.get("label_offset_y", 15)

    ha = "left" if offset_x >= 0 else "right"
    va = "bottom" if offset_y >= 0 else "top"

    # Leader line: thin line connecting object position to label
    if config.get("show_leader_lines", True):
        leader_len = math.hypot(offset_x, offset_y)
        if leader_len > 10:  # only draw if label is actually offset
            ax.plot(
                [x, x + offset_x * 0.7], [y, y + offset_y * 0.7],
                "-", color=color, linewidth=0.6, alpha=0.5,
                path_effects=[pe.withStroke(linewidth=1.5, foreground="black")],
            )

    ax.text(
        x + offset_x, y + offset_y,
        label, fontsize=font_size, color=color, fontweight="bold",
        ha=ha, va=va,
        path_effects=[
            pe.withStroke(linewidth=2.5, foreground="black"),
        ],
    )


def _render_grid(ax, wcs: WCS, width: int, height: int, config: dict) -> None:
    """Render RA/DEC coordinate grid overlay."""
    grid_color = config.get("grid_color", "#66AADD")
    grid_alpha = config.get("grid_alpha", 0.6)
    label_size = config.get("grid_label_size", 8)

    # Get FOV corners in world coordinates
    corners_pix = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=float)
    try:
        corners_wcs = wcs.all_pix2world(corners_pix, 0)
    except Exception:
        return

    ra_min, ra_max = corners_wcs[:, 0].min(), corners_wcs[:, 0].max()
    dec_min, dec_max = corners_wcs[:, 1].min(), corners_wcs[:, 1].max()

    # Handle RA wrap-around
    if ra_max - ra_min > 180:
        ras = corners_wcs[:, 0]
        ras[ras > 180] -= 360
        ra_min, ra_max = ras.min(), ras.max()

    fov_size = max(ra_max - ra_min, dec_max - dec_min)
    grid_step = choose_grid_step(fov_size)

    # RA lines
    ra_start = np.floor(ra_min / grid_step) * grid_step
    ra_end = np.ceil(ra_max / grid_step) * grid_step + grid_step
    for ra in np.arange(ra_start, ra_end, grid_step):
        dec_range = np.linspace(dec_min - 1, dec_max + 1, 200)
        ra_arr = np.full_like(dec_range, ra)
        try:
            pixels = wcs.all_world2pix(np.column_stack([ra_arr, dec_range]), 0)
        except Exception:
            continue
        x_px, y_px = pixels[:, 0], pixels[:, 1]
        mask = (x_px >= 0) & (x_px < width) & (y_px >= 0) & (y_px < height)
        if np.any(mask):
            ax.plot(x_px[mask], y_px[mask], "-", color=grid_color,
                    alpha=grid_alpha, linewidth=0.8)
            # Label at first visible point
            idx = np.where(mask)[0][0]
            ra_label = degrees_to_hms(ra % 360)
            ax.text(
                x_px[idx], y_px[idx] + 8, ra_label,
                fontsize=label_size, color=grid_color, alpha=grid_alpha + 0.2,
                ha="center", va="bottom",
                path_effects=[pe.withStroke(linewidth=1.5, foreground="black")],
            )

    # DEC lines
    dec_start = np.floor(dec_min / grid_step) * grid_step
    dec_end = np.ceil(dec_max / grid_step) * grid_step + grid_step
    for dec in np.arange(dec_start, dec_end, grid_step):
        if dec < -90 or dec > 90:
            continue
        ra_range = np.linspace(ra_min - 1, ra_max + 1, 200)
        dec_arr = np.full_like(ra_range, dec)
        try:
            pixels = wcs.all_world2pix(np.column_stack([ra_range, dec_arr]), 0)
        except Exception:
            continue
        x_px, y_px = pixels[:, 0], pixels[:, 1]
        mask = (x_px >= 0) & (x_px < width) & (y_px >= 0) & (y_px < height)
        if np.any(mask):
            ax.plot(x_px[mask], y_px[mask], "-", color=grid_color,
                    alpha=grid_alpha, linewidth=0.8)
            idx = np.where(mask)[0][0]
            dec_label = degrees_to_dms(dec)
            ax.text(
                x_px[idx] + 8, y_px[idx], dec_label,
                fontsize=label_size, color=grid_color, alpha=grid_alpha + 0.2,
                ha="left", va="center",
                path_effects=[pe.withStroke(linewidth=1.5, foreground="black")],
            )


def _render_info_box(
    ax, wcs: WCS, width: int, height: int, config: dict, n_objects: int,
) -> None:
    """Render an info box with field metadata in a corner."""
    try:
        center_coords = wcs.all_pix2world([[width / 2, height / 2]], 0)[0]
        center_ra, center_dec = center_coords

        corner1 = wcs.all_pix2world([[0, 0]], 0)[0]
        corner2 = wcs.all_pix2world([[width, height]], 0)[0]
        fov_w = abs(corner2[0] - corner1[0]) * np.cos(np.radians(center_dec))
        fov_h = abs(corner2[1] - corner1[1])

        pixel_scale = config.get("pixel_scale_arcsec", 0)

        # Rotation from CD matrix
        rotation = 0.0
        if hasattr(wcs.wcs, "cd") and wcs.wcs.cd is not None:
            cd = wcs.wcs.cd
            rotation = np.degrees(np.arctan2(cd[0, 1], cd[0, 0]))
    except Exception:
        return

    info_lines = []
    obj_name = config.get("object_name", "")
    if obj_name:
        info_lines.append(obj_name)

    info_lines.extend([
        f"Center: {degrees_to_hms(center_ra)}  {degrees_to_dms(center_dec)}",
        f"FOV: {fov_w * 60:.1f}' \u00d7 {fov_h * 60:.1f}'",
    ])
    if pixel_scale > 0:
        info_lines.append(f"Scale: {pixel_scale:.2f}\"/px")
    info_lines.append(f"Rotation: {rotation:.1f}\u00b0")
    info_lines.append(f"Objects annotated: {n_objects}")

    info_text = "\n".join(info_lines)

    ax.text(
        width * 0.02, height * 0.98, info_text,
        fontsize=9, color="white", fontfamily="monospace",
        va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="black", alpha=0.6),
        path_effects=[pe.withStroke(linewidth=1, foreground="black")],
    )


def _render_legend(ax, width: int, height: int, objects: list[dict], config: dict) -> None:
    """Render a color legend box showing which types are present in the annotation."""
    clrs = config.get("colors", DEFAULT_COLORS)
    color_by_type = config.get("color_by_type", True)

    # Collect unique types actually present in the annotated objects
    present_types = {}
    for obj in objects:
        t = obj.get("type", "Other")
        if t not in present_types:
            present_types[t] = OBJECT_TYPE_LABELS.get(t, t)

    if not present_types:
        return

    # Sort by the order defined in DEFAULT_COLORS
    type_order = list(DEFAULT_COLORS.keys())
    sorted_types = sorted(
        present_types.items(),
        key=lambda kv: type_order.index(kv[0]) if kv[0] in type_order else 99,
    )

    # Build legend as a single text block with colored lines.
    # Use "\u2588\u2588" (full block chars) as color swatches rendered per-line.
    font_size = max(9, config.get("font_size", 10))

    # Build the full legend text (white) — swatches are rendered separately
    lines = []
    for type_key, type_label in sorted_types:
        lines.append(f"       {type_label}")
    legend_text = "Legend\n" + "\n".join(lines)

    # Render the text block with a background box
    txt = ax.text(
        0.02, 0.02, legend_text,
        fontsize=font_size, color="white", fontweight="bold",
        fontfamily="sans-serif", linespacing=1.6,
        transform=ax.transAxes, ha="left", va="bottom",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="black", alpha=0.8,
                  edgecolor="#888888", linewidth=1),
        path_effects=[pe.withStroke(linewidth=2.5, foreground="black")],
    )

    # Now overlay the colored swatch blocks on each entry line.
    # We need the renderer to get exact text positions.
    fig = ax.get_figure()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox = txt.get_window_extent(renderer=renderer)

    # Convert text bbox to axes coordinates
    inv = ax.transAxes.inverted()
    ax_bbox = inv.transform(bbox)
    x0, y0 = ax_bbox[0]  # bottom-left in axes coords
    x1, y1 = ax_bbox[1]  # top-right in axes coords

    # Each line height in axes fraction
    n_lines = len(sorted_types) + 1  # +1 for title
    total_text_h = y1 - y0
    line_h = total_text_h / n_lines

    for i, (type_key, type_label) in enumerate(sorted_types):
        color = clrs.get(type_key, "#CCCCCC") if color_by_type else "#CCCCCC"
        # Lines go top-down: title is line 0, first entry is line 1, etc.
        # y position for line i+1 (skip title), centered vertically in line
        line_y = y1 - (i + 1.5) * line_h
        ax.text(
            x0 + 0.008, line_y,
            "\u2588\u2588", fontsize=font_size, color=color,
            fontfamily="monospace",
            transform=ax.transAxes, ha="left", va="center",
            path_effects=[pe.withStroke(linewidth=1, foreground="black")],
        )


def _render_compass(ax, wcs: WCS, width: int, height: int, config: dict) -> None:
    """Render N/E compass arrows in the corner."""
    # Place compass in bottom-right corner (in display coordinates)
    cx_display = width * 0.92
    cy_display = height * 0.08

    arrow_len = min(width, height) * 0.06
    font_size = max(11, int(min(width, height) * 0.015))

    # Convert display position to FITS coordinates for WCS
    # Display y is flipped: cy_fits = height - cy_display
    cx_fits = cx_display
    cy_fits = height - cy_display

    try:
        # Get the RA/DEC at the compass position
        ra0, dec0 = wcs.all_pix2world([[cx_fits, cy_fits]], 0)[0]

        # North direction: increase DEC slightly
        delta = 0.01  # degrees
        x_n_fits, y_n_fits = wcs.all_world2pix([[ra0, dec0 + delta]], 0)[0]
        # Convert to display coords
        dx_n = x_n_fits - cx_fits
        dy_n = -(y_n_fits - cy_fits)  # flip y for display
        norm_n = math.hypot(dx_n, dy_n)
        if norm_n > 0:
            dx_n, dy_n = dx_n / norm_n * arrow_len, dy_n / norm_n * arrow_len

        # East direction: decrease RA slightly (RA increases to the East in sky)
        x_e_fits, y_e_fits = wcs.all_world2pix([[ra0 - delta, dec0]], 0)[0]
        dx_e = x_e_fits - cx_fits
        dy_e = -(y_e_fits - cy_fits)  # flip y for display
        norm_e = math.hypot(dx_e, dy_e)
        if norm_e > 0:
            dx_e, dy_e = dx_e / norm_e * arrow_len, dy_e / norm_e * arrow_len
    except Exception:
        return

    compass_color = config.get("compass_color", "#88CCFF")

    # North arrow
    ax.annotate(
        "", (cx_display + dx_n, cy_display + dy_n),
        xytext=(cx_display, cy_display),
        arrowprops=dict(arrowstyle="-|>", color=compass_color, lw=2.5,
                        mutation_scale=15),
    )
    ax.text(
        cx_display + dx_n * 1.35, cy_display + dy_n * 1.35,
        "N", fontsize=font_size, fontweight="bold", color=compass_color,
        ha="center", va="center",
        path_effects=[pe.withStroke(linewidth=3, foreground="black")],
    )

    # East arrow
    ax.annotate(
        "", (cx_display + dx_e, cy_display + dy_e),
        xytext=(cx_display, cy_display),
        arrowprops=dict(arrowstyle="-|>", color=compass_color, lw=2.5,
                        mutation_scale=15),
    )
    ax.text(
        cx_display + dx_e * 1.35, cy_display + dy_e * 1.35,
        "E", fontsize=font_size, fontweight="bold", color=compass_color,
        ha="center", va="center",
        path_effects=[pe.withStroke(linewidth=3, foreground="black")],
    )


# ------------------------------------------------------------------------------
# WCS EXTRACTION
# ------------------------------------------------------------------------------

def extract_wcs_from_header(header_str: str) -> WCS | None:
    """Parse a FITS header string into an astropy WCS object."""
    if not header_str:
        return None
    try:
        hdr = astropy_fits.Header.fromstring(header_str, sep="\n")
        wcs = WCS(hdr)
        if wcs.has_celestial:
            return wcs
    except Exception:
        pass
    # Fallback: try parsing as 80-char FITS card images
    try:
        hdr = astropy_fits.Header.fromstring(header_str)
        wcs = WCS(hdr)
        if wcs.has_celestial:
            return wcs
    except Exception:
        pass
    return None


def extract_wcs_from_fits_file(siril, log_func=None) -> WCS | None:
    """
    Try to read WCS from the actual FITS file on disk.
    Fallback when header string parsing fails.
    """
    def _log(msg: str) -> None:
        if log_func:
            log_func(msg)

    # Strategy A: Use get_image_filename() to find the exact file
    fits_candidates = []
    try:
        img_filename = siril.get_image_filename()
        if img_filename:
            _log(f"WCS disk: image filename = {img_filename}")
            # get_image_filename may return just basename or full path
            if os.path.isfile(img_filename):
                fits_candidates.append(img_filename)
            else:
                # Try combining with working directory
                try:
                    wd = siril.get_siril_wd()
                    if wd:
                        # Siril may omit extension, try common ones
                        base = img_filename
                        for ext in ("", ".fit", ".fits", ".fts"):
                            candidate = os.path.join(wd, base + ext)
                            if os.path.isfile(candidate):
                                fits_candidates.append(candidate)
                                break
                except Exception:
                    pass
    except Exception as e:
        _log(f"WCS disk: get_image_filename() failed: {e}")

    # Strategy B: Scan working directory for FITS files
    if not fits_candidates:
        try:
            wd = siril.get_siril_wd()
            if not wd:
                wd = os.getcwd()
        except Exception:
            wd = os.getcwd()

        _log(f"WCS disk: scanning {wd}")
        try:
            for f in os.listdir(wd):
                if f.lower().endswith((".fit", ".fits", ".fts")):
                    fits_candidates.append(os.path.join(wd, f))
        except OSError:
            return None

        # Use most recently modified
        fits_candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)

    if not fits_candidates:
        _log("WCS disk: no FITS files found")
        return None

    _log(f"WCS disk: trying {len(fits_candidates)} file(s)")

    for fits_path in fits_candidates[:5]:
        try:
            with astropy_fits.open(fits_path) as hdul:
                header = hdul[0].header
                if "CRVAL1" in header and "CRVAL2" in header:
                    wcs = WCS(header)
                    if wcs.has_celestial:
                        _log(f"WCS disk: success from {os.path.basename(fits_path)}")
                        return wcs
                else:
                    _log(f"WCS disk: {os.path.basename(fits_path)} has no CRVAL1/CRVAL2")
        except Exception as e:
            _log(f"WCS disk: {os.path.basename(fits_path)} failed: {e}")
            continue

    return None


def compute_pixel_scale(wcs: WCS) -> float:
    """Compute pixel scale in arcsec/pixel from WCS."""
    try:
        if hasattr(wcs.wcs, "cd") and wcs.wcs.cd is not None:
            cd = wcs.wcs.cd
            scale1 = math.hypot(cd[0, 0], cd[0, 1]) * 3600.0
            scale2 = math.hypot(cd[1, 0], cd[1, 1]) * 3600.0
            return (scale1 + scale2) / 2.0
        elif hasattr(wcs.wcs, "cdelt") and wcs.wcs.cdelt is not None:
            return abs(wcs.wcs.cdelt[0]) * 3600.0
    except Exception:
        pass
    return 1.0


# ------------------------------------------------------------------------------
# PREVIEW WIDGET
# ------------------------------------------------------------------------------

class PreviewWidget(QLabel):
    """Shows a scaled preview of the annotated output image."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(400, 300)
        self.setText("Preview will appear here after annotation.")
        self.setStyleSheet("color: #888888; font-size: 11pt; border: 1px dashed #555555;")
        self._pixmap = None

    def set_image(self, path: str) -> None:
        """Load and display the annotated image from file."""
        from PyQt6.QtGui import QPixmap
        pix = QPixmap(path)
        if not pix.isNull():
            self._pixmap = pix
            scaled = pix.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.setPixmap(scaled)
            self.setStyleSheet("border: 1px solid #555555;")

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._pixmap is not None:
            scaled = self._pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.setPixmap(scaled)


# ------------------------------------------------------------------------------
# MAIN WINDOW
# ------------------------------------------------------------------------------

class AnnotateImageWindow(QMainWindow):
    """
    Main window for the Annotate Image script.

    Left panel: configuration controls (catalogs, display, extras, output).
    Right panel: image preview and status log.
    """

    def __init__(self, siril=None):
        super().__init__()
        self.siril = siril or s.SirilInterface()
        self._image_data = None
        self._img_width = 0
        self._img_height = 0
        self._img_channels = 0
        self._wcs: WCS | None = None
        self._pixel_scale = 1.0
        self._header_str = ""
        self._log_buffer: list[str] = []    # thread-safe log buffer
        self._siril_lock = threading.Lock()  # protect siril API calls from threads
        self._last_output_path = ""
        self._settings = QSettings(SETTINGS_ORG, SETTINGS_APP)
        self.init_ui()
        self._load_settings()
        # Keyboard shortcuts
        QShortcut(QKeySequence("F5"), self, self._on_annotate)

    # ------------------------------------------------------------------
    # LEFT PANEL
    # ------------------------------------------------------------------

    def _build_left_panel(self) -> QWidget:
        left = QWidget()
        left.setFixedWidth(LEFT_PANEL_WIDTH)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(4, 4, 4, 4)

        # Title
        lbl = QLabel(f"Svenesis Annotate Image {VERSION}")
        lbl.setStyleSheet("font-size: 15pt; font-weight: bold; color: #88aaff; margin-top: 5px;")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(lbl)

        self._build_catalogs_group(layout)
        self._build_display_group(layout)
        self._build_extras_group(layout)
        self._build_output_group(layout)
        self._build_action_buttons(layout)

        layout.addStretch()

        # Buy me a Coffee / Help / Close
        btn_coffee = QPushButton("\u2615  Buy me a Coffee")
        _nofocus(btn_coffee)
        btn_coffee.setObjectName("CoffeeButton")
        btn_coffee.setToolTip("Support the development of this tool")
        btn_coffee.clicked.connect(self._show_coffee_dialog)
        btn_help = QPushButton("Help")
        _nofocus(btn_help)
        btn_help.clicked.connect(self._show_help_dialog)
        btn_close = QPushButton("Close")
        _nofocus(btn_close)
        btn_close.setObjectName("CloseButton")
        btn_close.clicked.connect(self.close)
        layout.addWidget(btn_coffee)
        layout.addWidget(btn_help)
        layout.addWidget(btn_close)

        scroll.setWidget(content)

        outer = QVBoxLayout(left)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

        return left

    def _build_catalogs_group(self, parent_layout: QVBoxLayout) -> None:
        group = QGroupBox("Annotate Objects")
        outer = QVBoxLayout(group)

        # Object type checkboxes in two columns using a grid layout
        # Column 1: Common DSO types  |  Column 2: Specialized types
        self._type_checkboxes: dict[str, QCheckBox] = {}

        # (type_key, label, color, default_on, tooltip, column, row)
        type_defs = [
            # --- Column 0: Common DSO types ---
            ("Gal", "Galaxies", "#FFD700", True,
             "Galaxies from NGC/IC (VizieR VII/118) + SIMBAD.\n"
             "Includes galaxy clusters (Abell, Hickson) when\n"
             "Includes galaxy clusters (Abell, Hickson) via SIMBAD. Shown in gold.",
             0, 0),
            ("Neb", "Nebulae", "#FF4444", True,
             "Bright nebulae from NGC/IC catalog (VizieR VII/118).\n"
             "Includes emission, unclassified, and knots in galaxies.\n"
             "Orion, Lagoon, Eagle, Rosette, etc. Shown in red.",
             0, 1),
            ("PN", "Planetary Neb.", "#44FF44", True,
             "Planetary nebulae from NGC/IC catalog.\n"
             "Ring, Dumbbell, Helix, Owl, etc. Shown in green.",
             0, 2),
            ("OC", "Open Clusters", "#44AAFF", True,
             "Open star clusters from NGC/IC catalog.\n"
             "Pleiades, Double Cluster, Wild Duck, etc. Shown in blue.",
             0, 3),
            ("GC", "Globular Clusters", "#FF8800", True,
             "Globular star clusters from NGC/IC catalog.\n"
             "M13, Omega Centauri, 47 Tucanae, etc. Shown in orange.",
             0, 4),
            ("Star", "Named Stars", "#FFFFFF", True,
             "Bright stars from Yale BSC (VizieR V/50) + SIMBAD HD stars.\n"
             "Vega, Deneb, Polaris, Betelgeuse, etc.\n"
             "Filtered by magnitude limit. Shown in white.",
             0, 5),
            # --- Column 1: Specialized types ---
            ("RN", "Reflection Neb.", "#FF8888", False,
             "Reflection nebulae from SIMBAD (dust illuminated by stars).\n"
             "M78, Witch Head, Iris Nebula, vdB catalog, etc.\n"
             "Queried from SIMBAD. Shown in light red.",
             1, 0),
            ("SNR", "Supernova Rem.", "#FF44FF", False,
             "Supernova remnants from SIMBAD.\n"
             "Crab Nebula, Veil Nebula, Simeis 147, etc.\n"
             "Queried from SIMBAD. Shown in magenta.",
             1, 1),
            ("DN", "Dark Nebulae", "#888888", False,
             "Dark nebulae from Barnard catalog (VizieR VII/220A).\n"
             "Horsehead, Pipe, Snake, Barnard's E, etc.\n"
             "Best for Milky Way fields. No mag filter. Shown in grey.",
             1, 2),
            ("HII", "HII Regions", "#FF6666", False,
             "HII regions from Sharpless catalog (VizieR VII/20).\n"
             "Heart, Soul, Barnard's Loop, etc.\n"
             "Best for wide-field Milky Way images. Shown in red-pink.",
             1, 3),
            ("Ast", "Asterisms", "#AADDFF", False,
             "Asterisms (star patterns, not true clusters).\n"
             "Coathanger, Kemble's Cascade, etc. Shown in pale blue.",
             1, 4),
            ("QSO", "Quasars", "#DDAAFF", False,
             "Quasars and AGN from SIMBAD.\n"
             "Queried from SIMBAD. Shown in violet.",
             1, 5),
        ]

        grid = QGridLayout()
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(4)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)

        for type_key, label, color, default_on, tooltip, col, row in type_defs:
            chk = QCheckBox(f" {label}")
            chk.setChecked(default_on)
            chk.setToolTip(tooltip)
            chk.setStyleSheet(
                f"QCheckBox{{color:{color};spacing:3px;font-size:8pt}}"
                f"QCheckBox::indicator{{width:13px;height:13px;border:1px solid #666;"
                f"background:#3c3c3c;border-radius:3px}}"
                f"QCheckBox::indicator:checked{{background:{color};border:1px solid {color}}}"
            )
            _nofocus(chk)
            grid.addWidget(chk, row, col)
            self._type_checkboxes[type_key] = chk

        outer.addLayout(grid)

        # Separator
        sep = QLabel("")
        sep.setFixedHeight(4)
        outer.addWidget(sep)

        # Select All / Deselect All buttons
        btn_row = QHBoxLayout()
        btn_all = QPushButton("Select All")
        btn_all.setStyleSheet("font-size:8pt;padding:3px 8px")
        _nofocus(btn_all)
        btn_all.clicked.connect(lambda: self._set_all_types(True))
        btn_row.addWidget(btn_all)
        btn_none = QPushButton("Deselect All")
        btn_none.setStyleSheet("font-size:8pt;padding:3px 8px")
        _nofocus(btn_none)
        btn_none.clicked.connect(lambda: self._set_all_types(False))
        btn_row.addWidget(btn_none)
        btn_row.addStretch()
        outer.addLayout(btn_row)

        parent_layout.addWidget(group)

    def _set_all_types(self, state: bool) -> None:
        """Set all object type checkboxes to the given state."""
        for chk in self._type_checkboxes.values():
            chk.setChecked(state)

    def _build_display_group(self, parent_layout: QVBoxLayout) -> None:
        group = QGroupBox("Display")
        grid = QGridLayout(group)
        grid.setColumnMinimumWidth(0, 100)
        grid.setColumnMinimumWidth(1, 60)
        grid.setColumnStretch(2, 1)

        # Font size
        row = 0
        lbl = QLabel("Font size:")
        lbl.setFixedWidth(100)
        grid.addWidget(lbl, row, 0)
        self.spin_font = QSpinBox()
        self.spin_font.setRange(6, 24)
        self.spin_font.setValue(10)
        self.spin_font.setFixedWidth(60)
        _nofocus(self.spin_font)
        grid.addWidget(self.spin_font, row, 1)
        self.slider_font = QSlider(Qt.Orientation.Horizontal)
        self.slider_font.setRange(6, 24)
        self.slider_font.setValue(10)
        _nofocus(self.slider_font)
        grid.addWidget(self.slider_font, row, 2)
        self.spin_font.valueChanged.connect(self.slider_font.setValue)
        self.slider_font.valueChanged.connect(self.spin_font.setValue)

        # Marker size
        row = 1
        lbl = QLabel("Marker size:")
        lbl.setFixedWidth(100)
        grid.addWidget(lbl, row, 0)
        self.spin_marker = QSpinBox()
        self.spin_marker.setRange(8, 60)
        self.spin_marker.setValue(20)
        self.spin_marker.setFixedWidth(60)
        _nofocus(self.spin_marker)
        grid.addWidget(self.spin_marker, row, 1)
        self.slider_marker = QSlider(Qt.Orientation.Horizontal)
        self.slider_marker.setRange(8, 60)
        self.slider_marker.setValue(20)
        _nofocus(self.slider_marker)
        grid.addWidget(self.slider_marker, row, 2)
        self.spin_marker.valueChanged.connect(self.slider_marker.setValue)
        self.slider_marker.valueChanged.connect(self.spin_marker.setValue)

        # Magnitude limit
        row = 2
        lbl = QLabel("Mag limit:")
        lbl.setFixedWidth(100)
        grid.addWidget(lbl, row, 0)
        self.spin_mag = QDoubleSpinBox()
        self.spin_mag.setRange(1.0, 20.0)
        self.spin_mag.setValue(12.0)
        self.spin_mag.setSingleStep(0.5)
        self.spin_mag.setDecimals(1)
        self.spin_mag.setFixedWidth(60)
        self.spin_mag.setToolTip(
            "Only annotate objects brighter than this magnitude.\n"
            "Lower values = fewer, brighter objects only.\n"
            "12.0 is a good default for most wide-field images."
        )
        _nofocus(self.spin_mag)
        grid.addWidget(self.spin_mag, row, 1)
        self.slider_mag = QSlider(Qt.Orientation.Horizontal)
        self.slider_mag.setRange(10, 200)  # *10 for 1.0-20.0
        self.slider_mag.setValue(120)
        _nofocus(self.slider_mag)
        grid.addWidget(self.slider_mag, row, 2)
        self.spin_mag.valueChanged.connect(lambda v: self.slider_mag.setValue(int(v * 10)))
        self.slider_mag.valueChanged.connect(lambda v: self.spin_mag.setValue(v / 10.0))

        # Checkboxes in two aligned columns below the sliders
        chk_grid = QGridLayout()
        chk_grid.setHorizontalSpacing(16)
        chk_grid.setVerticalSpacing(4)
        chk_grid.setColumnStretch(0, 1)
        chk_grid.setColumnStretch(1, 1)
        chk_style = "QCheckBox{font-size:8pt;spacing:3px}"

        self.chk_ellipses = QCheckBox("Size ellipses")
        self.chk_ellipses.setChecked(True)
        self.chk_ellipses.setToolTip(
            "Draw ellipses proportional to the cataloged angular size.\n"
            "Disable for a cleaner look with crosshair markers only."
        )
        self.chk_ellipses.setStyleSheet(chk_style)
        _nofocus(self.chk_ellipses)
        chk_grid.addWidget(self.chk_ellipses, 0, 0)

        self.chk_magnitude = QCheckBox("Magnitude")
        self.chk_magnitude.setChecked(False)
        self.chk_magnitude.setToolTip("Append the visual magnitude to each label (e.g. 'M31 3.4m').")
        self.chk_magnitude.setStyleSheet(chk_style)
        _nofocus(self.chk_magnitude)
        chk_grid.addWidget(self.chk_magnitude, 0, 1)

        self.chk_type_label = QCheckBox("Object type")
        self.chk_type_label.setChecked(False)
        self.chk_type_label.setToolTip("Append the object type to each label (e.g. 'M31 [Galaxy]').")
        self.chk_type_label.setStyleSheet(chk_style)
        _nofocus(self.chk_type_label)
        chk_grid.addWidget(self.chk_type_label, 1, 0)

        self.chk_common_names = QCheckBox("Common names")
        self.chk_common_names.setChecked(True)
        self.chk_common_names.setToolTip(
            "Show common names where available (e.g. 'M31 (Andromeda Galaxy)').\n"
            "Disable for a more compact display."
        )
        self.chk_common_names.setStyleSheet(chk_style)
        _nofocus(self.chk_common_names)
        chk_grid.addWidget(self.chk_common_names, 1, 1)

        self.chk_color_by_type = QCheckBox("Color by type")
        self.chk_color_by_type.setChecked(True)
        self.chk_color_by_type.setToolTip(
            "Use different colors for different object types:\n"
            "Gold = Galaxies, Red = Nebulae, Green = Planetary Nebulae,\n"
            "Blue = Open Clusters, Orange = Globular Clusters, etc."
        )
        self.chk_color_by_type.setStyleSheet(chk_style)
        _nofocus(self.chk_color_by_type)
        chk_grid.addWidget(self.chk_color_by_type, 2, 0)

        # Add the checkbox grid below the slider grid
        row = 3
        grid.addLayout(chk_grid, row, 0, 1, 3)

        parent_layout.addWidget(group)

    def _build_extras_group(self, parent_layout: QVBoxLayout) -> None:
        group = QGroupBox("Extras")
        grid = QGridLayout(group)
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(4)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        chk_style = "QCheckBox{font-size:8pt;spacing:3px}"

        self.chk_grid = QCheckBox("RA/DEC grid")
        self.chk_grid.setChecked(True)
        self.chk_grid.setToolTip(
            "Overlay an equatorial coordinate grid with RA/DEC labels.\n"
            "Grid spacing is chosen automatically based on the field of view."
        )
        self.chk_grid.setStyleSheet(chk_style)
        _nofocus(self.chk_grid)
        grid.addWidget(self.chk_grid, 0, 0)

        self.chk_info_box = QCheckBox("Info box")
        self.chk_info_box.setChecked(True)
        self.chk_info_box.setToolTip(
            "Show a semi-transparent info box with:\n"
            "- Center RA/DEC coordinates\n"
            "- Field of view dimensions\n"
            "- Pixel scale and rotation angle\n"
            "- Number of annotated objects"
        )
        self.chk_info_box.setStyleSheet(chk_style)
        _nofocus(self.chk_info_box)
        grid.addWidget(self.chk_info_box, 0, 1)

        self.chk_compass = QCheckBox("N/E compass")
        self.chk_compass.setChecked(False)
        self.chk_compass.setToolTip(
            "Show North and East direction arrows in the corner.\n"
            "Useful for understanding image orientation."
        )
        self.chk_compass.setStyleSheet(chk_style)
        _nofocus(self.chk_compass)
        grid.addWidget(self.chk_compass, 1, 0)

        self.chk_legend = QCheckBox("Color legend")
        self.chk_legend.setChecked(True)
        self.chk_legend.setToolTip(
            "Show a color legend box explaining what each annotation color means.\n"
            "Only shows types that are actually present in the annotated image.\n"
            "Placed in the bottom-left corner."
        )
        self.chk_legend.setStyleSheet(chk_style)
        _nofocus(self.chk_legend)
        grid.addWidget(self.chk_legend, 1, 1)

        self.chk_leader_lines = QCheckBox("Leader lines")
        self.chk_leader_lines.setChecked(True)
        self.chk_leader_lines.setToolTip(
            "Draw thin connecting lines from each label to its object marker.\n"
            "Essential in crowded fields to see which label belongs to which object.\n"
            "Disable for a cleaner look on sparse fields."
        )
        self.chk_leader_lines.setStyleSheet(chk_style)
        _nofocus(self.chk_leader_lines)
        grid.addWidget(self.chk_leader_lines, 2, 0)

        parent_layout.addWidget(group)

    def _build_output_group(self, parent_layout: QVBoxLayout) -> None:
        group = QGroupBox("Output")
        layout = QVBoxLayout(group)

        # Format radio buttons
        fmt_layout = QHBoxLayout()
        fmt_label = QLabel("Format:")
        fmt_layout.addWidget(fmt_label)
        self.radio_png = QRadioButton("PNG")
        self.radio_png.setChecked(True)
        _nofocus(self.radio_png)
        self.radio_tiff = QRadioButton("TIFF")
        _nofocus(self.radio_tiff)
        self.radio_jpeg = QRadioButton("JPEG")
        _nofocus(self.radio_jpeg)
        self.format_group = QButtonGroup(self)
        self.format_group.addButton(self.radio_png)
        self.format_group.addButton(self.radio_tiff)
        self.format_group.addButton(self.radio_jpeg)
        fmt_layout.addWidget(self.radio_png)
        fmt_layout.addWidget(self.radio_tiff)
        fmt_layout.addWidget(self.radio_jpeg)
        fmt_layout.addStretch()
        layout.addLayout(fmt_layout)

        # DPI
        dpi_layout = QHBoxLayout()
        dpi_label = QLabel("DPI:")
        dpi_label.setFixedWidth(100)
        dpi_layout.addWidget(dpi_label)
        self.spin_dpi = QSpinBox()
        self.spin_dpi.setRange(72, 300)
        self.spin_dpi.setValue(150)
        self.spin_dpi.setFixedWidth(60)
        self.spin_dpi.setToolTip("Output resolution. Higher = larger file.\n150 is good for screens, 300 for print.")
        _nofocus(self.spin_dpi)
        dpi_layout.addWidget(self.spin_dpi)
        self.slider_dpi = QSlider(Qt.Orientation.Horizontal)
        self.slider_dpi.setRange(72, 300)
        self.slider_dpi.setValue(150)
        _nofocus(self.slider_dpi)
        dpi_layout.addWidget(self.slider_dpi)
        self.spin_dpi.valueChanged.connect(self.slider_dpi.setValue)
        self.slider_dpi.valueChanged.connect(self.spin_dpi.setValue)
        layout.addLayout(dpi_layout)

        # Filename
        name_layout = QHBoxLayout()
        name_label = QLabel("Filename:")
        name_label.setFixedWidth(100)
        name_layout.addWidget(name_label)
        self.edit_filename = QLineEdit("annotated")
        self.edit_filename.setToolTip(
            "Base filename for the output (without extension).\n"
            "A timestamp will be appended automatically."
        )
        name_layout.addWidget(self.edit_filename)
        layout.addLayout(name_layout)

        parent_layout.addWidget(group)

    def _build_action_buttons(self, parent_layout: QVBoxLayout) -> None:
        group = QGroupBox("Actions")
        layout = QVBoxLayout(group)

        self.btn_annotate = QPushButton("Annotate Image")
        self.btn_annotate.setObjectName("AnnotateButton")
        self.btn_annotate.setToolTip("Annotate the current image and export (F5)")
        _nofocus(self.btn_annotate)
        self.btn_annotate.clicked.connect(self._on_annotate)
        layout.addWidget(self.btn_annotate)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setVisible(False)
        self.progress.setTextVisible(True)
        layout.addWidget(self.progress)

        self.lbl_status = QLabel("Status: Ready")
        self.lbl_status.setStyleSheet("color: #888888; font-size: 9pt;")
        layout.addWidget(self.lbl_status)

        parent_layout.addWidget(group)

    # ------------------------------------------------------------------
    # RIGHT PANEL
    # ------------------------------------------------------------------

    def _build_right_panel(self) -> QWidget:
        right = QWidget()
        r_layout = QVBoxLayout(right)
        r_layout.setContentsMargins(4, 4, 4, 4)

        # Image info bar
        self.lbl_image_info = QLabel("No image loaded")
        self.lbl_image_info.setStyleSheet(
            "font-size: 10pt; color: #aaaaaa; padding: 4px; "
            "background-color: #333333; border-radius: 4px;"
        )
        self.lbl_image_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        r_layout.addWidget(self.lbl_image_info)

        # Tab widget for preview and log
        self.tabs = QTabWidget()

        # Preview tab
        self.preview_widget = PreviewWidget()
        self.tabs.addTab(self.preview_widget, "Preview")

        # Log tab
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet(
            "background-color: #1e1e1e; color: #cccccc; font-family: monospace; font-size: 9pt;"
        )
        self.tabs.addTab(self.log_text, "Log")

        r_layout.addWidget(self.tabs, 1)

        # Open output folder button
        btn_row = QHBoxLayout()
        self.btn_open_folder = QPushButton("Open Output Folder")
        _nofocus(self.btn_open_folder)
        self.btn_open_folder.setEnabled(False)
        self.btn_open_folder.clicked.connect(self._on_open_folder)
        btn_row.addWidget(self.btn_open_folder)

        self.btn_open_image = QPushButton("Open Annotated Image")
        _nofocus(self.btn_open_image)
        self.btn_open_image.setEnabled(False)
        self.btn_open_image.clicked.connect(self._on_open_image)
        btn_row.addWidget(self.btn_open_image)

        r_layout.addLayout(btn_row)

        return right

    # ------------------------------------------------------------------
    # INIT UI
    # ------------------------------------------------------------------

    def init_ui(self) -> None:
        main = QWidget()
        self.setCentralWidget(main)
        layout = QHBoxLayout(main)
        layout.addWidget(self._build_left_panel())
        layout.addWidget(self._build_right_panel(), 1)
        self.setWindowTitle("Svenesis Annotate Image")
        self.setStyleSheet(DARK_STYLESHEET)
        self.resize(1200, 800)

    # ------------------------------------------------------------------
    # PERSISTENT SETTINGS
    # ------------------------------------------------------------------

    def _load_settings(self) -> None:
        st = self._settings
        # Object type checkboxes
        # Must match defaults in _build_catalogs_group type_defs
        type_defaults = {"Gal": True, "Neb": True, "PN": True,
                         "OC": True, "GC": True, "Star": True,
                         "RN": False, "SNR": False, "DN": False,
                         "HII": False, "Ast": False, "QSO": False}
        for key, chk in self._type_checkboxes.items():
            chk.setChecked(st.value(f"type_{key}", type_defaults.get(key, True), type=bool))
        self.spin_font.setValue(int(st.value("font_size", 10)))
        self.spin_marker.setValue(int(st.value("marker_size", 20)))
        self.spin_mag.setValue(float(st.value("mag_limit", 12.0)))
        self.chk_ellipses.setChecked(st.value("show_ellipses", True, type=bool))
        self.chk_magnitude.setChecked(st.value("show_magnitude", False, type=bool))
        self.chk_type_label.setChecked(st.value("show_type_label", False, type=bool))
        self.chk_common_names.setChecked(st.value("show_common_names", True, type=bool))
        self.chk_color_by_type.setChecked(st.value("color_by_type", True, type=bool))
        self.chk_grid.setChecked(st.value("show_grid", True, type=bool))
        self.chk_info_box.setChecked(st.value("show_info_box", True, type=bool))
        self.chk_compass.setChecked(st.value("show_compass", False, type=bool))
        self.chk_legend.setChecked(st.value("show_legend", True, type=bool))
        self.chk_leader_lines.setChecked(st.value("show_leader_lines", True, type=bool))
        self.spin_dpi.setValue(int(st.value("dpi", 150)))

        fmt = st.value("output_format", "PNG")
        if fmt == "TIFF":
            self.radio_tiff.setChecked(True)
        elif fmt == "JPEG":
            self.radio_jpeg.setChecked(True)
        else:
            self.radio_png.setChecked(True)

    def _save_settings(self) -> None:
        st = self._settings
        for key, chk in self._type_checkboxes.items():
            st.setValue(f"type_{key}", chk.isChecked())
        st.setValue("font_size", self.spin_font.value())
        st.setValue("marker_size", self.spin_marker.value())
        st.setValue("mag_limit", self.spin_mag.value())
        st.setValue("show_ellipses", self.chk_ellipses.isChecked())
        st.setValue("show_magnitude", self.chk_magnitude.isChecked())
        st.setValue("show_type_label", self.chk_type_label.isChecked())
        st.setValue("show_common_names", self.chk_common_names.isChecked())
        st.setValue("color_by_type", self.chk_color_by_type.isChecked())
        st.setValue("show_grid", self.chk_grid.isChecked())
        st.setValue("show_info_box", self.chk_info_box.isChecked())
        st.setValue("show_compass", self.chk_compass.isChecked())
        st.setValue("show_legend", self.chk_legend.isChecked())
        st.setValue("show_leader_lines", self.chk_leader_lines.isChecked())
        st.setValue("dpi", self.spin_dpi.value())

        if self.radio_tiff.isChecked():
            st.setValue("output_format", "TIFF")
        elif self.radio_jpeg.isChecked():
            st.setValue("output_format", "JPEG")
        else:
            st.setValue("output_format", "PNG")

    def closeEvent(self, event) -> None:
        self._save_settings()
        super().closeEvent(event)

    # ------------------------------------------------------------------
    # LOGGING
    # ------------------------------------------------------------------

    def _log(self, msg: str) -> None:
        """Append a message to the log tab and Siril console.
        Thread-safe: if called from a worker thread, buffers the message
        for later flush on the main thread.
        """
        if threading.current_thread() is not threading.main_thread():
            # Buffer messages from worker threads — Qt widgets are NOT thread-safe
            self._log_buffer.append(f"[AnnotateImage] {msg}")
            return
        self.log_text.append(f"[AnnotateImage] {msg}")
        try:
            self.siril.log(f"[AnnotateImage] {msg}")
        except (SirilError, OSError, RuntimeError, AttributeError):
            pass

    def _flush_log_buffer(self) -> None:
        """Flush buffered log messages from worker threads to the UI (main thread only)."""
        while self._log_buffer:
            msg = self._log_buffer.pop(0)
            self.log_text.append(msg)
            try:
                self.siril.log(msg)
            except (SirilError, OSError, RuntimeError, AttributeError):
                pass

    # ------------------------------------------------------------------
    # PROGRESS
    # ------------------------------------------------------------------

    def _update_progress(self, value: int, label: str = "") -> None:
        self.progress.setValue(value)
        if label:
            self.lbl_status.setText(f"Status: {label}")
        QApplication.processEvents()

    # ------------------------------------------------------------------
    # SIRIL IMAGE LOADING
    # ------------------------------------------------------------------

    def _load_from_siril(self) -> bool:
        """Load the current image and WCS from Siril."""
        no_image_msg = "No image is currently loaded in Siril. Please load a FITS image first."
        try:
            if not self.siril.connected:
                self.siril.connect()

            with self.siril.image_lock():
                # Get preview image data (auto-stretched, 8-bit)
                try:
                    preview = self.siril.get_image_pixeldata(preview=True, linked=True)
                    if preview is not None:
                        self._image_data = np.array(preview, dtype=np.uint8)
                    else:
                        raise ValueError("No preview data")
                except Exception:
                    # Fallback: get raw data — process in-place to limit memory
                    fit = self.siril.get_image()
                    fit.ensure_data_type(np.float32)
                    data = np.array(fit.data, dtype=np.float32)
                    del fit  # free the siril image copy
                    # Simple autostretch for display
                    #if data.ndim == 3:
                        # (C, H, W) -> (H, W, C)
                        #data = np.transpose(data, (1, 2, 0))
                    if data.ndim == 2:
                        data = np.stack([data] * 3, axis=-1)
                    # Normalize to 0-255 in-place
                    vmin, vmax = np.percentile(data, [1, 99])
                    if vmax > vmin:
                        data -= vmin
                        data *= (255.0 / (vmax - vmin))
                        np.clip(data, 0, 255, out=data)
                    self._image_data = data.astype(np.uint8)
                    del data  # free float32 copy

                fit = self.siril.get_image()
                self._img_width = fit.width
                self._img_height = fit.height
                self._img_channels = fit.channels

            # Ensure image data is (H, W, 3)
            if self._image_data is not None:
                if self._image_data.ndim == 3 and self._image_data.shape[0] in (1, 3):
                    # (C, H, W) -> (H, W, C)
                    self._image_data = np.transpose(self._image_data, (1, 2, 0))
                if self._image_data.ndim == 2:
                    self._image_data = np.stack([self._image_data] * 3, axis=-1)
                if self._image_data.shape[2] == 1:
                    self._image_data = np.repeat(self._image_data, 3, axis=2)

            # --- WCS / plate-solve detection ---
            # Uses the same approach as Galaxy_Annotations.py from siril-scripts.
            self._is_plate_solved = False
            self._wcs = None
            self._header_str = ""

            # Step 1: Get FITS header as DICT and build WCS (Galaxy_Annotations approach)
            try:
                header_dict = self.siril.get_image_fits_header(return_as="dict")
                if header_dict:
                    self._log(f"FITS header dict: got {len(header_dict)} keys")
                    wcs = WCS(header_dict, naxis=[1, 2])
                    if wcs.has_celestial:
                        self._wcs = wcs
                        self._is_plate_solved = True
                        self._log("WCS: extracted from header dict")
                    else:
                        self._log("WCS: header dict has no celestial WCS")
                else:
                    self._log("FITS header dict: returned empty/None")
            except Exception as e:
                self._log(f"FITS header dict: failed: {e}")

            # Step 2: Fallback — header as string
            if self._wcs is None:
                try:
                    hdr_str = self.siril.get_image_fits_header(return_as="str")
                    if hdr_str:
                        self._header_str = hdr_str
                        self._log(f"FITS header str: got {len(hdr_str)} chars")
                        self._wcs = extract_wcs_from_header(hdr_str)
                        if self._wcs is not None:
                            self._is_plate_solved = True
                            self._log("WCS: extracted from header string")
                        else:
                            self._log("WCS: header string parsing failed")
                            if "CRVAL1" in hdr_str:
                                self._is_plate_solved = True
                except Exception as e:
                    self._log(f"FITS header str: failed: {e}")

            # Step 3: Check keywords for plate-solve flag
            if not self._is_plate_solved:
                try:
                    kw = self.siril.get_image_keywords()
                    if kw is not None:
                        self._is_plate_solved = bool(getattr(kw, "pltsolvd", False))
                        if not self._is_plate_solved:
                            self._is_plate_solved = bool(getattr(kw, "wcsdata", None))
                        self._log(f"Keywords: pltsolvd={getattr(kw, 'pltsolvd', '?')}")
                except Exception as e:
                    self._log(f"Keywords: failed: {e}")

            # Step 4: Build WCS from pix2radec sampling (if plate-solved but no WCS yet)
            if self._wcs is None and self._is_plate_solved:
                self._log("WCS: trying pix2radec sampling...")
                self._wcs = self._build_wcs_from_siril()
                if self._wcs is not None:
                    self._log("WCS: built from sirilpy pix2radec")

            # Step 5: Try pix2radec even without plate-solve flag
            if self._wcs is None:
                try:
                    cx, cy = self._img_width / 2, self._img_height / 2
                    result = self.siril.pix2radec(cx, cy)
                    if result is not None:
                        ra, dec = result
                        self._log(f"WCS: pix2radec works! center=({ra:.4f}, {dec:.4f})")
                        self._is_plate_solved = True
                        self._wcs = self._build_wcs_from_siril()
                        if self._wcs is not None:
                            self._log("WCS: built from sirilpy pix2radec")
                    else:
                        self._log("WCS: pix2radec returned None")
                except Exception as e:
                    self._log(f"WCS: pix2radec failed: {e}")

            # Step 6: Last resort — read FITS file from disk
            if self._wcs is None:
                self._log("WCS: trying FITS file on disk...")
                self._wcs = extract_wcs_from_fits_file(self.siril, log_func=self._log)
                if self._wcs is not None:
                    self._is_plate_solved = True
                    self._log("WCS: extracted from FITS file on disk")

            # Finalize
            if self._wcs is not None:
                self._is_plate_solved = True
                self._pixel_scale = compute_pixel_scale(self._wcs)
            else:
                self._pixel_scale = 1.0

            # Update info bar
            ch_str = "RGB" if self._img_channels >= 3 else "Mono"
            wcs_str = "plate-solved \u2713" if self._is_plate_solved else "NOT plate-solved \u2717"
            self.lbl_image_info.setText(
                f"{self._img_width} \u00d7 {self._img_height} px  |  {ch_str}  |  {wcs_str}"
            )
            self._log(f"Result: plate_solved={self._is_plate_solved}, "
                       f"wcs={'OK' if self._wcs else 'NONE'}, "
                       f"scale={self._pixel_scale:.2f}\"/px")

            return True

        except NoImageError:
            QMessageBox.warning(self, "No Image", no_image_msg)
            return False
        except SirilConnectionError:
            QMessageBox.warning(
                self, "Connection Error",
                "Could not connect to Siril. Make sure Siril is running."
            )
            return False
        except SirilError as e:
            if "no image" in str(e).lower():
                QMessageBox.warning(self, "No Image", no_image_msg)
            else:
                QMessageBox.warning(self, "Siril Error", str(e))
            return False
        except Exception as e:
            QMessageBox.critical(
                self, "Error",
                f"Failed to load image:\n{e}\n\n{traceback.format_exc()}"
            )
            return False

    def _build_wcs_from_siril(self) -> WCS | None:
        """
        Build an astropy WCS object by sampling sirilpy's pix2radec.
        Used as last resort when header parsing fails but Siril has WCS internally.
        """
        try:
            w, h = self._img_width, self._img_height
            cx, cy = w / 2.0, h / 2.0

            # Get center coordinates
            ra_c, dec_c = self.siril.pix2radec(cx, cy)

            # Get pixel scale and rotation by sampling nearby points
            delta = 10.0  # pixels
            ra_r, dec_r = self.siril.pix2radec(cx + delta, cy)
            ra_u, dec_u = self.siril.pix2radec(cx, cy + delta)

            # Build CD matrix from finite differences
            cos_dec = np.cos(np.radians(dec_c))
            cd1_1 = (ra_r - ra_c) * cos_dec / delta  # dRA/dx (degrees/pixel)
            cd1_2 = (ra_u - ra_c) * cos_dec / delta  # dRA/dy
            cd2_1 = (dec_r - dec_c) / delta            # dDEC/dx
            cd2_2 = (dec_u - dec_c) / delta            # dDEC/dy

            # Handle RA wrap-around
            if cd1_1 > 180:
                cd1_1 -= 360
            elif cd1_1 < -180:
                cd1_1 += 360
            if cd1_2 > 180:
                cd1_2 -= 360
            elif cd1_2 < -180:
                cd1_2 += 360

            # Create WCS
            wcs = WCS(naxis=2)
            wcs.wcs.crpix = [cx + 1, cy + 1]  # FITS is 1-indexed
            wcs.wcs.crval = [ra_c, dec_c]
            wcs.wcs.cd = np.array([[cd1_1, cd1_2], [cd2_1, cd2_2]])
            wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
            wcs.array_shape = (h, w)

            # Validate: round-trip check
            test_ra, test_dec = wcs.all_pix2world([[cx, cy]], 0)[0]
            if abs(test_ra - ra_c) < 0.01 and abs(test_dec - dec_c) < 0.01:
                return wcs
            else:
                self._log(f"WCS: round-trip check failed: "
                          f"({test_ra:.4f},{test_dec:.4f}) vs ({ra_c:.4f},{dec_c:.4f})")
        except Exception as e:
            self._log(f"WCS: _build_wcs_from_siril failed: {e}")
        return None

    # ------------------------------------------------------------------
    # ANNOTATION
    # ------------------------------------------------------------------

    def _on_annotate(self) -> None:
        """Main annotation workflow."""
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.log_text.clear()

        self._log("Starting annotation...")
        self._update_progress(5, "Loading image from Siril...")

        if not self._load_from_siril():
            self.progress.setVisible(False)
            self.lbl_status.setText("Status: Failed to load image")
            return

        self._log(f"Image loaded: {self._img_width} x {self._img_height} px")

        if not self._is_plate_solved or self._wcs is None:
            if self._is_plate_solved and self._wcs is None:
                msg = (
                    "The image appears plate-solved but the WCS data could not be read.\n\n"
                    "Try saving the image as FITS first, then reload and run this script.\n"
                    "The FITS file must be in Siril's working directory."
                )
            else:
                msg = (
                    "The image is not plate-solved. Please run Plate Solving first.\n\n"
                    "In Siril: Tools \u2192 Astrometry \u2192 Image Plate Solver..."
                )
            QMessageBox.warning(self, "Not Plate-Solved", msg)
            self._log("ERROR: Image is not plate-solved or WCS could not be read. Aborting.")
            self.progress.setVisible(False)
            self.lbl_status.setText("Status: Image not plate-solved")
            return

        # Log WCS info
        try:
            center = self._wcs.all_pix2world(
                [[self._img_width / 2, self._img_height / 2]], 0
            )[0]
            self._log(f"WCS: Center RA={degrees_to_hms(center[0])}  DEC={degrees_to_dms(center[1])}")
            self._log(f"Pixel Scale: {self._pixel_scale:.2f}\"/px")
        except Exception:
            pass

        self._update_progress(10, "Querying online catalogs...")

        # Collect catalog objects
        all_objects: list[dict] = []
        mag_limit = self.spin_mag.value()

        # Determine which object types are enabled
        enabled_types = {k for k, chk in self._type_checkboxes.items() if chk.isChecked()}
        self._log(f"Enabled types: {', '.join(sorted(enabled_types))}")

        # Calculate FOV radius for catalog queries
        cx, cy = self._img_width / 2, self._img_height / 2
        try:
            center_result = self.siril.pix2radec(cx, cy)
            corner_result = self.siril.pix2radec(0, 0)
            if center_result and corner_result:
                c1 = SkyCoord(center_result[0], center_result[1], unit="deg")
                c2 = SkyCoord(corner_result[0], corner_result[1], unit="deg")
                fov_radius_deg = c1.separation(c2).deg
                center_ra, center_dec = center_result
            else:
                self._log("ERROR: Cannot determine FOV — pix2radec failed")
                return
        except Exception as e:
            self._log(f"ERROR: FOV calculation failed: {e}")
            return

        fov_w = self._img_width * self._pixel_scale / 3600
        fov_h = self._img_height * self._pixel_scale / 3600
        self._log(f"FOV: {fov_w:.2f}\u00b0 x {fov_h:.2f}\u00b0 "
                  f"({fov_w * 60:.1f}' x {fov_h * 60:.1f}'), "
                  f"query radius={fov_radius_deg:.2f}\u00b0 ({fov_radius_deg * 60:.1f}')")

        # Common kwargs for all VizieR query functions
        query_kw = dict(
            center_ra=center_ra, center_dec=center_dec,
            radius_deg=fov_radius_deg,
            siril=self.siril, log_func=self._log,
            img_width=self._img_width, img_height=self._img_height,
            siril_lock=self._siril_lock,
        )

        catalog_summary = []

        # ── Parallel catalog queries ──────────────────────────────
        # All VizieR + SIMBAD queries are independent network calls.
        # Run them concurrently to reduce total wall-clock time.
        self._update_progress(12, "Querying online catalogs (parallel)...")

        dso_types = {"Gal", "Neb", "OC", "GC", "PN", "RN", "SNR", "Ast", "Other"}
        futures: dict[str, any] = {}

        with ThreadPoolExecutor(max_workers=5, thread_name_prefix="catalog") as pool:
            if enabled_types & dso_types:
                futures["ngc"] = pool.submit(
                    query_vizier_ngc_ic, mag_limit=mag_limit, **query_kw)
            if "HII" in enabled_types:
                futures["sh"] = pool.submit(query_vizier_sharpless, **query_kw)
            if "DN" in enabled_types:
                futures["bn"] = pool.submit(query_vizier_barnard, **query_kw)
            if "Star" in enabled_types:
                futures["star"] = pool.submit(
                    query_vizier_bright_stars, mag_limit=mag_limit, **query_kw)
            futures["simbad"] = pool.submit(self._query_simbad, mag_limit)

            # Collect results as they complete (order doesn't matter for futures,
            # but we process in catalog priority order for deduplication)
            results: dict[str, list[dict]] = {}
            for key, future in futures.items():
                try:
                    results[key] = future.result(timeout=120)
                except Exception as exc:
                    self._log(f"Catalog query '{key}' failed: {exc}")
                    results[key] = []

        # Flush log messages buffered by worker threads
        self._flush_log_buffer()

        # ── Merge results (sequential — dedup depends on order) ──
        if "ngc" in results and results["ngc"]:
            ngc_objs = results["ngc"]
            typed = [o for o in ngc_objs if o["type"] in enabled_types]
            if ngc_objs:
                filtered_out = len(ngc_objs) - len(typed)
                if filtered_out > 0:
                    self._log(f"  NGC/IC type filter: {len(typed)} kept, "
                              f"{filtered_out} excluded by type selection")
            if typed:
                catalog_summary.append(f"NGC/IC: {len(typed)}")
                all_objects.extend(typed)

        if "sh" in results and results["sh"]:
            deduped = self._deduplicate(results["sh"], all_objects)
            if deduped:
                catalog_summary.append(f"Sharpless: {len(deduped)}")
                all_objects.extend(deduped)

        if "bn" in results and results["bn"]:
            deduped = self._deduplicate(results["bn"], all_objects)
            if deduped:
                catalog_summary.append(f"Barnard: {len(deduped)}")
                all_objects.extend(deduped)

        if "star" in results and results["star"]:
            deduped = self._deduplicate(results["star"], all_objects)
            if deduped:
                catalog_summary.append(f"Stars: {len(deduped)}")
                all_objects.extend(deduped)

        if "simbad" in results and results["simbad"]:
            typed = [o for o in results["simbad"] if o["type"] in enabled_types]
            deduped = self._deduplicate(typed, all_objects)
            if deduped:
                catalog_summary.append(f"SIMBAD: {len(deduped)}")
                all_objects.extend(deduped)

        self._log("\u2500" * 40)
        self._log(f"Catalogs loaded: {', '.join(catalog_summary)}")
        self._log(f"Magnitude limit: {mag_limit:.1f}")
        self._log(f"Total objects in field: {len(all_objects)}")

        # Resolve common names via SIMBAD TAP (batch query)
        if all_objects:
            self._update_progress(25, "Resolving common names via SIMBAD...")
            resolve_common_names(all_objects, log_func=self._log)

        if len(all_objects) == 0:
            self._log("No catalog objects found in the field of view.")
            self._log("The image will be exported with grid/info box only (if enabled).")

        # Log found objects
        for obj in all_objects:
            common = f" ({obj['common_name']})" if obj.get("common_name") else ""
            mag_str = f"{obj['mag']:.1f}" if obj['mag'] != 0.0 else "?"
            size_str = f"  size={obj['size_arcmin']:.1f}'" if obj.get('size_arcmin', 0) > 0 else ""
            self._log(f"  {obj['name']}{common}  [{obj['type']}]  mag={mag_str}{size_str}")

        self._log("\u2500" * 40)

        self._update_progress(15, "Resolving label positions...")

        # Resolve label collisions
        all_objects = resolve_label_collisions(all_objects, self._img_width, self._img_height)

        # Build config
        config = {
            "colors": DEFAULT_COLORS,
            "font_size": self.spin_font.value(),
            "marker_size": self.spin_marker.value(),
            "line_width": 1.5,
            "pixel_scale_arcsec": self._pixel_scale,
            "show_ellipses": self.chk_ellipses.isChecked(),
            "show_magnitude": self.chk_magnitude.isChecked(),
            "show_type_label": self.chk_type_label.isChecked(),
            "show_common_names": self.chk_common_names.isChecked(),
            "color_by_type": self.chk_color_by_type.isChecked(),
            "show_grid": self.chk_grid.isChecked(),
            "show_info_box": self.chk_info_box.isChecked(),
            "show_compass": self.chk_compass.isChecked(),
            "show_legend": self.chk_legend.isChecked(),
            "show_leader_lines": self.chk_leader_lines.isChecked(),
            "dpi": self.spin_dpi.value(),
            "grid_color": "#66AADD",
            "grid_alpha": 0.6,
            "grid_label_size": 8,
            "compass_color": "#88CCFF",
        }

        # Try to get object name from FITS header
        try:
            kw = self.siril.get_keywords()
            if kw is not None:
                obj_name = getattr(kw, "object", "") or ""
                if not obj_name:
                    # Try objctra/objctdec as fallback object identifier
                    obj_name = getattr(kw, "object_name", "") or ""
                if obj_name:
                    config["object_name"] = obj_name
        except Exception:
            pass
        # Fallback: try parsing OBJECT from FITS header string
        if "object_name" not in config and self._header_str:
            for line in self._header_str.split("\n"):
                if line.startswith("OBJECT"):
                    val = line.split("=", 1)[-1].split("/")[0].strip().strip("'\" ")
                    if val:
                        config["object_name"] = val
                        break

        # Output path
        ext_map = {"PNG": ".png", "TIFF": ".tiff", "JPEG": ".jpg"}
        if self.radio_tiff.isChecked():
            ext = ext_map["TIFF"]
        elif self.radio_jpeg.isChecked():
            ext = ext_map["JPEG"]
        else:
            ext = ext_map["PNG"]

        base_name = self.edit_filename.text().strip() or "annotated"
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{base_name}_{timestamp}{ext}"

        # Save next to the image if possible, otherwise working directory
        try:
            wd = self.siril.get_siril_wd()
            output_dir = wd if wd else os.getcwd()
        except Exception:
            output_dir = os.getcwd()

        output_path = os.path.join(output_dir, filename)

        # Render
        try:
            result_path = render_annotated_image(
                self._image_data,
                all_objects,
                self._wcs,
                config,
                output_path,
                progress_callback=self._update_progress,
            )
            # Free image data after rendering — no longer needed
            self._image_data = None
            gc.collect()
            self._last_output_path = result_path
            self.btn_open_folder.setEnabled(True)
            self.btn_open_image.setEnabled(True)

            # Show preview
            self.preview_widget.set_image(result_path)
            self.tabs.setCurrentIndex(0)  # switch to preview tab

            file_size = os.path.getsize(result_path)
            size_str = f"{file_size / 1024 / 1024:.1f} MB" if file_size > 1024 * 1024 else f"{file_size / 1024:.0f} KB"
            self._log(f"Output: {result_path} ({size_str})")
            self._log(f"Annotation complete: {len(all_objects)} objects annotated.")
            self.lbl_status.setText(f"Status: Done \u2014 {len(all_objects)} objects annotated")

        except Exception as e:
            self._log(f"ERROR: {e}")
            QMessageBox.critical(
                self, "Rendering Error",
                f"Failed to render annotated image:\n{e}\n\n{traceback.format_exc()}"
            )
            self.lbl_status.setText("Status: Error during rendering")

        self.progress.setVisible(False)

    # ------------------------------------------------------------------
    # OUTPUT ACTIONS
    # ------------------------------------------------------------------

    def _query_simbad(self, mag_limit: float) -> list[dict]:
        """
        Query SIMBAD for all objects in the field of view.
        Returns list of dicts compatible with embedded catalog format.
        """
        try:
            Simbad  # verify import succeeded at module level
        except NameError:
            self._log("SIMBAD: astroquery not installed. Run: pip install astroquery")
            QMessageBox.warning(
                self, "Missing Package",
                "The 'astroquery' package is required for SIMBAD queries.\n\n"
                "Install it with: pip install astroquery"
            )
            return []

        try:
            # Get field center and radius via siril.pix2radec (thread-safe)
            cx, cy = self._img_width / 2, self._img_height / 2
            center_result = _safe_pix2radec(self.siril, cx, cy, self._siril_lock)
            corner_result = _safe_pix2radec(self.siril, 0, 0, self._siril_lock)
            if center_result is None or corner_result is None:
                self._log("SIMBAD: pix2radec failed for center/corner")
                return []
            center_ra, center_dec = center_result
            corner_ra, corner_dec = corner_result
            center_coord = SkyCoord(center_ra, center_dec, unit="deg")
            corner_coord = SkyCoord(corner_ra, corner_dec, unit="deg")
            fov_radius = center_coord.separation(corner_coord)

            self._log(f"SIMBAD: FOV radius={fov_radius.arcmin:.1f}' "
                      f"around RA={center_ra:.4f} DEC={center_dec:.4f}")

            # Configure Simbad query
            custom_simbad = Simbad()
            custom_simbad.add_votable_fields("V", "galdim_majaxis", "otype")
            custom_simbad.ROW_LIMIT = 2000

            # For wide fields (> 1°), tile the query to avoid SIMBAD row
            # limits filling up with objects from only part of the FOV.
            MAX_QUERY_RADIUS_DEG = 0.75  # SIMBAD works best with ≤45' radius
            fov_radius_deg = fov_radius.deg

            if fov_radius_deg <= MAX_QUERY_RADIUS_DEG:
                query_centers = [center_coord]
                query_radius = fov_radius
            else:
                # Tile the FOV with overlapping circles
                step = MAX_QUERY_RADIUS_DEG * 1.2  # overlap factor
                half_w_deg = (self._img_width * self._pixel_scale / 3600) / 2
                half_h_deg = (self._img_height * self._pixel_scale / 3600) / 2
                query_centers = []
                dec_start = center_dec - half_h_deg
                dec_end = center_dec + half_h_deg
                d = dec_start
                while d <= dec_end + step * 0.5:
                    # RA step needs cos(dec) correction
                    ra_step = step / max(0.1, np.cos(np.radians(d)))
                    ra_start = center_ra - half_w_deg / max(0.1, np.cos(np.radians(d)))
                    ra_end = center_ra + half_w_deg / max(0.1, np.cos(np.radians(d)))
                    r = ra_start
                    while r <= ra_end + ra_step * 0.5:
                        query_centers.append(SkyCoord(r, d, unit="deg"))
                        r += ra_step
                    d += step
                query_radius = Angle(MAX_QUERY_RADIUS_DEG, unit="deg")
                self._log(f"SIMBAD: wide field — tiling into {len(query_centers)} queries "
                          f"(radius={query_radius.arcmin:.1f}' each)")

            # Execute queries — parallelize tiles for wide fields
            from astropy.table import vstack  # lightweight, rarely used elsewhere
            all_results = []
            if len(query_centers) == 1:
                # Single query — no need for threading overhead
                try:
                    r = custom_simbad.query_region(query_centers[0], radius=query_radius)
                    if r is not None and len(r) > 0:
                        all_results.append(r)
                        self._log(f"SIMBAD: {len(r)} objects")
                except Exception as e:
                    self._log(f"SIMBAD: query failed: {e}")
            else:
                # Parallel tile queries — each thread gets its own Simbad instance
                # because astroquery's Simbad is NOT thread-safe
                def _query_tile(idx_qc):
                    i, qc = idx_qc
                    tile_simbad = Simbad()
                    tile_simbad.add_votable_fields("V", "galdim_majaxis", "otype")
                    tile_simbad.ROW_LIMIT = 2000
                    return i, tile_simbad.query_region(qc, radius=query_radius)

                with ThreadPoolExecutor(
                    max_workers=min(4, len(query_centers)),
                    thread_name_prefix="simbad_tile",
                ) as tile_pool:
                    tile_futures = tile_pool.map(_query_tile, enumerate(query_centers))
                    for i, r in tile_futures:
                        if r is not None and len(r) > 0:
                            all_results.append(r)
                            self._log(f"SIMBAD: tile {i + 1}/{len(query_centers)}: "
                                      f"{len(r)} objects")

            if not all_results:
                self._log("SIMBAD: no objects found")
                return []

            if len(all_results) == 1:
                result = all_results[0]
            else:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")  # suppress vstack MergeConflictWarning
                    result = vstack(all_results)
                # Remove duplicates by main_id
                seen = set()
                unique_mask = []
                id_col_tmp = None
                for c in result.colnames:
                    if c.lower() == "main_id":
                        id_col_tmp = c
                        break
                if id_col_tmp:
                    for row in result:
                        name = str(row[id_col_tmp]).strip()
                        if name not in seen:
                            seen.add(name)
                            unique_mask.append(True)
                        else:
                            unique_mask.append(False)
                    result = result[unique_mask]

            self._log(f"SIMBAD: {len(result)} objects returned")

            # Detect column names (astroquery changed them across versions)
            colnames = [c.lower() for c in result.colnames]
            id_col = "main_id" if "main_id" in colnames else "MAIN_ID"
            ra_col = "ra" if "ra" in colnames else "RA"
            dec_col = "dec" if "dec" in colnames else "DEC"
            # Find the actual column names case-sensitively
            for c in result.colnames:
                if c.lower() == "main_id":
                    id_col = c
                elif c.lower() == "ra":
                    ra_col = c
                elif c.lower() == "dec":
                    dec_col = c

            self._log(f"SIMBAD: columns = {result.colnames}")

            # SIMBAD object type to our type mapping
            type_map = {
                "G": "Gal", "GiG": "Gal", "GiP": "Gal", "BiC": "Gal",
                "IG": "Gal", "PaG": "Gal", "SBG": "Gal", "SyG": "Gal",
                "Sy1": "Gal", "Sy2": "Gal", "AGN": "Gal", "LIN": "Gal",
                "ClG": "Gal", "GrG": "Gal", "CGG": "Gal",  # Galaxy clusters/groups (Abell, Hickson)
                "EmG": "Gal", "H2G": "Gal", "bCG": "Gal", "LSB": "Gal",  # Galaxy subtypes
                "HII": "HII", "RNe": "RN", "ISM": "Neb", "EmO": "Neb",
                "DNe": "DN", "dNe": "DN", "MoC": "DN",  # Dark nebulae / molecular clouds
                "SNR": "SNR", "PN": "PN", "Cl*": "OC", "GlC": "GC",
                "OpC": "OC", "As*": "Ast", "QSO": "QSO",
                "**": "Star", "*": "Star", "V*": "Star", "PM*": "Star",
                "HB*": "Star", "C*": "Star", "S*": "Star", "LP*": "Star",
                "Mi*": "Star", "sr*": "Star", "Ce*": "Star", "RR*": "Star",
                "WR*": "Star", "Be*": "Star", "Pe*": "Star", "HV*": "Star",
                "No*": "Star", "Psr": "Star", "**?": "Star",
                "Galaxy": "Gal", "GinCl": "Gal", "GinPair": "Gal",
                "StarburstG": "Gal", "Seyfert": "Gal", "BClG": "Gal",
                "InteractingG": "Gal",
            }

            # Only keep objects from well-known astrophotography catalogs.
            # Everything else (survey IDs, sub-regions, etc.) is filtered out.
            # Pre-grouped by first character for O(1) dispatch instead of O(n) scan.
            _USEFUL_PREFIXES = (
                "NGC", "IC ", "IC1", "IC2", "IC3", "IC4", "IC5",
                "M ", "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9",
                "UGC", "MCG", "Arp", "Abell", "Mrk", "VV ",
                "Ced", "Sh2", "LBN", "LDN", "Barnard", "B ",
                "Cr ", "Mel", "Tr ", "Pal", "PGC", "ESO", "CGCG",
                "Hickson", "HCG", "Minkowski", "PK", "PN G",
                "Hen", "He ", "Haro", "Ho ", "K ", "Terzan",
                "NAME ", "V*", "HD ", "HR ",
                "vdB", "RCW", "Gum", "Collinder", "Stock", "DWB",
            )
            # Build first-char → prefix list for fast filtering
            _prefix_by_char: dict[str, list[str]] = {}
            for p in _USEFUL_PREFIXES:
                _prefix_by_char.setdefault(p[0], []).append(p)

            def _is_useful(name: str) -> bool:
                candidates = _prefix_by_char.get(name[0]) if name else None
                return candidates is not None and any(name.startswith(p) for p in candidates)

            def _safe_float(val, default: float = 0.0) -> float:
                """Safely convert a possibly-masked value to float."""
                try:
                    if val is None or val is np.ma.masked:
                        return default
                    f = float(val)
                    return default if np.isnan(f) else f
                except (ValueError, TypeError):
                    return default

            # Pre-resolve column names once (avoid per-row lookups)
            _size_col = None
            for sc in ("galdim_majaxis", "GALDIM_MAJAXIS", "DIM_MAJAXIS", "dim_majaxis"):
                if sc in result.colnames:
                    _size_col = sc
                    break
            _otype_col = None
            for tc in ("otype", "OTYPE"):
                if tc in result.colnames:
                    _otype_col = tc
                    break
            _has_V = "V" in result.colnames

            objects = []
            skipped_junk = 0
            for row in result:
                try:
                    name = str(row[id_col]).strip()
                except Exception:
                    continue

                # Filter: only keep objects from well-known catalogs
                if not _is_useful(name):
                    skipped_junk += 1
                    continue

                # Parse coordinates
                try:
                    ra = _safe_float(row[ra_col], default=float("nan"))
                    dec = _safe_float(row[dec_col], default=float("nan"))
                    if np.isnan(ra) or np.isnan(dec):
                        # Fall back to sexagesimal parsing
                        coord = SkyCoord(
                            str(row[ra_col]), str(row[dec_col]),
                            unit=(u.hourangle, u.deg),
                        )
                        ra = coord.ra.deg
                        dec = coord.dec.deg
                except Exception:
                    continue

                # Magnitude
                mag = _safe_float(row["V"] if _has_V else None, default=0.0)
                if mag != 0.0 and mag > mag_limit:
                    continue

                # Size
                size = _safe_float(row[_size_col], 0.0) if _size_col else 0.0

                # Type
                otype = ""
                if _otype_col:
                    try:
                        otype = str(row[_otype_col]).strip()
                    except Exception:
                        pass
                obj_type = type_map.get(otype, None)
                if obj_type is None:
                    # Any star-like otype (contains '*') maps to Star
                    if "*" in otype:
                        obj_type = "Star"
                    else:
                        obj_type = "Other"

                # Convert to pixel via siril.radec2pix (thread-safe)
                try:
                    result_pix = _safe_radec2pix(self.siril, ra, dec, self._siril_lock)
                    if result_pix is None:
                        continue
                    x, y = float(result_pix[0]), float(result_pix[1])
                except Exception:
                    continue

                margin = 30
                if margin < x < self._img_width - margin and margin < y < self._img_height - margin:
                    objects.append({
                        "name": name,
                        "ra": ra,
                        "dec": dec,
                        "type": obj_type,
                        "mag": mag,
                        "size_arcmin": size,
                        "common_name": "",
                        "pixel_x": x,
                        "pixel_y": y,
                    })

            self._log(f"SIMBAD: {len(objects)} useful objects in FOV "
                      f"({skipped_junk} survey catalog entries filtered out)")
            return objects

        except Exception as e:
            self._log(f"SIMBAD: query failed: {e}")
            return []

    @staticmethod
    def _deduplicate(
        new_objs: list[dict], existing: list[dict], min_dist: int = 40,
    ) -> list[dict]:
        """Remove objects from new_objs that are too close to or have the same name as existing ones."""
        def _norm_name(n: str) -> str:
            return _RE_WHITESPACE.sub(' ', n).strip().upper()
        existing_names = {_norm_name(ex["name"]) for ex in existing}

        # Spatial grid for fast proximity lookup.
        # Use cell = min_dist so 3x3 neighborhood covers all candidates.
        # Store actual coordinates per cell for precise distance check.
        cell = max(min_dist, 1)
        grid: dict[tuple[int, int], list[tuple[float, float]]] = {}
        for ex in existing:
            gx, gy = int(ex["pixel_x"]) // cell, int(ex["pixel_y"]) // cell
            grid.setdefault((gx, gy), []).append((ex["pixel_x"], ex["pixel_y"]))

        deduped = []
        for obj in new_objs:
            if _norm_name(obj["name"]) in existing_names:
                continue
            # Check 3x3 neighborhood with actual distance
            ox, oy = obj["pixel_x"], obj["pixel_y"]
            gx = int(ox) // cell
            gy = int(oy) // cell
            too_close = False
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for (ex, ey) in grid.get((gx + dx, gy + dy), ()):
                        if abs(ox - ex) < min_dist and abs(oy - ey) < min_dist:
                            too_close = True
                            break
                    if too_close:
                        break
                if too_close:
                    break
            if not too_close:
                deduped.append(obj)
                existing_names.add(_norm_name(obj["name"]))
                grid.setdefault((gx, gy), []).append((ox, oy))
        return deduped

    def _on_open_folder(self) -> None:
        if self._last_output_path:
            folder = os.path.dirname(self._last_output_path)
            QDesktopServices.openUrl(QUrl.fromLocalFile(folder))

    def _on_open_image(self) -> None:
        if self._last_output_path and os.path.isfile(self._last_output_path):
            QDesktopServices.openUrl(QUrl.fromLocalFile(self._last_output_path))

    # ------------------------------------------------------------------
    # BUY ME A COFFEE
    # ------------------------------------------------------------------

    def _show_coffee_dialog(self) -> None:
        BMC_URL = "https://buymeacoffee.com/sramuschkat"

        dlg = QDialog(self)
        dlg.setWindowTitle("\u2615 Support Svenesis Annotate Image")
        dlg.setMinimumSize(520, 480)
        dlg.setStyleSheet(
            "QDialog{background-color:#1e1e1e;color:#e0e0e0}"
            "QLabel{color:#e0e0e0}"
            "QPushButton{font-weight:bold;padding:8px;border-radius:6px}"
        )
        layout = QVBoxLayout(dlg)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(12)

        # Combined header + message
        header_msg = QLabel(
            "<div style='text-align:center; font-size:12pt; line-height:1.6;'>"
            "<span style='font-size:48pt;'>\u2615</span><br>"
            "<span style='font-size:18pt; font-weight:bold; color:#FFDD00;'>"
            "Buy me a Coffee</span><br><br>"
            "<b style='color:#e0e0e0;'>Enjoying Svenesis Annotate Image?</b><br><br>"
            "This tool is free and open source. It's built with love for the "
            "astrophotography community by <b style='color:#88aaff;'>Sven Ramuschkat</b> "
            "(<span style='color:#88aaff;'>svenesis.org</span>).<br><br>"
            "If this tool has helped you create beautiful annotated images "
            "or made sharing your astrophotography easier \u2014 consider buying "
            "me a coffee to keep development going!<br><br>"
            "<span style='color:#FFDD00;'>\u2615 Every coffee fuels a new feature, "
            "bug fix, or clear-sky night of testing.</span><br><br>"
            "<span style='color:#aaaaaa;'>Your support helps maintain:</span><br>"
            "\u2022 Svenesis Gradient Analyzer \u2022 Svenesis Image Advisor<br>"
            "\u2022 Svenesis Annotate Image \u2022 Svenesis Script Security Scanner<br>"
            "</div>"
        )
        header_msg.setWordWrap(True)
        header_msg.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_msg.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(header_msg)

        layout.addSpacing(8)

        # Open link button
        btn_open = QPushButton("\u2615  Buy me a Coffee  \u2615")
        btn_open.setStyleSheet(
            "QPushButton{"
            "  background-color:#FFDD00; color:#000000;"
            "  font-size:14pt; font-weight:bold;"
            "  padding:12px 24px; border-radius:8px;"
            "  border:2px solid #ccb100;"
            "}"
            "QPushButton:hover{"
            "  background-color:#ffe740; border-color:#ddcc00;"
            "}"
        )
        btn_open.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_open.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(BMC_URL)))
        layout.addWidget(btn_open)

        layout.addSpacing(4)

        # Close button
        btn_close = QPushButton("Close")
        btn_close.setStyleSheet(
            "QPushButton{background-color:#444;color:#ddd;border:1px solid #666;padding:6px}"
            "QPushButton:hover{background-color:#555}"
        )
        _nofocus(btn_close)
        btn_close.clicked.connect(dlg.accept)
        layout.addWidget(btn_close)

        # Footer
        footer = QLabel(
            f"<div style='text-align:center; line-height:1.8;'>"
            f"<a style='color:#88aaff; font-size:12pt;' href='{BMC_URL}'>{BMC_URL}</a><br>"
            f"<span style='font-size:13pt; color:#999;'>"
            f"Thank you for supporting open-source astrophotography tools!<br>"
            f"Clear skies \u2728</span>"
            f"</div>"
        )
        footer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        footer.setTextFormat(Qt.TextFormat.RichText)
        footer.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextBrowserInteraction
        )
        footer.setOpenExternalLinks(True)
        layout.addWidget(footer)

        dlg.exec()

    # ------------------------------------------------------------------
    # HELP
    # ------------------------------------------------------------------

    def _show_help_dialog(self) -> None:
        dlg = QDialog(self)
        dlg.setWindowTitle("Svenesis Annotate Image \u2014 Help")
        dlg.setMinimumSize(800, 600)
        dlg.setStyleSheet(
            "QDialog{background-color:#1e1e1e;color:#e0e0e0}"
            "QLabel{color:#e0e0e0}"
        )
        layout = QVBoxLayout(dlg)
        layout.setContentsMargins(12, 12, 12, 12)

        tabs = QTabWidget()

        # --- Getting Started ---
        tab1 = QTextEdit()
        tab1.setReadOnly(True)
        tab1.setHtml(
            "<h2 style='color:#88aaff;'>Getting Started</h2>"
            "<p><b>What does Annotate Image do?</b></p>"
            "<p>This script takes your plate-solved astrophotography image and creates "
            "an annotated version with labeled deep-sky objects, stars, coordinate grids, "
            "and field information \u2014 ready to share on social media or forums.</p>"
            "<hr>"
            "<p><b>Requirements:</b></p>"
            "<ul>"
            "<li><b>Plate-solved image:</b> Your image must have a WCS (World Coordinate System) "
            "solution in its FITS header. In Siril: <i>Tools \u2192 Astrometry \u2192 Image Plate Solver</i>.</li>"
            "<li><b>Internet connection:</b> The script queries VizieR and SIMBAD online "
            "for catalog data \u2014 no hardcoded object lists.</li>"
            "<li>The script will check for plate-solve status automatically and warn you if missing.</li>"
            "</ul>"
            "<hr>"
            "<p><b>Quick Start:</b></p>"
            "<ol>"
            "<li>Load a plate-solved image in Siril</li>"
            "<li>Run this script</li>"
            "<li>Select which <b>object types</b> to annotate (left panel checkboxes)</li>"
            "<li>Adjust display settings (font size, magnitude limit, etc.)</li>"
            "<li>Click <b>Annotate Image</b> (or press <b>F5</b>)</li>"
            "<li>The annotated image is saved as PNG/TIFF/JPEG in Siril's working directory</li>"
            "<li>Use <b>Open Image</b> or <b>Open Folder</b> buttons to view the result</li>"
            "</ol>"
            "<hr>"
            "<p><b>How it works:</b></p>"
            "<p>The script queries five online catalogs <b>in parallel</b> for fast results:</p>"
            "<ul>"
            "<li><b>VizieR VII/118</b> (NGC 2000.0) \u2014 NGC, IC, and Messier objects</li>"
            "<li><b>VizieR VII/20</b> (Sharpless) \u2014 HII regions</li>"
            "<li><b>VizieR VII/220A</b> (Barnard) \u2014 Dark nebulae</li>"
            "<li><b>VizieR V/50</b> (Yale BSC) \u2014 Named bright stars</li>"
            "<li><b>SIMBAD</b> \u2014 Supplementary objects (UGC, Abell, Arp, Hickson, "
            "vdB, Markarian, etc.) plus common name resolution</li>"
            "</ul>"
            "<p>Objects are filtered by the <b>type checkboxes</b> you select \u2014 so you control "
            "<i>what kinds</i> of objects appear, not which catalog they come from. "
            "Duplicates across catalogs are automatically removed.</p>"
            "<hr>"
            "<p><b>Color Coding:</b></p>"
            "<table cellpadding='4'>"
            "<tr><td style='color:#FFD700;'>\u2588\u2588</td><td>Galaxies</td>"
            "<td style='color:#FF4444;'>\u2588\u2588</td><td>Nebulae</td></tr>"
            "<tr><td style='color:#44FF44;'>\u2588\u2588</td><td>Planetary Nebulae</td>"
            "<td style='color:#44AAFF;'>\u2588\u2588</td><td>Open Clusters</td></tr>"
            "<tr><td style='color:#FF8800;'>\u2588\u2588</td><td>Globular Clusters</td>"
            "<td style='color:#FF44FF;'>\u2588\u2588</td><td>Supernova Remnants</td></tr>"
            "<tr><td style='color:#888888;'>\u2588\u2588</td><td>Dark Nebulae</td>"
            "<td style='color:#FFFFFF;'>\u2588\u2588</td><td>Named Stars</td></tr>"
            "<tr><td style='color:#FF6666;'>\u2588\u2588</td><td>HII Regions</td>"
            "<td style='color:#FF8888;'>\u2588\u2588</td><td>Reflection Nebulae</td></tr>"
            "<tr><td style='color:#AADDFF;'>\u2588\u2588</td><td>Asterisms</td>"
            "<td style='color:#DDAAFF;'>\u2588\u2588</td><td>Quasars</td></tr>"
            "</table>"
            "<hr>"
            "<p><b>Large Mosaics:</b> The script handles very large images by "
            "automatically downscaling the display data for rendering and capping "
            "the output resolution. Annotations remain sharp as vector overlays.</p>"
        )
        tabs.addTab(tab1, "Getting Started")

        # --- Object Types ---
        tab2 = QTextEdit()
        tab2.setReadOnly(True)
        tab2.setHtml(
            "<h2 style='color:#88aaff;'>Object Types</h2>"
            "<p>Select which types of astronomical objects to annotate using the "
            "checkboxes in the left panel. All catalogs are always queried \u2014 "
            "you control visibility by object type, not by data source.</p>"
            "<p>The left column contains common types (enabled by default), "
            "the right column contains specialized types (disabled by default). "
            "Use <b>Select All</b> / <b>Deselect All</b> for quick toggling.</p>"
            "<hr>"
            "<table cellpadding='6' style='width:100%'>"
            "<tr style='background:#2a2a2a'>"
            "<td colspan='2'><b style='color:#88aaff'>Common Types (default ON)</b></td></tr>"
            "<tr><td style='color:#FFD700;font-weight:bold;width:150px'>Galaxies</td>"
            "<td>Spiral, elliptical, irregular galaxies, galaxy clusters (Abell, Hickson), "
            "and galaxy groups from NGC/IC and SIMBAD. "
            "M31, M51, M81, NGC 4565, Centaurus A, etc.</td></tr>"
            "<tr><td style='color:#FF4444;font-weight:bold'>Nebulae</td>"
            "<td>Bright emission nebulae from NGC/IC catalog. "
            "Orion, Lagoon, Eagle, Rosette, Carina, etc.</td></tr>"
            "<tr><td style='color:#44FF44;font-weight:bold'>Planetary Nebulae</td>"
            "<td>Dying star shells. Ring (M57), Dumbbell (M27), "
            "Helix, Owl, Cat's Eye, Ghost of Jupiter, etc.</td></tr>"
            "<tr><td style='color:#44AAFF;font-weight:bold'>Open Clusters</td>"
            "<td>Young star groups. Pleiades (M45), Double Cluster, "
            "Wild Duck, Beehive, Jewel Box, etc.</td></tr>"
            "<tr><td style='color:#FF8800;font-weight:bold'>Globular Clusters</td>"
            "<td>Ancient dense star balls. M13, M3, Omega Centauri, "
            "47 Tucanae, M22, etc.</td></tr>"
            "<tr><td style='color:#FFFFFF;font-weight:bold'>Named Stars</td>"
            "<td>Bright stars from the Yale BSC and SIMBAD HD stars. "
            "Vega, Deneb, Polaris, Betelgeuse, etc. Filtered by magnitude limit.</td></tr>"
            "<tr style='background:#2a2a2a'>"
            "<td colspan='2'><b style='color:#88aaff'>Specialized Types (default OFF)</b></td></tr>"
            "<tr><td style='color:#FF8888;font-weight:bold'>Reflection Nebulae</td>"
            "<td>Dust clouds illuminated by nearby stars (from SIMBAD). "
            "M78, Witch Head, Iris, vdB catalog, etc.</td></tr>"
            "<tr><td style='color:#FF44FF;font-weight:bold'>Supernova Remnants</td>"
            "<td>Explosion debris (from SIMBAD). Crab (M1), Veil Nebula, "
            "Simeis 147, Pencil Nebula, etc.</td></tr>"
            "<tr><td style='color:#888888;font-weight:bold'>Dark Nebulae</td>"
            "<td>Opaque dust clouds from the Barnard catalog. Horsehead (B33), "
            "Pipe (B78), Snake (B72), Coalsack. <b>Not filtered by magnitude.</b></td></tr>"
            "<tr><td style='color:#FF6666;font-weight:bold'>HII Regions</td>"
            "<td>Sharpless catalog ionized hydrogen regions. Large emission "
            "complexes: Heart, Soul, Barnard's Loop. "
            "Best for wide-field Milky Way images.</td></tr>"
            "<tr><td style='color:#AADDFF;font-weight:bold'>Asterisms</td>"
            "<td>Star patterns that are not true clusters. "
            "Coathanger, Kemble's Cascade, etc.</td></tr>"
            "<tr><td style='color:#DDAAFF;font-weight:bold'>Quasars</td>"
            "<td>Quasi-stellar objects and AGN from SIMBAD. "
            "Extremely distant active galactic nuclei.</td></tr>"
            "</table>"
            "<hr>"
            "<p><b>Magnitude Limit:</b> Only objects brighter than this value "
            "are annotated. 12.0 is a good default for most images. "
            "Dark nebulae bypass this filter since they have no standard magnitude. "
            "Set to 20.0 to see all cataloged objects regardless of brightness.</p>"
        )
        tabs.addTab(tab2, "Object Types")

        # --- Display Options ---
        tab3 = QTextEdit()
        tab3.setReadOnly(True)
        tab3.setHtml(
            "<h2 style='color:#88aaff;'>Display Options</h2>"
            "<h3>Controls</h3>"
            "<p><b>Font Size:</b> Controls the label text size (6\u201324 pt). "
            "Larger for social media sharing, smaller for high-resolution prints. "
            "Default: 10 pt.</p>"
            "<p><b>Marker Size:</b> Controls the crosshair marker size for point-like "
            "objects without catalog size data (8\u201360 px). Default: 20 px.</p>"
            "<p><b>Magnitude Limit:</b> Slider and spin box to set the faintest magnitude "
            "to include (1.0\u201320.0). Lower = fewer, brighter objects only.</p>"
            "<hr>"
            "<h3>Display Checkboxes</h3>"
            "<table cellpadding='4' style='width:100%'>"
            "<tr><td style='width:140px'><b>Size ellipses</b></td>"
            "<td>Show extended objects (galaxies, nebulae) as ellipses proportional "
            "to their cataloged angular size. When off, all objects use crosshair markers. "
            "Large ellipses (&gt;100 px) are drawn thinner; very large ones (&gt;200 px) "
            "are also more transparent.</td></tr>"
            "<tr><td><b>Magnitude</b></td>"
            "<td>Append the visual magnitude to each label (e.g. 'M31 3.4m').</td></tr>"
            "<tr><td><b>Object type</b></td>"
            "<td>Append the type designation (e.g. 'M31 [Galaxy]').</td></tr>"
            "<tr><td><b>Common names</b></td>"
            "<td>Show well-known names where available (e.g. 'NGC 224 (Andromeda Galaxy)'). "
            "Names are resolved via SIMBAD TAP query. Catalog-like names "
            "(FAUST, IRAS, 2MASS, etc.) are automatically filtered out.</td></tr>"
            "<tr><td><b>Color by type</b></td>"
            "<td>Use different colors per object category (see color table). "
            "When off, all annotations are drawn in a uniform light grey.</td></tr>"
            "</table>"
            "<hr>"
            "<h3>Extras Checkboxes</h3>"
            "<table cellpadding='4' style='width:100%'>"
            "<tr><td style='width:140px'><b>RA/DEC grid</b></td>"
            "<td>Overlays equatorial coordinate grid lines with RA (hours) and DEC (degrees) "
            "labels. Grid spacing is chosen automatically based on the field of view "
            "(from arcminutes for narrow fields to degrees for mosaics).</td></tr>"
            "<tr><td><b>Info box</b></td>"
            "<td>Semi-transparent box (top-left) showing center RA/DEC, "
            "field of view dimensions, pixel scale, rotation angle, and the number "
            "of annotated objects.</td></tr>"
            "<tr><td><b>N/E compass</b></td>"
            "<td>North and East direction arrows (bottom-right) derived from the WCS "
            "solution. Shows the true orientation of the image on the sky.</td></tr>"
            "<tr><td><b>Color legend</b></td>"
            "<td>Auto-generated legend (bottom-left) with colored swatches for each "
            "object type present in the annotation. Types not in the image are omitted.</td></tr>"
            "<tr><td><b>Leader lines</b></td>"
            "<td>Thin connecting lines from each label to its object marker. "
            "Essential in crowded fields. Can be disabled for a cleaner look.</td></tr>"
            "</table>"
        )
        tabs.addTab(tab3, "Display & Extras")

        # --- Output ---
        tab4 = QTextEdit()
        tab4.setReadOnly(True)
        tab4.setHtml(
            "<h2 style='color:#88aaff;'>Output</h2>"
            "<p><b>Format:</b></p>"
            "<ul>"
            "<li><b>PNG:</b> Lossless compression, best for sharing. Recommended default.</li>"
            "<li><b>TIFF:</b> Lossless, larger files. Good for further editing.</li>"
            "<li><b>JPEG:</b> Lossy compression, smaller files. Good for web uploads.</li>"
            "</ul>"
            "<p><b>DPI:</b> Controls the output resolution (72\u2013300). "
            "150 DPI is good for screens and social media. "
            "300 DPI for print-quality output (larger file size). "
            "For very large images (&gt;8000 px), DPI is automatically reduced "
            "to keep file size manageable.</p>"
            "<p><b>Filename:</b> Base name for the output file. A timestamp is appended "
            "automatically (e.g. <i>annotated_20260405_193649.png</i>) to avoid "
            "overwriting previous annotations.</p>"
            "<p>The output file is saved in Siril's working directory (the same folder "
            "as your image).</p>"
            "<hr>"
            "<h3 style='color:#88aaff;'>Performance</h3>"
            "<p>All five catalog queries run <b>in parallel</b> using thread pools, "
            "so annotation typically takes just 2\u20135 seconds even for wide fields.</p>"
            "<p>For wide-field mosaics (&gt;1\u00b0), SIMBAD queries are automatically "
            "tiled into overlapping regions and queried in parallel.</p>"
            "<p><b>Memory:</b> Large images (&gt;6000 px) are automatically downscaled "
            "for the display layer. Annotations remain sharp as vector overlays. "
            "Image data is freed after rendering to minimize memory usage.</p>"
        )
        tabs.addTab(tab4, "Output & Performance")

        layout.addWidget(tabs)
        lbl_guide = QLabel(
            '<span style="font-size:10pt;">📖 '
            '<a href="https://github.com/sramuschkat/Siril-Scripts/blob/main/'
            'Instructions/Svenesis-AnnotateImage-Instructions.md"'
            ' style="color:#88aaff;">Full User Guide (online)</a></span>'
        )
        lbl_guide.setOpenExternalLinks(True)
        layout.addWidget(lbl_guide)
        btn = QPushButton("Close")
        _nofocus(btn)
        btn.clicked.connect(dlg.accept)
        layout.addWidget(btn)
        dlg.exec()


# ------------------------------------------------------------------------------
# ENTRY POINT
# ------------------------------------------------------------------------------

def main() -> int:
    app = QApplication(sys.argv)
    try:
        siril = s.SirilInterface()
        win = AnnotateImageWindow(siril)
        win.showMaximized()
        try:
            siril.log(f"Svenesis Annotate Image v{VERSION} loaded.")
        except (SirilError, OSError, RuntimeError):
            pass
        return app.exec()
    except NoImageError:
        QMessageBox.warning(
            None,
            "No Image",
            "No image is currently loaded in Siril. Please load an image first."
        )
        return 1
    except Exception as e:
        QMessageBox.critical(
            None,
            "Svenesis Annotate Image Error",
            f"{e}\n\n{traceback.format_exc()}"
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())

