##############################################
# VeraLux — Alchemy
# Linear-Phase Narrowband Normalization & Mixing
# Author: Riccardo Paterniti (2025)
# Contact: info@veralux.space
##############################################

# (c) 2025 Riccardo Paterniti
# VeraLux — Alchemy
# SPDX-License-Identifier: GPL-3.0-or-later
# Version 1.0.3
#
# Credits / Origin
# ----------------
#   • Architecture: VeraLux Shared GUI Framework
#   • Math basis: Robust Linear Fitting & Matrix Mixing
#   • Quantum Unmixing: DBXtract-derived Ha/OIII crosstalk compensation
#   • Visualization: Siril AutoStretch (MTF) Engine (Ported from C)
#

"""
Overview
--------
VeraLux Alchemy is a linear-phase workstation designed to normalize, unmix,
and mix narrowband signals (e.g., from OSC Dual-Band filters) prior to stretching.

It addresses the common issue of red-dominated HOO/SHO composites by:
• statistically aligning weak signals (OIII/SII) to the strong signal (H-alpha)
• optionally separating Ha and OIII using a sensor-based crosstalk compensation model
• providing a real-time linear mixing matrix to define the final color palette

All operations are performed strictly in the linear domain.

Key Features
------------
• **Linear-Phase Processing**  
  All operations occur on linear data. Output is linear and stretch-ready.

• **Robust Normalization**  
  Aligns background (Median) and signal strength (MAD-based) of weak channels
  to the reference channel using robust statistics.

• **Quantum Unmixing (Optional)**  
  Dual-band Ha/OIII separation using sensor-specific crosstalk compensation
  coefficients derived from DBXtract. This is a physical signal model, not a correction.

• **Real-Time Palette Mixer**  
  Define HOO, pseudo-SHO, or custom blends interactively.

• **WYSIWYG Preview**  
  Uses Siril's AutoStretch (MTF) engine to visualize the linear result exactly
  as it will appear after stretching.

Usage
-----
1. **Load Data**  
   Import a Linear RGB image (e.g., OSC dual-band result).

2. **(Optional) Quantum Unmixing**  
   Select the sensor profile and enable Quantum Unmixing to separate Ha and OIII
   before normalization.

3. **Normalize**  
   Use Auto Signal Fit (MAD) and/or OIII Boost to align weak signals to H-alpha.
   Background Neutralization aligns black points across channels.

4. **Mix**  
   Use the R, G, B sliders to define the final palette.

5. **Process**  
   Click PROCESS to generate a LINEAR (dark) output file.
   The result is intended to be stretched with VeraLux HMS.

Inputs & Outputs
----------------
Input:  Linear RGB FITS (32-bit floating point).
Output: Balanced / Mixed Linear RGB FITS (32-bit floating point).

Compatibility
-------------
• Siril 1.3+
• Python 3.10+ (via sirilpy)
• Dependencies: sirilpy, PyQt6, numpy, astropy

License
-------
Released under GPL-3.0-or-later.
"""

import sys
import os
import math
import webbrowser

try:
    import sirilpy as s
    from sirilpy import LogColor
except ImportError:
    print("Error: sirilpy module not found.")
    sys.exit(1)

s.ensure_installed("numpy", "astropy", "PyQt6")

import numpy as np
from astropy.io import fits

from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QGridLayout,
                             QWidget, QLabel, QSlider, QPushButton, QGroupBox,
                             QMessageBox, QProgressBar, QComboBox, QCheckBox,
                             QFileDialog, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
                             QButtonGroup, QRadioButton)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QEvent, QSettings
from PyQt6.QtGui import QImage, QPixmap, QPainter, QColor, QPen, QBrush

# ---------------------
#  THEME & STYLING
# ---------------------
DARK_STYLESHEET = """
QWidget { background-color: #2b2b2b; color: #e0e0e0; font-size: 10pt; }
QToolTip { background-color: #333333; color: #ffffff; border: 1px solid #88aaff; }
QGroupBox { border: 1px solid #444444; margin-top: 5px; font-weight: bold; border-radius: 4px; padding-top: 12px; }
QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 3px; color: #88aaff; }
QLabel { color: #cccccc; }

QCheckBox, QRadioButton { spacing: 5px; color: #cccccc; }
QCheckBox::indicator, QRadioButton::indicator { width: 14px; height: 14px; border: 1px solid #666666; background: #3c3c3c; border-radius: 4px; }
QCheckBox::indicator:checked, QRadioButton::indicator:checked { background-color: #285299; border: 1px solid #88aaff; }

/* Sliders */
QSlider { min-height: 24px; }
QSlider::groove:horizontal { background: #444444; height: 6px; border-radius: 3px; }
QSlider::sub-page:horizontal { background: transparent; }
QSlider::add-page:horizontal { background: transparent; }
QSlider::handle:horizontal { 
    background-color: #cccccc; border: 1px solid #666666; 
    width: 14px; height: 14px; margin: -4px 0; border-radius: 7px; 
}
QSlider::handle:horizontal:hover { background-color: #ffffff; border-color: #88aaff; }
QSlider::handle:horizontal:pressed { background-color: #ffffff; border-color: #ffffff; }

/* Normalization Slider (Cyan) */
QSlider#NormSlider::handle:horizontal { background-color: #00cccc; border: 1px solid #008888; }
QSlider#NormSlider::handle:horizontal:hover { background-color: #00ffff; border-color: #ffffff; }

/* Channel Sliders (RGB) */
QSlider#SliderR::handle:horizontal { background-color: #ff6666; border: 1px solid #cc0000; }
QSlider#SliderG::handle:horizontal { background-color: #66ff66; border: 1px solid #00cc00; }
QSlider#SliderB::handle:horizontal { background-color: #6666ff; border: 1px solid #0000cc; }

QPushButton { background-color: #444444; color: #dddddd; border: 1px solid #666666; border-radius: 4px; padding: 6px; font-weight: bold;}
QPushButton:hover { background-color: #555555; border-color: #777777; }
QPushButton#ProcessButton { background-color: #285299; border: 1px solid #1e3f7a; }
QPushButton#ProcessButton:hover { background-color: #355ea1; }
QPushButton#CloseButton { background-color: #5a2a2a; border: 1px solid #804040; }
QPushButton#CloseButton:hover { background-color: #7a3a3a; }
QPushButton#ZoomBtn { min-width: 30px; font-weight: bold; background-color: #3c3c3c; }

/* GHOST HELP BUTTON */
QPushButton#HelpButton { 
    background-color: transparent; 
    color: #555555; 
    border: none; 
    font-weight: bold; 
    min-width: 20px;
}
QPushButton#HelpButton:hover { 
    color: #aaaaaa; 
}

QPushButton#CoffeeButton { 
    background-color: transparent; 
    border: none; 
    font-size: 15pt; 
    padding: 2px; 
    margin-right: 2px;
}
QPushButton#CoffeeButton:hover { 
    background-color: rgba(255, 255, 255, 20); 
    border-radius: 4px; 
}

QComboBox { background-color: #3c3c3c; color: #ffffff; border: 1px solid #555555; padding: 3px; border-radius: 3px; }
"""

VERSION = "HxO_1.0"
# ------------------------------------------------------------------------------
# VERSION HISTORY
# ------------------------------------------------------------------------------
# 1.0.3: "Buy me a coffee" button added.
# 1.0.2: Metadata Preservation Fix. Switched output generation logic to use Siril's 
#        native I/O pipeline. Now preserves FITS Headers/WCS of the source image.
# 1.0.1: astropy import fix.
# ------------------------------------------------------------------------------

# =============================================================================
#  QUANTUM UNMIXING (Dual-band OSC Ha/OIII separation)
#  Coefficients derived from Siril Sensor Database (Linear Interpolation)
#  Wavelengths: OIII=500.7nm | Ha=656.3nm | SII=671.6nm
#  Keys: 
#  r1: OIII->R (Crosstalk) | r2: Ha->R (Signal)    | r3: SII->R
#  g1: OIII->G (Signal)    | g2: Ha->G (Crosstalk) | g3: SII->G
#  b1: OIII->B (Signal)    | b2: Ha->B (Crosstalk) | b3: SII->B
# =============================================================================

QUANTUM_COEFFS = {
    "Generic OSC": {
        "r1": 0.000, "r2": 1.000, "r3": 1.000, 
        "g1": 1.000, "g2": 0.000, "g3": 0.000, 
        "b1": 1.000, "b2": 0.000, "b3": 0.000
    },
    
    # --- Sony IMX Sensors (Dedicated Astro) ---
    "Sony IMX071": {"r1": 0.031, "r2": 0.776, "r3": 0.697, "g1": 0.730, "g2": 0.106, "g3": 0.090, "b1": 0.518, "b2": 0.033, "b3": 0.027},
    "Sony IMX178": {"r1": 0.024, "r2": 0.354, "r3": 0.111, "g1": 0.680, "g2": 0.053, "g3": 0.012, "b1": 0.370, "b2": 0.022, "b3": 0.006},
    "Sony IMX183": {"r1": 0.038, "r2": 0.665, "r3": 0.627, "g1": 0.722, "g2": 0.162, "g3": 0.122, "b1": 0.505, "b2": 0.075, "b3": 0.058},
    "Sony IMX193": {"r1": 0.041, "r2": 0.658, "r3": 0.672, "g1": 0.792, "g2": 0.134, "g3": 0.116, "b1": 0.395, "b2": 0.016, "b3": 0.013},
    "Sony IMX224": {"r1": 0.050, "r2": 0.656, "r3": 0.624, "g1": 0.812, "g2": 0.139, "g3": 0.112, "b1": 0.504, "b2": 0.033, "b3": 0.022},
    "Sony IMX269": {"r1": 0.037, "r2": 0.669, "r3": 0.579, "g1": 0.835, "g2": 0.125, "g3": 0.091, "b1": 0.577, "b2": 0.020, "b3": 0.013},
    "Sony IMX294": {"r1": 0.031, "r2": 0.658, "r3": 0.686, "g1": 0.902, "g2": 0.149, "g3": 0.166, "b1": 0.501, "b2": 0.052, "b3": 0.058},
    "Sony IMX385": {"r1": 0.054, "r2": 0.945, "r3": 0.871, "g1": 0.842, "g2": 0.476, "g3": 0.417, "b1": 0.518, "b2": 0.082, "b3": 0.076},
    "Sony IMX410": {"r1": 0.045, "r2": 0.658, "r3": 0.620, "g1": 0.803, "g2": 0.142, "g3": 0.119, "b1": 0.501, "b2": 0.030, "b3": 0.021},
    "Sony IMX415": {"r1": 0.077, "r2": 0.873, "r3": 0.811, "g1": 0.951, "g2": 0.283, "g3": 0.281, "b1": 0.771, "b2": 0.127, "b3": 0.119},
    "Sony IMX455": {"r1": 0.033, "r2": 0.651, "r3": 0.590, "g1": 0.672, "g2": 0.063, "g3": 0.081, "b1": 0.407, "b2": 0.018, "b3": 0.035},
    "Sony IMX462": {"r1": 0.043, "r2": 0.697, "r3": 0.825, "g1": 0.840, "g2": 0.321, "g3": 0.510, "b1": 0.490, "b2": 0.158, "b3": 0.315},
    "Sony IMX477": {"r1": 0.079, "r2": 0.741, "r3": 0.718, "g1": 0.970, "g2": 0.134, "g3": 0.108, "b1": 0.497, "b2": 0.040, "b3": 0.035},
    "Sony IMX482": {"r1": 0.038, "r2": 0.658, "r3": 0.686, "g1": 0.902, "g2": 0.149, "g3": 0.166, "b1": 0.501, "b2": 0.052, "b3": 0.058},
    "Sony IMX533": {"r1": 0.029, "r2": 0.803, "r3": 0.743, "g1": 0.893, "g2": 0.161, "g3": 0.176, "b1": 0.504, "b2": 0.051, "b3": 0.076},
    "Sony IMX571": {"r1": 0.023, "r2": 0.822, "r3": 0.757, "g1": 0.852, "g2": 0.083, "g3": 0.082, "b1": 0.501, "b2": 0.022, "b3": 0.035},
    "Sony IMX585": {"r1": 0.075, "r2": 0.983, "r3": 0.966, "g1": 0.835, "g2": 0.198, "g3": 0.252, "b1": 0.435, "b2": 0.052, "b3": 0.079},
    "Sony IMX662": {"r1": 0.043, "r2": 0.768, "r3": 0.840, "g1": 0.884, "g2": 0.286, "g3": 0.457, "b1": 0.493, "b2": 0.080, "b3": 0.139},
    "Sony IMX676": {"r1": 0.063, "r2": 0.648, "r3": 0.612, "g1": 0.865, "g2": 0.126, "g3": 0.103, "b1": 0.491, "b2": 0.038, "b3": 0.031},
    "Sony IMX678": {"r1": 0.067, "r2": 0.609, "r3": 0.611, "g1": 0.916, "g2": 0.150, "g3": 0.128, "b1": 0.494, "b2": 0.037, "b3": 0.031},
    "Sony IMX715": {"r1": 0.072, "r2": 0.665, "r3": 0.672, "g1": 0.871, "g2": 0.136, "g3": 0.124, "b1": 0.502, "b2": 0.043, "b3": 0.035},

    # --- Canon DSLR (Unmodified filters block mostly Ha) ---
    "Canon EOS 1D Mark III": {"r1": 0.010, "r2": 0.231, "r3": 0.147, "g1": 0.947, "g2": 0.034, "g3": 0.007, "b1": 0.679, "b2": 0.001, "b3": 0.001},
    "Canon EOS 20D": {"r1": 0.014, "r2": 0.244, "r3": 0.131, "g1": 0.845, "g2": 0.043, "g3": 0.024, "b1": 0.513, "b2": 0.002, "b3": 0.003},
    "Canon EOS 300D": {"r1": 0.008, "r2": 0.232, "r3": 0.063, "g1": 0.702, "g2": 0.012, "g3": 0.010, "b1": 0.485, "b2": 0.001, "b3": 0.001},
    "Canon EOS 40D": {"r1": 0.020, "r2": 0.224, "r3": 0.134, "g1": 0.916, "g2": 0.022, "g3": 0.012, "b1": 0.536, "b2": 0.004, "b3": 0.006},
    "Canon EOS 500D": {"r1": 0.081, "r2": 0.247, "r3": 0.128, "g1": 0.835, "g2": 0.038, "g3": 0.028, "b1": 0.577, "b2": 0.002, "b3": 0.000},
    "Canon EOS 50D": {"r1": 0.080, "r2": 0.231, "r3": 0.117, "g1": 0.843, "g2": 0.043, "g3": 0.024, "b1": 0.565, "b2": 0.004, "b3": 0.002},
    "Canon EOS 600D": {"r1": 0.035, "r2": 0.187, "r3": 0.125, "g1": 0.825, "g2": 0.024, "g3": 0.018, "b1": 0.521, "b2": 0.002, "b3": 0.001},
    "Canon EOS 60D": {"r1": 0.035, "r2": 0.212, "r3": 0.126, "g1": 0.819, "g2": 0.023, "g3": 0.017, "b1": 0.535, "b2": 0.002, "b3": 0.001},

    # --- Nikon DSLR (Unmodified) ---
    "Nikon D200": {"r1": 0.040, "r2": 0.219, "r3": 0.062, "g1": 0.556, "g2": 0.009, "g3": 0.005, "b1": 0.505, "b2": 0.001, "b3": 0.000},
    "Nikon D3": {"r1": 0.040, "r2": 0.193, "r3": 0.063, "g1": 0.574, "g2": 0.008, "g3": 0.005, "b1": 0.506, "b2": 0.001, "b3": 0.000},
    "Nikon D3X": {"r1": 0.029, "r2": 0.201, "r3": 0.120, "g1": 0.649, "g2": 0.013, "g3": 0.009, "b1": 0.523, "b2": 0.001, "b3": 0.001},
    "Nikon D300s": {"r1": 0.049, "r2": 0.222, "r3": 0.039, "g1": 0.533, "g2": 0.011, "g3": 0.008, "b1": 0.537, "b2": 0.003, "b3": 0.005},
    "Nikon D40": {"r1": 0.020, "r2": 0.133, "r3": 0.108, "g1": 0.560, "g2": 0.003, "g3": 0.003, "b1": 0.722, "b2": 0.001, "b3": 0.001},
    "Nikon D50": {"r1": 0.019, "r2": 0.158, "r3": 0.085, "g1": 0.524, "g2": 0.003, "g3": 0.004, "b1": 0.381, "b2": 0.001, "b3": 0.002},
    "Nikon D5100": {"r1": 0.044, "r2": 0.177, "r3": 0.078, "g1": 0.661, "g2": 0.018, "g3": 0.011, "b1": 0.521, "b2": 0.003, "b3": 0.004},
    "Nikon D700": {"r1": 0.040, "r2": 0.198, "r3": 0.074, "g1": 0.589, "g2": 0.007, "g3": 0.004, "b1": 0.505, "b2": 0.001, "b3": 0.001},
    "Nikon D7200": {"r1": 0.073, "r2": 0.093, "r3": 0.074, "g1": 0.532, "g2": 0.010, "g3": 0.011, "b1": 0.518, "b2": 0.006, "b3": 0.009},
    "Nikon D80": {"r1": 0.022, "r2": 0.179, "r3": 0.048, "g1": 0.509, "g2": 0.007, "g3": 0.005, "b1": 0.490, "b2": 0.001, "b3": 0.001},
    "Nikon D90": {"r1": 0.044, "r2": 0.240, "r3": 0.059, "g1": 0.547, "g2": 0.013, "g3": 0.011, "b1": 0.511, "b2": 0.002, "b3": 0.002},

    # --- Other / Smart Telescopes ---
    "Fujifilm X-Trans 5 HR": {"r1": 0.051, "r2": 0.049, "r3": 0.047, "g1": 0.413, "g2": 0.697, "g3": 0.724, "b1": 0.377, "b2": 0.650, "b3": 0.670},
    "Samsung ISOCELL": {"r1": 0.144, "r2": 0.665, "r3": 0.675, "g1": 0.499, "g2": 0.082, "g3": 0.063, "b1": 0.353, "b2": 0.055, "b3": 0.071},
    "ZWO Seestar S30": {"r1": 0.063, "r2": 0.648, "r3": 0.612, "g1": 0.865, "g2": 0.126, "g3": 0.103, "b1": 0.491, "b2": 0.038, "b3": 0.031},
    "ZWO Seestar S50": {"r1": 0.024, "r2": 0.822, "r3": 0.757, "g1": 0.852, "g2": 0.083, "g3": 0.082, "b1": 0.501, "b2": 0.022, "b3": 0.035}
}

# =============================================================================
#  CORE MATH: LINEAR ALGEBRA
# =============================================================================

class VeraLuxNBCore:
    @staticmethod
    def normalize_input(img_data):
        """Ensures float32 0-1 range."""
        input_dtype = img_data.dtype
        img_float = img_data.astype(np.float32)
        if np.issubdtype(input_dtype, np.integer):
            if input_dtype == np.uint8: return img_float / 255.0
            elif input_dtype == np.uint16: return img_float / 65535.0
            else: return img_float / float(np.iinfo(input_dtype).max)
        elif np.issubdtype(input_dtype, np.floating):
            current_max = np.max(img_data)
            if current_max <= 1.0 + 1e-5: return img_float
            if current_max <= 65535.0: return img_float / 65535.0
            return img_float
        return img_float

    @staticmethod
    def calc_stats(channel):
        """Robust statistics: Median (Background) and Signal Strength."""
        # Use stride for speed
        stride = max(1, channel.size // 1000000)
        sample = channel.flatten()[::stride]
        
        median = np.median(sample)
        # MAD for noise estimation
        mad = np.median(np.abs(sample - median))
        
        # Signal Strength: 99.5th percentile (approx peak of stars/nebula)
        # Subtract median to get pure signal amplitude
        peak = np.percentile(sample, 99.5)
        signal_strength = max(1e-9, peak - median)
        
        return median, mad, signal_strength

    @staticmethod
    def linear_fit_channels(rgb_img, align_bg=True, auto_gain=True, manual_boost=1.0):
        """
        Aligns G and B channels to R (Reference).
        """
        R = rgb_img[0]
        G = rgb_img[1]
        B = rgb_img[2]
        
        med_r, mad_r, str_r = VeraLuxNBCore.calc_stats(R)
        med_g, mad_g, str_g = VeraLuxNBCore.calc_stats(G)
        med_b, mad_b, str_b = VeraLuxNBCore.calc_stats(B)
        
        # 1. Background Alignment (Offset)
        if align_bg:
            G = G - med_g + med_r
            B = B - med_b + med_r
        
        # 2. Linear Gain Match
        gain_g = 1.0
        gain_b = 1.0
        
        if auto_gain:
            gain_g = str_r / str_g
            gain_b = str_r / str_b
            
        # Apply Manual Boost (multiplies the weak channels)
        gain_g *= manual_boost
        gain_b *= manual_boost
        
        # Apply Gain (around the black point/median to preserve background)
        G = (G - med_r) * gain_g + med_r
        B = (B - med_r) * gain_b + med_r
        
        # Clip to safe range
        G = np.clip(G, 0.0, 1.0)
        B = np.clip(B, 0.0, 1.0)
        
        return np.stack([R, G, B])

    @staticmethod
    def _quantum_unmix_ha_oiii(norm_rgb, coef):
        """Dual-band OSC unmixing: estimate HA and OIII from RGB with overlap coefficients.
        Follows DBXtract's algebra (Ha/OIII only) with robust median background handling.
        """
        r, g, b = norm_rgb[0], norm_rgb[1], norm_rgb[2]

        # Robust per-channel backgrounds (median)
        bg_r = float(np.median(r)); bg_g = float(np.median(g)); bg_b = float(np.median(b))
        r0 = r - bg_r; g0 = g - bg_g; b0 = b - bg_b

        # Guard against invalid coefficients
        r2 = float(coef.get("r2", 1.0))
        r1 = float(coef.get("r1", 0.0))
        g1 = float(coef.get("g1", 1.0)); g2 = float(coef.get("g2", 0.0))
        b1 = float(coef.get("b1", 1.0)); b2 = float(coef.get("b2", 0.0))

        eps = 1e-8
        if abs(r2) < eps:
            # Degenerate: fallback to classic mapping
            Ha = r
            OIII = (g + b) * 0.5
            return Ha, OIII

        # Crosstalk suppression clamp used by DBXtract
        cota = min(g2 / r2, 0.12)

        den_g = (g1 - g2 * r1 / r2)
        den_b = (b1 - b2 * r1 / r2)
        if abs(den_g) < eps or abs(den_b) < eps:
            Ha = r
            OIII = (g + b) * 0.5
            return Ha, OIII

        OIII_G = (g0 - cota * r0) / den_g
        OIII_B = (b0 - (b2 * r0 / r2)) / den_b

        bg_gb = max(bg_b, bg_g)

        OIII = ((2.0 * g1 * OIII_G) + (b1 * OIII_B)) / (2.0 * g1 + b1 + eps) + bg_gb
        HA = (r0 - r1 * (OIII - bg_gb)) / (r2 + eps) + (bg_r + bg_gb)

        HA = np.clip(HA, 0.0, 1.0)
        OIII = np.clip(OIII, 0.0, 1.0)
        return HA, OIII

    @staticmethod
    def mix_channels(norm_rgb, mix_r, mix_g, mix_b, quantum_unmix=False, sensor_profile="Generic OSC"):
        # (faux SHO) using the formula
        # ((Oiii*Ha)^~(Oiii*Ha))*Ha + ~((Oiii*Ha)^~(Oiii*Ha))*Oiii
        # from
        # https://thecoldestnights.com/2020/06/pixinsight-dynamic-narrowband-combinations-with-pixelmath/?fbclid=IwAR1_YtLbAGGBoL-N-I7E8vIqsgB4o_mOwOSr-5F7RpXY4zrQzfmsnzB4lYU

        """
        Mixes channels based on Ha/OIII contribution.

        If quantum_unmix is enabled, Ha/OIII are first separated using overlap coefficients
        (dual-band crosstalk compensation), then the palette mixer is applied.
        """
        if quantum_unmix:
            coef = QUANTUM_COEFFS.get(sensor_profile, QUANTUM_COEFFS.get("Generic OSC"))
            Ha, OIII = VeraLuxNBCore._quantum_unmix_ha_oiii(norm_rgb, coef)
        else:
            Ha = norm_rgb[0]
            OIII = norm_rgb[1]

        R_out = Ha * (1.0 - mix_r) + OIII * mix_r

        # ((Oiii*Ha)^~(Oiii*Ha))*Ha + ~((Oiii*Ha)^~(Oiii*Ha))*Oiii
        O = (OIII*65535.0).astype(np.uint16) 
        H = (Ha*65535.0).astype(np.uint16) 
        fake1 = (np.bitwise_xor((O*H),~(O*H))).astype(np.float32) / 65535.0
        fake2 = (~(np.bitwise_xor((O*H),~(O*H)))).astype(np.float32) / 65535.0
        G_out = fake1 * H * (1.0 - mix_g) +  fake2 * O * mix_g
        #G_out = (fake).astype(np.float32) / 65535.0

        G_out = Ha * (1.0 - mix_g) + OIII * mix_g

        B_out = Ha * (1.0 - mix_b) + OIII * mix_b



        # invert green and blue
        return np.stack([R_out, G_out, B_out])

# =============================================================================
#  VISUAL ENGINE (Siril AutoStretch Exact)
# =============================================================================

def MTF(x, m, lo, hi):
    m = float(m); lo = float(lo); hi = float(hi)
    dist = hi - lo
    if dist < 1e-9: return np.where(x > lo, 1.0, 0.0).astype(np.float32)
    xp = (x - lo) / dist
    xp = np.clip(xp, 0.0, 1.0)
    num = (m - 1.0) * xp
    den = (2.0 * m - 1.0) * xp - m
    with np.errstate(divide='ignore', invalid='ignore'): y = num / den
    return np.nan_to_num(y, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)

def find_linked_params_siril(img_rgb):
    """
    Replicates 'find_linked_midtones_balance' from Siril.
    """
    MAD_NORM = 1.4826
    SHADOWS_CLIPPING = -2.8
    TARGET_BG = 0.25
    
    nb_channels = 3
    sum_c0 = 0.0
    sum_m = 0.0
    
    for i in range(nb_channels):
        ch = img_rgb[i]
        stride = max(1, ch.size // 500000)
        sample = ch.flatten()[::stride]
        median = float(np.median(sample))
        diff = np.abs(sample - median)
        mad = float(np.median(diff)) * MAD_NORM
        if mad == 0.0: mad = 0.001
        sum_c0 += median + (SHADOWS_CLIPPING * mad)
        sum_m += median

    c0 = sum_c0 / float(nb_channels)
    if c0 < 0.0: c0 = 0.0
    m_avg = sum_m / float(nb_channels)
    m2 = m_avg - c0
    midtones = MTF(m2, TARGET_BG, 0.0, 1.0)
    
    return c0, midtones, 1.0

def apply_siril_autostretch(img_rgb):
    """Applies the Linked AutoStretch for preview."""
    shadows, midtones, highlights = find_linked_params_siril(img_rgb)
    out = np.zeros_like(img_rgb)
    for i in range(3):
        out[i] = MTF(img_rgb[i], midtones, shadows, highlights)
    return np.clip(out, 0.0, 1.0)

# =============================================================================
#  GUI HELPERS
# =============================================================================


class ResetSlider(QSlider):
    def __init__(self, orientation, default_val=0, parent=None):
        super().__init__(orientation, parent)
        self.default_val = default_val
    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.setValue(self.default_val); event.accept()
        else: super().mouseDoubleClickEvent(event)

# --- FitGraphicsView for double-click fit ---
class FitGraphicsView(QGraphicsView):
    def __init__(self, scene, on_double_click=None, parent=None):
        super().__init__(scene, parent)
        self._on_double_click = on_double_click

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and callable(self._on_double_click):
            self._on_double_click()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

# =============================================================================
#  WORKER THREAD
# =============================================================================

class NormalizationWorker(QThread):
    result_ready = pyqtSignal(object, object)
    
    def __init__(self, img_input, params):
        super().__init__()
        self.img = img_input
        self.p = params
        
    def run(self):
        # IMPORTANT:
        # Quantum Unmixing MUST be applied on the original OSC channel ratios.
        # Therefore, when enabled, we unmix first, then normalize the extracted OIII vs Ha.
        if self.p.get('quantum_unmix', False):
            coef = QUANTUM_COEFFS.get(self.p.get('sensor_profile', 'Generic OSC'), QUANTUM_COEFFS.get('Generic OSC'))
            Ha, OIII = VeraLuxNBCore._quantum_unmix_ha_oiii(self.img, coef)
            base_rgb = np.stack([Ha, OIII, OIII])
        else:
            base_rgb = self.img

        norm_rgb = VeraLuxNBCore.linear_fit_channels(
            base_rgb,
            align_bg=self.p['bg_align'],
            auto_gain=self.p['auto_fit'],
            manual_boost=self.p['boost']
        )

        # After unmixing, we already have Ha in R and OIII in G/B, so mix in classic mode.
        linear_out = VeraLuxNBCore.mix_channels(
            norm_rgb,
            self.p['mix_r'],
            self.p['mix_g'],
            self.p['mix_b'],
            quantum_unmix=False,
            sensor_profile=self.p.get('sensor_profile', 'Generic OSC')
        )
        preview = apply_siril_autostretch(linear_out)
        self.result_ready.emit(linear_out, preview)

# =============================================================================
#  MAIN WINDOW
# =============================================================================

class AlchemyGUI(QMainWindow):
    def __init__(self, siril, app):
        super().__init__()
        self.siril = siril
        self.app = app
        self.setWindowTitle(f"VeraLux Alchemy v{VERSION}")
        self.setStyleSheet(DARK_STYLESHEET)
        self.resize(1300, 600)
        self.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint)
        
        self.settings = QSettings("VeraLux", "Alchemy")
        
        self.img_full = None
        self.img_proxy = None
        
        self.debounce = QTimer()
        self.debounce.setSingleShot(True); self.debounce.setInterval(150)
        self.debounce.timeout.connect(self.run_worker)
        
        # Header Log
        header_msg = (
            "\n##############################################\n"
            "# VeraLux — Alchemy\n"
            "# Linear-Phase Narrowband Normalization & Mixing\n"
            "# Author: Riccardo Paterniti (2025)\n"
            "##############################################"
        )
        try: self.siril.log(header_msg)
        except: print(header_msg)
        
        self.init_ui()
        self.cache_input() # Auto-load on startup
        
    def init_ui(self):
        main = QWidget(); self.setCentralWidget(main)
        # Main Layout (Horizontal split)
        main_layout = QHBoxLayout(main) 
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # --- LEFT PANEL ---
        left_container = QWidget(); left_container.setFixedWidth(380)
        left = QVBoxLayout(left_container); left.setContentsMargins(0,0,0,0)
        
        # --- HEADER (Inside Left Column) ---
        lbl_title = QLabel("VeraLux Alchemy HxO")
        lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_title.setStyleSheet("font-size: 16pt; font-weight: bold; color: #88aaff; margin-top: 5px;")
        left.addWidget(lbl_title)

        lbl_subtitle = QLabel("Linear-Phase Narrowband Normalization & Mixing")
        lbl_subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_subtitle.setStyleSheet("font-size: 10pt; color: #999999; font-style: italic; margin-bottom: 15px;")
        left.addWidget(lbl_subtitle)
        
        # 0. Sensor Profile / Quantum Unmixing
        g0 = QGroupBox("0. Sensor Profile")
        l0 = QGridLayout(g0)
        l0.setContentsMargins(12, 16, 12, 12)
        l0.setHorizontalSpacing(6)
        l0.setVerticalSpacing(8)

        lbl_sensor = QLabel("Sensor:")
        lbl_sensor.setFixedWidth(70)
        lbl_sensor.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        self.cmb_sensor = QComboBox()
        self.cmb_sensor.addItems(list(QUANTUM_COEFFS.keys()))
        self.cmb_sensor.setMinimumWidth(190)
        # Prefer "Generic OSC" as default if present
        if "Generic OSC" in QUANTUM_COEFFS:
            self.cmb_sensor.setCurrentText("Generic OSC")
        self.cmb_sensor.setToolTip(
            "Select the camera sensor profile used for\n"
            "dual-band Quantum Unmixing.\n"
            "This affects the Ha/OIII crosstalk\n"
            "separation coefficients."
        )
        self.cmb_sensor.currentIndexChanged.connect(self.trigger_update)

        l0.addWidget(lbl_sensor, 0, 0)
        l0.addWidget(self.cmb_sensor, 0, 1)
        l0.setColumnStretch(1, 1)

        self.chk_quantum = QCheckBox("Quantum Unmixing (Ha/OIII Crosstalk Compensation Model)")
        self.chk_quantum.setChecked(False)
        self.chk_quantum.setToolTip(
            "Enable dual-band Quantum Unmixing to\n"
            "separate Ha and OIII by correcting\n"
            "OSC crosstalk (DBXtract coefficients).\n"
            "\n"
            "When OFF, Alchemy uses classic mapping:\n"
            "Ha = R\n"
            "OIII = (G + B) / 2"
        )
        self.chk_quantum.stateChanged.connect(self.trigger_update)
        self.chk_quantum.setStyleSheet("margin-left: 0px;")

        l0.addWidget(self.chk_quantum, 1, 0, 1, 2)
        l0.setColumnStretch(2, 0)
        left.addWidget(g0)

        # 1. Normalization
        g1 = QGroupBox("1. Normalization (Linear Fit)"); l1 = QVBoxLayout(g1)
        
        self.chk_bg = QCheckBox("Background Neutralization")
        self.chk_bg.setChecked(True)
        self.chk_bg.setToolTip("Aligns the median (black point) of all channels to Red.")
        self.chk_bg.toggled.connect(self.trigger_update)
        l1.addWidget(self.chk_bg)
        
        self.chk_auto = QCheckBox("Auto Signal Fit (MAD)")
        self.chk_auto.setChecked(True)
        self.chk_auto.setToolTip("Automatically calculates gain to match OIII signal strength to Ha.")
        self.chk_auto.toggled.connect(self.trigger_update)
        l1.addWidget(self.chk_auto)
        
        l1.addWidget(QLabel("OIII Boost (Manual Gain):"))
        h_boost = QHBoxLayout()
        self.s_boost = ResetSlider(Qt.Orientation.Horizontal, 100); self.s_boost.setObjectName("NormSlider")
        self.s_boost.setRange(50, 500); self.s_boost.setValue(100) # 0.5x to 5.0x
        self.s_boost.valueChanged.connect(self.update_labels)
        self.s_boost.valueChanged.connect(self.trigger_update)
        self.lbl_boost = QLabel("1.00x"); self.lbl_boost.setAlignment(Qt.AlignmentFlag.AlignRight)
        h_boost.addWidget(self.s_boost); h_boost.addWidget(self.lbl_boost)
        l1.addLayout(h_boost)
        left.addWidget(g1)
        
        # 2. Palette Mixer
        g2 = QGroupBox("2. Palette Mixer (Ha <-> OIII)"); l2 = QVBoxLayout(g2)
        
        # Red Channel
        l2.addWidget(QLabel("Output RED Channel:"))
        self.s_mix_r = ResetSlider(Qt.Orientation.Horizontal, 0); self.s_mix_r.setObjectName("SliderR")
        self.s_mix_r.setRange(0, 100); self.s_mix_r.setValue(0) # 0 = Ha
        self.s_mix_r.valueChanged.connect(self.trigger_update)
        l2.addWidget(self.s_mix_r)
        self.lbl_r_info = QLabel("100% Ha | 0% OIII")
        self.lbl_r_info.setStyleSheet("color: #ff8888; font-size: 8pt;")
        l2.addWidget(self.lbl_r_info)
        self.s_mix_r.valueChanged.connect(lambda v: self.lbl_r_info.setText(f"{100-v}% Ha | {v}% OIII"))
        
        l2.addSpacing(10)
        
        # Green Channel
        l2.addWidget(QLabel("Output GREEN Channel:"))
        self.s_mix_g = ResetSlider(Qt.Orientation.Horizontal, 100); self.s_mix_g.setObjectName("SliderG")
        self.s_mix_g.setRange(0, 100); self.s_mix_g.setValue(100) # 100 = OIII (HOO standard)
        self.s_mix_g.valueChanged.connect(self.trigger_update)
        l2.addWidget(self.s_mix_g)
        self.lbl_g_info = QLabel("0% Ha | 100% OIII")
        self.lbl_g_info.setStyleSheet("color: #88ff88; font-size: 8pt;")
        l2.addWidget(self.lbl_g_info)
        self.s_mix_g.valueChanged.connect(lambda v: self.lbl_g_info.setText(f"{100-v}% Ha | {v}% OIII"))
        
        l2.addSpacing(10)
        
        # Blue Channel
        l2.addWidget(QLabel("Output BLUE Channel:"))
        self.s_mix_b = ResetSlider(Qt.Orientation.Horizontal, 100); self.s_mix_b.setObjectName("SliderB")
        self.s_mix_b.setRange(0, 100); self.s_mix_b.setValue(100) # 100 = OIII
        self.s_mix_b.valueChanged.connect(self.trigger_update)
        l2.addWidget(self.s_mix_b)
        self.lbl_b_info = QLabel("0% Ha | 100% OIII")
        self.lbl_b_info.setStyleSheet("color: #8888ff; font-size: 8pt;")
        l2.addWidget(self.lbl_b_info)
        self.s_mix_b.valueChanged.connect(lambda v: self.lbl_b_info.setText(f"{100-v}% Ha | {v}% OIII"))
        
        # Preset Buttons
        h_pre = QHBoxLayout()
        b_hoo = QPushButton("HOO"); b_hoo.clicked.connect(lambda: self.set_preset(0, 100, 100))
        b_sho = QPushButton("Pseudo-SHO"); b_sho.clicked.connect(lambda: self.set_preset(0, 50, 100))
        b_hso = QPushButton("HSO"); b_hso.clicked.connect(lambda: self.set_preset(0, 0, 100))
        h_pre.addWidget(b_hoo); h_pre.addWidget(b_sho); h_pre.addWidget(b_hso)
        l2.addLayout(h_pre)
        
        left.addWidget(g2)
        left.addStretch()
        
        # Footer
        footer = QHBoxLayout()
        
        # Help
        self.btn_help = QPushButton("?"); self.btn_help.setObjectName("HelpButton"); self.btn_help.setFixedWidth(20)
        self.btn_help.clicked.connect(self.print_help)
        footer.addWidget(self.btn_help)
        
        b_def = QPushButton("Defaults"); b_def.clicked.connect(self.set_defaults)
        footer.addWidget(b_def)
        
        b_cls = QPushButton("Close"); b_cls.setObjectName("CloseButton")
        b_cls.clicked.connect(self.close)
        footer.addWidget(b_cls)
        
        b_proc = QPushButton("PROCESS"); b_proc.setObjectName("ProcessButton")
        b_proc.clicked.connect(self.process_final)
        footer.addWidget(b_proc)
        
        left.addLayout(footer)
        main_layout.addWidget(left_container)
        
        # --- RIGHT PANEL (Preview) ---
        right = QVBoxLayout()
        
        # Toolbar
        tb = QHBoxLayout()
        b_out = QPushButton("-"); b_out.setObjectName("ZoomBtn"); b_out.clicked.connect(self.zoom_out)
        b_fit = QPushButton("Fit"); b_fit.setObjectName("ZoomBtn"); b_fit.clicked.connect(self.fit_view)
        b_11 = QPushButton("1:1"); b_11.setObjectName("ZoomBtn"); b_11.clicked.connect(self.zoom_1to1)
        b_in = QPushButton("+"); b_in.setObjectName("ZoomBtn"); b_in.clicked.connect(self.zoom_in)
        lbl_hint = QLabel("Preview: AutoStretch (Linear data is saved) / Double-click to fit")
        lbl_hint.setStyleSheet("color: #ffb000; font-size: 8pt; font-style: italic; margin-left: 10px;")
        
        self.btn_coffee = QPushButton("☕")
        self.btn_coffee.setObjectName("CoffeeButton")
        self.btn_coffee.setToolTip("Support the developer - Buy me a coffee!")
        self.btn_coffee.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_coffee.clicked.connect(lambda: webbrowser.open("https://buymeacoffee.com/riccardo.paterniti"))

        self.chk_ontop = QCheckBox("On Top"); self.chk_ontop.setChecked(True)
        self.chk_ontop.toggled.connect(self.toggle_ontop)
        
        tb.addWidget(b_out); tb.addWidget(b_fit); tb.addWidget(b_11); tb.addWidget(b_in); tb.addWidget(lbl_hint)
        tb.addStretch(); tb.addWidget(self.btn_coffee); tb.addWidget(self.chk_ontop)
        right.addLayout(tb)
        
        # View
        self.scene = QGraphicsScene()
        self.view = FitGraphicsView(self.scene, on_double_click=self.fit_view)
        self.view.setStyleSheet("background-color: #151515; border: none;")
        self.view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        right.addWidget(self.view)

        self.pix_item = QGraphicsPixmapItem(); self.scene.addItem(self.pix_item)
        main_layout.addLayout(right)

    # --- LOGIC ---
    def cache_input(self):
        try:
            if not self.siril.connected: self.siril.connect()
            with self.siril.image_lock():
                img = self.siril.get_image_pixeldata()
            if img is None: return
            
            img = VeraLuxNBCore.normalize_input(img)
            if img.ndim == 2: img = np.array([img, img, img]) # Mono to RGB
            
            self.img_full = img
            
            # Make Proxy (Max 2048px)
            h, w = img.shape[1], img.shape[2]
            scale = 2048 / max(h, w)
            if scale < 1.0:
                step = int(1/scale)
                self.img_proxy = img[:, ::step, ::step].copy()
            else:
                self.img_proxy = img.copy()
            
            self.trigger_update()
            
        except Exception as e:
            print(f"Input Error: {e}")

    def update_labels(self):
        val = self.s_boost.value() / 100.0
        self.lbl_boost.setText(f"{val:.2f}x")

    def set_preset(self, r, g, b):
        self.s_mix_r.setValue(r)
        self.s_mix_g.setValue(g)
        self.s_mix_b.setValue(b)

    def set_defaults(self):
        self.s_boost.setValue(100); self.set_preset(0, 100, 100)
        self.chk_bg.setChecked(True); self.chk_auto.setChecked(True)

    def trigger_update(self):
        if self.img_proxy is None: return
        self.debounce.start()

    def run_worker(self):
        # Params
        p = {
            'bg_align': self.chk_bg.isChecked(),
            'auto_fit': self.chk_auto.isChecked(),
            'boost': self.s_boost.value() / 100.0,
            'mix_r': self.s_mix_r.value() / 100.0,
            'mix_g': self.s_mix_g.value() / 100.0,
            'mix_b': self.s_mix_b.value() / 100.0,
            'quantum_unmix': self.chk_quantum.isChecked(),
            'sensor_profile': self.cmb_sensor.currentText()
        }
        self.worker = NormalizationWorker(self.img_proxy, p)
        self.worker.result_ready.connect(self.update_display)
        self.worker.start()

    def update_display(self, linear, preview):
        disp = np.clip(preview * 255, 0, 255).astype(np.uint8)
        disp = np.ascontiguousarray(np.flipud(disp.transpose(1, 2, 0)))
        h, w, c = disp.shape
        qimg = QImage(disp.data.tobytes(), w, h, c*w, QImage.Format.Format_RGB888)
        self.pix_item.setPixmap(QPixmap.fromImage(qimg))
        self.scene.setSceneRect(0, 0, w, h)
        
        # Fit on first load
        if not self.view.transform().isIdentity(): pass
        else: self.fit_view()

    def process_final(self):
        if self.img_full is None: return
        self.setEnabled(False)
        
        # Gather Params
        p = {
            'bg_align': self.chk_bg.isChecked(),
            'auto_fit': self.chk_auto.isChecked(),
            'boost': self.s_boost.value() / 100.0,
            'mix_r': self.s_mix_r.value() / 100.0,
            'mix_g': self.s_mix_g.value() / 100.0,
            'mix_b': self.s_mix_b.value() / 100.0,
            'quantum_unmix': self.chk_quantum.isChecked(),
            'sensor_profile': self.cmb_sensor.currentText()
        }
        
        try:
            # 1. Build base RGB (optionally Quantum-unmixed) at full resolution
            if p.get('quantum_unmix', False):
                coef = QUANTUM_COEFFS.get(p.get('sensor_profile', 'Generic OSC'), QUANTUM_COEFFS.get('Generic OSC'))
                Ha, OIII = VeraLuxNBCore._quantum_unmix_ha_oiii(self.img_full, coef)
                base_rgb = np.stack([Ha, OIII, OIII])
            else:
                base_rgb = self.img_full

            # 2. Linear Fit Full Res
            norm_rgb = VeraLuxNBCore.linear_fit_channels(
                base_rgb, p['bg_align'], p['auto_fit'], p['boost']
            )

            # 3. Mix Full Res (classic mode; Ha in R, OIII in G/B)
            final_linear = VeraLuxNBCore.mix_channels(
                norm_rgb, p['mix_r'], p['mix_g'], p['mix_b'], quantum_unmix=False
            )
            
            # 4. Auto-save (Siril Native I/O)
            # Saves into Python's CWD (usually matches Siril's if launched from it)
            filename = "VeraLux_Alchemy_Linear.fit"
            path = os.path.join(os.getcwd(), filename)
            safe_path = path.replace(os.sep, '/')
            
            out = final_linear.astype(np.float32)
            
            # A. Inject processed pixels into the CURRENT Siril image container
            # Since Alchemy works on the open image, this container holds 
            # the correct WCS/Headers.
            with self.siril.image_lock():
                self.siril.set_image_pixeldata(out)
            
            # B. Command Siril to save
            # This triggers Siril's internal C routine which writes the 
            # preserved headers + the new pixels we just injected.
            self.siril.cmd(f'save "{safe_path}"')
            
            # C. Reload to confirm and refresh GUI state
            self.siril.cmd(f'load "{safe_path}"')
            self.siril.log(f"VeraLux Alchemy: Saved {filename} & Loaded. Ready for HMS.", LogColor.GREEN)
            self.close()
                
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
        finally:
            self.setEnabled(True)

    # --- HELP ---
    def print_help(self):
        msg = [
            "==========================================================================",
            "   VERALUX ALCHEMY v1.0 — OPERATIONAL GUIDE",
            "   Linear-Phase Narrowband Normalization & Mixing Engine",
            "==========================================================================",
            "",
            "OVERVIEW",
            "-----------------",
            "VeraLux Alchemy allows you to normalize, optionally unmix, and mix",
            "narrowband signals (from OSC Dual-Band or Mono filters) before stretching.",
            "All operations are performed strictly in the linear domain.",
            "",
            "[0] SENSOR PROFILE & QUANTUM UNMIXING (Optional)",
            "    • Sensor Profile: Selects the camera sensor model used for",
            "      Ha/OIII crosstalk compensation.",
            "    • Quantum Unmixing: Applies a physical signal model to separate",
            "      Ha and OIII in OSC dual-band data.",
            "      This is a compensation model, not a correction.",
            "",
            "[1] NORMALIZATION (Linear Fit)",
            "    • Auto Signal Fit (MAD): Automatically calculates the gain needed",
            "      to align weak signals (OIII) to the reference channel (H-alpha),",
            "      using robust statistics to ignore outliers (stars).",
            "    • Background Neutralization: Aligns the black points of all channels",
            "      to prevent color casts in the shadows.",
            "    • OIII Boost: Manual gain control for the weak signal.",
            "      Increase if OIII is faint, decrease if it introduces noise.",
            "",
            "[2] PALETTE MIXER",
            "    Use the sliders to define the composition of the final RGB image.",
            "    • Slider Left (0%): 100% H-alpha",
            "    • Slider Right (100%): 100% OIII",
            "",
            "    PRESETS:",
            "    - HOO: Classic bi-color (R=Ha, G=OIII, B=OIII).",
            "    - Pseudo-SHO: Gold/Blue look by mixing Ha and OIII in Green.",
            "",
            "[3] PREVIEW & OUTPUT",
            "    • WYSIWYG Preview: Displays Siril's AutoStretch applied to the",
            "      current linear mix.",
            "    • Output: Clicking PROCESS generates a LINEAR (dark) FITS file,",
            "      ready to be stretched with VeraLux HMS.",
            "",
            "Support & Info: info@veralux.space",
            "=========================================================================="
        ]
        try:
            for l in msg: 
                # Siril log guard: avoid sending empty strings
                txt = l if l.strip() else " "
                self.siril.log(txt)
            self.status_update("Guide printed to Console.")
        except: 
            print("\n".join(msg))

    def status_update(self, text):
        # Placeholder if status bar is added later
        pass

    # --- VIEWPORT ---
    def toggle_ontop(self, checked):
        if checked: self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)
        else: self.setWindowFlags(self.windowFlags() & ~Qt.WindowType.WindowStaysOnTopHint)
        self.show()

    def zoom_in(self): self.view.scale(1.2, 1.2)
    def zoom_out(self): self.view.scale(1/1.2, 1/1.2)
    def zoom_1to1(self): self.view.resetTransform()
    def fit_view(self): self.view.fitInView(self.pix_item, Qt.AspectRatioMode.KeepAspectRatio)

def main():
    app = QApplication(sys.argv)
    siril = s.SirilInterface()
    try: siril.connect()
    except: pass
    gui = AlchemyGUI(siril, app)
    gui.show()
    app.exec()

if __name__ == "__main__":
    main()
