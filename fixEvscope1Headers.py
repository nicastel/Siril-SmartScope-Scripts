# -*- coding: UTF-8 -*-
# ==============================================================================

# ------------------------------------------------------------------------------
# Project: Python siril script to fix evscope1 fits header
#
# ------------------------------------------------------------------------------
#    Author:  Nicolas CASTEL <nic.castel (at) gmail.com>
#
# This program is provided without any guarantee.
#
# The license is  LGPL-v3
# For details, see GNU General Public License, version 3 or later.
#                        "https://www.gnu.org/licenses/gpl.html"
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------
import os

import sirilpy as s

s.ensure_installed("astropy")
from astropy.io import fits

siril = s.SirilInterface()
try:
    siril.connect()
    print("Connected successfully!")
except SirilConnectionError as e:
    print(f"Connection failed: {e}")

for file in os.listdir(siril.get_siril_wd()):
    if file.endswith(".fits") or file.endswith(".fit"):
        data, hdr = fits.getdata(file, header=True)
        hdr.set("RA", hdr["FOVRA"])  # add a RA header
        hdr.set("DEC", hdr["FOVDEC"])  # add a DEC header
        hdr.set("FOCALLEN", 450.0)  # add a FOCALLEN header
        hdr.set("XPIXSZ", 3.75)  # add a XPIXSZ header
        hdr.set("YPIXSZ", 3.75)  # add a YPIXSZ header
        if hdr["SOFTVER"].startswith("4.2"):
            hdr.set("XBAYROFF", 0)  # add a XPIXSZ header
            hdr.set("YBAYROFF", 1)  # add a YPIXSZ header
        fits.writeto(file, data, hdr, overwrite=True)
        print(file)
