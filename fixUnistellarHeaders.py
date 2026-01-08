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
    if not file.startswith(".") and ( file.endswith(".fits") or file.endswith(".fit") ):
        print("Fixing "+file)
        data, hdr = fits.getdata(file, header=True)
        if hdr.get("FOVRA") is not None:
            hdr.set(
                "RA", hdr["FOVRA"]
            )  # add a RA header based on the FOVRA unistellar header
            hdr.set(
                "DEC", hdr["FOVDEC"]
            )  # add a DEC header based on the FOVDEC unistellar header
        telescope = None
        if hdr["INSTRUME"].startswith("IMX224"):  # eVscope1 or eQuinox1
            hdr.set("FOCALLEN", 450.0)  # add a FOCALLEN header
            hdr.set("XPIXSZ", 3.75)  # add a XPIXSZ header
            hdr.set("YPIXSZ", 3.75)  # add a YPIXSZ header
            telescope = "eVscope v1.0"
        if hdr["INSTRUME"].startswith("IMX347"):  # eVscope2 or eQuinox2
            hdr.set("FOCALLEN", 450.0)  # add a FOCALLEN header
            hdr.set("XPIXSZ", 2.9)  # add a XPIXSZ header
            hdr.set("YPIXSZ", 2.9)  # add a YPIXSZ header
            telescope = "eVscope v2.0"
        if hdr["INSTRUME"].startswith("IMX415"):  # Odyssey or Odyssey Pro
            hdr.set("FOCALLEN", 320.0)  # add a FOCALLEN header
            hdr.set("XPIXSZ", 1.45)  # add a XPIXSZ header
            hdr.set("YPIXSZ", 1.45)  # add a YPIXSZ header

        if hdr.get("SOFTVER") is not None and hdr["SOFTVER"].startswith("4.2"):  # fix for bayer issue with latest FW 4.2
            hdr.set("XBAYROFF", 0)  # add a XPIXSZ header
            hdr.set("YBAYROFF", 1)  # add a YPIXSZ header
        elif telescope is not None:
            hdr.set("TELESCOP", telescope)  # add a TELESCOP header for older FW version

        fits.writeto(file, data, hdr, overwrite=True)
        print(file+" header fixed")

print("Done!")
