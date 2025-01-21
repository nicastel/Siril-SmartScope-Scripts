# -*- coding: UTF-8 -*-
# ==============================================================================

# ------------------------------------------------------------------------------
# Project: Simple python siril script to run CosmicClarity Sharpen
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

import sys
import os
import sys
from sirilpy.connection import SirilInterface

# SET THIS PATH ! :
PATH = "/Users/nicolas.castel/Documents/Dev/CosmicClaritySuite_macos"

# default is linux
EXE = "SetiAstroCosmicClarity"
if sys.platform == "darwin":
    # MAC OS X
    EXE = "SetiAstroCosmicClarityMac"
elif os.name == "nt":
    # Windows, Cygwin, etc. (either 32-bit or 64-bit)
    EXE = "setiastrocosmicclarity.exe"

print("CosmicClarity:begin") 
siril = SirilInterface()

try:
    siril.connect()
    
    WD = siril.get_siril_wd()
    
    FILE = os.path.basename(siril.get_image_filename())
    
    siril.cmd("savetif", os.path.join(PATH, "input", FILE) , "-astro")
    
    print(os.popen(os.path.join(PATH,EXE)).read())
    
    siril.cmd("load",  os.path.join(PATH, "output", FILE + "_sharpened.tif"))
    
    os.remove(os.path.join(PATH, "input", FILE + ".tif"))
    os.remove(os.path.join(PATH, "output", FILE + "_sharpened.tif"))
    
except Exception as e :
    print("\n**** ERROR *** " +  str(e) + "\n" )    

siril.disconnect()
del siril
print("CosmicClarity:end")

       

