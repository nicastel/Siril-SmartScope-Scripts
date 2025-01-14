# -*- coding: UTF-8 -*-
# ==============================================================================

# ------------------------------------------------------------------------------
# Project: Simple python siril script to run CosmicClarity Denoise
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
from siril.connection import SirilInterface


# SET THIS PATH ! :
PATH = "/Users/nicolas.castel/Documents/Dev/CosmicClaritySuite_macos/"

EXE = "SetiAstroCosmicClarity_denoisemac"
    
print("CosmicClarity:begin") 
siril = SirilInterface()

try:
    siril.connect()
    
    WD = siril.get_siril_wd()
    
    FILE = os.path.basename(siril.get_image_filename())
    
    siril.cmd("savetif", PATH + "input/" + FILE , "-astro")
    
    print(os.popen(PATH + EXE).read())
    
    siril.cmd("load", PATH + "output/" + FILE + "_denoised.tif")
    
    os.remove( PATH + "input/" + FILE + ".tif")
    os.remove( PATH + "output/" + FILE + "_denoised.tif")
    
except Exception as e :
    print("\n**** ERROR *** " +  str(e) + "\n" )    

siril.disconnect()
del siril
print("CosmicClarity:end")

       

