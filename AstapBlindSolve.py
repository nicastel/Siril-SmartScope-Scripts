# -*- coding: UTF-8 -*-
# ==============================================================================

# ------------------------------------------------------------------------------
# Project: Simple python siril script to blind solve using astap_cli
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
from sirilpy.connection import SirilInterface

# SET THESE 2 PATHS ! :
PATH = "/Users/nicolas.castel/Documents/Dev/astap/astap_cli"
D80 = "/Users/nicolas.castel/Documents/Dev/astap/d80"

print("ASTAP:begin") 
siril = SirilInterface()

try:
    siril.connect()
    
    FILE = siril.get_image_filename()
    
    siril.cmd("save", FILE + "wcs.fit")
    
    CMD = PATH + " -f " + FILE + "wcs.fit"
    CMD += " -D d80 -d " + D80
    CMD += " -update"
    
    print(CMD)
    
    print(os.popen(CMD).read())
    
    siril.cmd("load",  FILE + "wcs.fit")
    
    #annotate using astrometry information
    siril.cmd("conesearch")
    
except Exception as e :
    print("\n**** ERROR *** " +  str(e) + "\n" )    

siril.disconnect()
del siril
print("ASTAP:end")

       

