# -*- coding: UTF-8 -*-
# ==============================================================================

# ------------------------------------------------------------------------------
# Project: Simple python siril script to run SetiAstro Statistical Stretch
# based on https://github.com/setiastro/setiastrosuite code
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
import numpy as np

def stretch_mono_image(image, target_median, normalize=False, apply_curves=False, curves_boost=0.0):
    black_point = max(np.min(image), np.median(image) - 2.7 * np.std(image))
    rescaled_image = (image - black_point) / (1 - black_point)
    median_image = np.median(rescaled_image)
    stretched_image = ((median_image - 1) * target_median * rescaled_image) / (median_image * (target_median + rescaled_image - 1) - target_median * rescaled_image)
    if apply_curves:
        stretched_image = apply_curves_adjustment(stretched_image, target_median, curves_boost)

    if normalize:
        stretched_image = stretched_image / np.max(stretched_image)

    return np.clip(stretched_image, 0, 1)


def stretch_color_image(image, target_median, linked=True, normalize=False, apply_curves=False, curves_boost=0.0):
    if linked:
        combined_median = np.median(image)
        combined_std = np.std(image)
        black_point = max(np.min(image), combined_median - 2.7 * combined_std)
        rescaled_image = (image - black_point) / (1 - black_point)
        median_image = np.median(rescaled_image)
        stretched_image = ((median_image - 1) * target_median * rescaled_image) / (median_image * (target_median + rescaled_image - 1) - target_median * rescaled_image)
    else:
        stretched_image = np.zeros_like(image)
        for channel in range(image.shape[2]):
            black_point = max(np.min(image[..., channel]), np.median(image[..., channel]) - 2.7 * np.std(image[..., channel]))
            rescaled_channel = (image[..., channel] - black_point) / (1 - black_point)
            median_channel = np.median(rescaled_channel)
            stretched_image[..., channel] = ((median_channel - 1) * target_median * rescaled_channel) / (median_channel * (target_median + rescaled_channel - 1) - target_median * rescaled_channel)

    if apply_curves:
        stretched_image = apply_curves_adjustment(stretched_image, target_median, curves_boost)

    if normalize:
        stretched_image = stretched_image / np.max(stretched_image)

    return np.clip(stretched_image, 0, 1)


def apply_curves_adjustment(image, target_median, curves_boost):
    curve = [
        [0.0, 0.0],
        [0.5 * target_median, 0.5 * target_median],
        [target_median, target_median],
        [(1 / 4 * (1 - target_median) + target_median),
         np.power((1 / 4 * (1 - target_median) + target_median), (1 - curves_boost))],
        [(3 / 4 * (1 - target_median) + target_median),
         np.power(np.power((3 / 4 * (1 - target_median) + target_median), (1 - curves_boost)), (1 - curves_boost))],
        [1.0, 1.0]
    ]
    adjusted_image = np.interp(image, [p[0] for p in curve], [p[1] for p in curve])
    return adjusted_image

print("StatisticalStrech:begin")
siril = SirilInterface()

try:
    siril.connect()

    if siril.claim_thread() :
        siril.undo_save_state("StatisticalStrech")
        image = siril.get_image_pixeldata()
        stretched = stretch_color_image(image, 0.25)
        siril.set_image_pixeldata(stretched)
        siril.release_thread()
    else :
        print("StatisticalStrech aborted, a siril processing is ongoing, end it and retry")

except Exception as e :
    print("\n**** ERROR *** " +  str(e) + "\n" )

siril.disconnect()
del siril
print("StatisticalStrech:end")
