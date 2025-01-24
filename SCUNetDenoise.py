# -*- coding: UTF-8 -*-
# ==============================================================================

# ------------------------------------------------------------------------------
# Project: Python siril script to run SCUNet denoiser via spandrel
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
import sirilpy as s
import numpy as np
import urllib.request 
import ssl
import tempfile
import math

s.ensure_installed("torch")
import torch

s.ensure_installed("opencv-python")
import cv2

s.ensure_installed("spandrel")
from spandrel import ImageModelDescriptor, ModelLoader

# suppported for SCUNet : Nvidia GPU / Apple MPS / DirectML on Windows / CPU
def get_device() -> torch.device:
    if os.name == 'nt':
        s.ensure_installed("torch_directml")
        import torch_directml
        return torch_directml.default_device()
    else:
        return torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

def image_to_tensor(img: np.ndarray) -> torch.Tensor:
    img = img.astype(np.float32) / 255.0
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    if img.shape[2] == 1:
        pass
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    tensor = torch.from_numpy(img)
    return tensor.unsqueeze(0)


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    image = tensor.cpu().squeeze().numpy()
    image = np.transpose(image, (1, 2, 0))
    image = np.clip((image * 255.0).round(), 0, 255)
    image = image.astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image

def image_inference_tensor(
    model: ImageModelDescriptor, tensor: torch.Tensor
) -> torch.Tensor:
    with torch.no_grad():
        return model(tensor)
        
def tile_process(model: ImageModelDescriptor, data: np.ndarray, scale, tile_size, yield_extra_details=False):
        """
        Process data [height, width, channel] into tiles of size [tile_size, tile_size, channel],
        feed them one by one into the model, then yield the resulting output tiles.
        """

        tile_pad=16

        # [height, width, channel] -> [1, channel, height, width]
        data = np.rollaxis(data, 2, 0)
        data = np.expand_dims(data, axis=0)
        data = np.clip(data, 0, 255)

        batch, channel, height, width = data.shape
        print("height :"+str(height)+" width :"+str(width))

        tiles_x = width // tile_size
        tiles_y = height // tile_size

        for i in range(tiles_x * tiles_y):
            x = i % tiles_y
            y = math.floor(i/tiles_y)
            
            print("tile x :"+str(x)+" y :"+str(y))

            input_start_x = y * tile_size
            input_start_y = x * tile_size

            input_end_x = min(input_start_x + tile_size, width)
            input_end_y = min(input_start_y + tile_size, height)
            
            print("input_start_x :"+str(input_start_x)+" input_end_x :"+str(input_end_x))
            print("input_start_y :"+str(input_start_y)+" input_end_y :"+str(input_end_y))

            input_start_x_pad = max(input_start_x - tile_pad, 0)
            input_end_x_pad = min(input_end_x + tile_pad, width)
            input_start_y_pad = max(input_start_y - tile_pad, 0)
            input_end_y_pad = min(input_end_y + tile_pad, height)
            
            print("input_start_x_pad :"+str(input_start_x_pad)+" input_end_x_pad :"+str(input_end_x_pad))
            print("input_start_y_pad :"+str(input_start_y_pad)+" input_end_y_pad :"+str(input_end_y_pad))

            input_tile_width = input_end_x - input_start_x
            input_tile_height = input_end_y - input_start_y

            input_tile = data[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad].astype(np.float32) / 255

            output_tile = image_inference_tensor(model,torch.from_numpy(input_tile))
            progress = (i+1) / (tiles_y * tiles_x)

            output_start_x_tile = (input_start_x - input_start_x_pad) * scale
            output_end_x_tile = output_start_x_tile + (input_tile_width * scale)
            output_start_y_tile = (input_start_y - input_start_y_pad) * scale
            output_end_y_tile = output_start_y_tile + (input_tile_height * scale)
            
            print("output_start_x_tile :"+str(output_start_x_tile)+" output_end_x_tile :"+str(output_end_x_tile))
            print("output_start_y_tile :"+str(output_start_y_tile)+" output_end_y_tile :"+str(output_end_y_tile))         

            output_tile = output_tile[:, :, output_start_y_tile:output_end_y_tile, output_start_x_tile:output_end_x_tile]

            output_tile = (np.rollaxis(output_tile.cpu().detach().numpy(), 1, 4).squeeze(0).clip(0,1) * 255).astype(np.uint8)

            if yield_extra_details:
                yield (output_tile, input_start_y, input_start_x, input_tile_width, input_tile_height, progress)
            else:
                yield output_tile

        yield None

# Set image warning and max sizes
WARNING_SIZE = 4096
MAX_SIZE = None
#Image.MAX_IMAGE_PIXELS = 8192

print("SCUNetDenoise:begin")
siril = s.SirilInterface()
temp_filename = None

try:
    siril.connect()
    siril.reset_progress()
    
    modelpath = os.path.join(siril.get_siril_configdir(),"scunet_color_real_psnr.pth")
    
    if os.path.isfile(modelpath) :
        print("SCUnet model found : "+modelpath) 
    else :
        ssl._create_default_https_context = ssl._create_stdlib_context
        urllib.request.urlretrieve("https://github.com/cszn/KAIR/releases/download/v1.0/scunet_color_real_psnr.pth", modelpath)
        print("SCUnet model downloaded : "+modelpath)
    
    # load a model from disk
    model = ModelLoader().load_from_file(r""+modelpath)
    # make sure it's an image to image model
    assert isinstance(model, ImageModelDescriptor)
    
    model.eval()
    
    siril.update_progress("SCUNet model initialised",0.05)
    
    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as temp_file:
        temp_filename = temp_file.name
        siril.log(f"Temporary file created: {temp_filename}")

    # Save current file
    siril.cmd("savetif", os.path.splitext(temp_filename)[0], " -astro")
    siril.log(f"FITS file saved: {temp_filename}")
        
    # read image out send it to the GPU
    imagecv2in = cv2.imread(temp_filename, cv2.IMREAD_COLOR)
    original_height, original_width, channels = imagecv2in.shape
    
    print("original_height :"+str(original_height)+" original_width :"+str(original_width))

    tile_size = 1024
    scale = 1

    # Because tiles may not fit perfectly, we resize to the closest multiple of tile_size
    imgcv2resized = cv2.resize(imagecv2in,(original_width//tile_size * tile_size + tile_size,original_height//tile_size * tile_size + tile_size),interpolation=cv2.INTER_CUBIC)

    # Allocate an image to save the tiles
    imgresult = cv2.copyMakeBorder(imgcv2resized,0,0,0,0,cv2.BORDER_REPLICATE)

    for i, tile in enumerate(tile_process(model, imgcv2resized, scale, tile_size, yield_extra_details=True)):

        if tile is None:
            break

        tile_data, x, y, w, h, p = tile
        if w != 0 and h != 0 :
            imgresult[x*scale:x*scale+tile_size*scale,y*scale:y*scale+tile_size*scale] = tile_data
            
        siril.update_progress("Image denoising ongoing",p)

    # Resize back to the expected size
    imagecv2out = cv2.resize(imgresult,(original_width*scale,original_height*scale),interpolation=cv2.INTER_CUBIC)

    # write the image to the disk
    cv2.imwrite(temp_filename, imagecv2out)
    
    siril.update_progress("Image denoised",1.0)
    
    # Load back into Siril
    siril.cmd("load", temp_filename)
    siril.log(f"FITS file loaded: {temp_filename}")

except Exception as e :
    print("\n**** ERROR *** " +  str(e) + "\n" )
finally:
    # Clean up: delete the temporary file
    if temp_filename and os.path.exists(temp_filename):
        try:
           os.remove(temp_filename)
           siril.log(f"Temporary file deleted: {temp_filename}")
        except OSError as e:
           siril.log(f"Failed to delete temporary file: {str(e)}")

siril.disconnect()
del siril
print("SCUNetDenoise:end")
