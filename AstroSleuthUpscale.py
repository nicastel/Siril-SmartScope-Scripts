# -*- coding: UTF-8 -*-
# ==============================================================================

# ------------------------------------------------------------------------------
# Project: Python siril script to run AstroSleuth Upscaler via spandrel
# using model from https://github.com/Aveygo/AstroSleuth
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
import asyncio
import sirilpy as s
import numpy as np
import urllib.request 
import ssl
import tempfile
import math

s.ensure_installed("ttkthemes")
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from ttkthemes import ThemedTk
from sirilpy import tksiril

s.ensure_installed("torch")
import torch

s.ensure_installed("opencv-python")
import cv2

s.ensure_installed("spandrel")
from spandrel import ImageModelDescriptor, ModelLoader

VERSION = "1.0.0"

# Set image warning and max sizes
WARNING_SIZE = 4096
MAX_SIZE = None
#Image.MAX_IMAGE_PIXELS = 8192

# list of models
models = [["AstroSleuthV1","https://github.com/Aveygo/AstroSleuth/releases/download/v1/AstroSleuthV1.pth"],
    ["AstroSleuthV2","https://github.com/Aveygo/AstroSleuth/releases/download/v2/AstroSleuthV2.pth"], 
    ["AstroSleuthFAST","https://github.com/Aveygo/AstroSleuth/releases/download/v3/AstroSleuthFAST.pth"],  
    ["AstroSleuthNEXT","https://github.com/Aveygo/AstroSleuth/releases/download/v4/AstroSleuthNEXT.pth"]]

# suppported for SCUNet : Nvidia GPU / Apple MPS / DirectML on Windows / CPU
def get_device() -> torch.device:
    if torch.cuda.is_available():
        print("cuda acceleration used")
        return torch.device("cuda")
    if os.name == 'nt':
        s.ensure_installed("torch-directml")
        import torch_directml
        print("directml acceleration used")
        return torch_directml.default_device()
    if torch.backends.mps.is_available() :
        print("mps acceleration used")
        return torch.device("mps")
    else:
        print("cpu used")
        return torch.device("cpu")

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

class SirilAstroSleuth:
    def __init__(self, root):
        self.root = root
        self.root.title(f"AstroSleuth Upscale - v{VERSION}")
        self.root.resizable(False, False)

        self.style = tksiril.standard_style()

        # Initialize Siril connection
        self.siril = s.SirilInterface()

        if not self.siril.connect():
            self.siril.error_messagebox("Failed to connect to Siril")
            self.close_dialog()
            return

        if not self.siril.is_image_loaded():
            self.siril.error_messagebox("No image loaded")
            self.close_dialog()
            return

        if not self.siril.cmd("requires", "1.3.6"):
            self.close_dialog()
            return

        if not self.siril.cmd("requires", "1.3.6"):
            messagebox.showerror("Error", "Siril version requirement not met")
            self.close_dialog()
            return

        tksiril.match_theme_to_siril(self.root, self.siril)

        # Create widgets
        self.create_widgets()
        
    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(
            main_frame,
            text="AstroSleuth Upscale Settings",
            style="Header.TLabel"
        )
        title_label.pack(pady=(0, 20))
        
        # Model Selection Frame
        model_frame = ttk.LabelFrame(main_frame, text="Model selection", padding=10)
        model_frame.pack(fill=tk.X, padx=5, pady=5)

        self.model_var = tk.StringVar(value="https://github.com/Aveygo/AstroSleuth/releases/download/v1/AstroSleuthV1.pth")
        for model in models:
            ttk.Radiobutton(
                model_frame,
                text=model[0],
                variable=self.model_var,
                value=model[1]
            ).pack(anchor=tk.W, pady=2)

        # Action Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=10)

        close_btn = ttk.Button(
            button_frame,
            text="Close",
            command=self.close_dialog,
            style="TButton"
        )
        close_btn.pack(side=tk.LEFT, padx=5)

        apply_btn = ttk.Button(
            button_frame,
            text="Apply",
            command=self._on_apply,
            style="TButton"
        )
        apply_btn.pack(side=tk.LEFT, padx=5)

    def _on_apply(self):
        # Wrap the async method to run in the event loop
        self.root.after(0, self._run_async_task)

    def _run_async_task(self):
        asyncio.run(self._apply_changes())

    def close_dialog(self):
        self.siril.disconnect()
        self.root.quit()
        self.root.destroy()

    async def _apply_changes(self):
        temp_filename = None
        try:
             # Read user input values
            model = self.model_var.get()
            print (model)
            
            self.siril.reset_progress()

            modelpath = os.path.join(self.siril.get_siril_configdir(),os.path.basename(model))
    
            if os.path.isfile(modelpath) :
                print("model found : "+modelpath) 
            else :
                ssl._create_default_https_context = ssl._create_stdlib_context
                urllib.request.urlretrieve(model, modelpath)
                print("model downloaded : "+modelpath)
                
            device = get_device()
            
            # load a model from disk
            model = ModelLoader().load_from_file(r""+modelpath)
            # make sure it's an image to image model
            assert isinstance(model, ImageModelDescriptor)
            
            model.eval()
            
            self.siril.update_progress("model initialised",0.05)
            
            # Create a temporary file
            with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as temp_file:
                temp_filename = temp_file.name
                self.siril.log(f"Temporary file created: {temp_filename}")

            # Save current file
            self.siril.cmd("savetif", os.path.splitext(temp_filename)[0], " -astro")
            self.siril.log(f"FITS file saved: {temp_filename}")
                
            # read image out send it to the GPU
            imagecv2in = cv2.imread(temp_filename, cv2.IMREAD_COLOR)
            original_height, original_width, channels = imagecv2in.shape
            
            print("original_height :"+str(original_height)+" original_width :"+str(original_width))

            tile_size = 256
            scale = 4

            # Because tiles may not fit perfectly, we resize to the closest multiple of tile_size
            imgcv2resized = cv2.resize(imagecv2in,(original_width//tile_size * tile_size + tile_size,original_height//tile_size * tile_size + tile_size),interpolation=cv2.INTER_CUBIC)

            # Allocate an image to save the tiles
            imgresult = cv2.resize(imgcv2resized,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC)
         
            for i, tile in enumerate(tile_process(model, imgcv2resized, scale, tile_size, yield_extra_details=True)):

                if tile is None:
                    break

                tile_data, x, y, w, h, p = tile
                if w != 0 and h != 0 :
                    imgresult[x*scale:x*scale+tile_size*scale,y*scale:y*scale+tile_size*scale] = tile_data
                    
                self.siril.update_progress("Image upscale ongoing",p)

            # Resize back to the expected size
            imagecv2out = cv2.resize(imgresult,(original_width*scale,original_height*scale),interpolation=cv2.INTER_CUBIC)

            # write the image to the disk
            cv2.imwrite(temp_filename, imagecv2out)
            
            self.siril.update_progress("Image upscaled",1.0)
            
            # Load back into Siril
            self.siril.cmd("load", temp_filename)
            self.siril.log(f"FITS file loaded: {temp_filename}")
            self.siril.log("AstroSleuth Upscale complete.")

        except Exception as e:
            print(f"Error in apply_changes: {str(e)}")
            self.siril.update_progress(f"Error: {str(e)}", 0)
        finally:
            # Clean up: delete the temporary file
            if temp_filename and os.path.exists(temp_filename):
                try:
                   os.remove(temp_filename)
                   self.siril.log(f"Temporary file deleted: {temp_filename}")
                except OSError as e:
                   self.siril.log(f"Failed to delete temporary file: {str(e)}")
            # Release the thread in the finally: block so that it is guaranteed to be released
            self.siril.release_thread()

def main():
    try:
        root = ThemedTk()
        app = SirilAstroSleuth(root)
        root.mainloop()
    except Exception as e:
        print(f"Error initializing application: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()

