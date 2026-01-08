#
# ***********************************************
#
# Original Author:  Nicolas CASTEL <nic.castel (at) gmail.com>
#
# Copyright (C) 2025 - Carlo Mollicone - AstroBOH
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Version 2.0 by Carlo Mollicone (CarCarlo147) and can be reached at:
# https://www.astroboh.it
# https://www.facebook.com/carlo.mollicone.9
#
# ***********************************************
#
# --------------------------------------------------------------------------------------------------
# This program is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free Software Foundation,
# either version 3 of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with this program.
# If not, see <https://www.gnu.org/licenses/>.
# --------------------------------------------------------------------------------------------------
#
# Description:
# ------------------------------------------------------------------------------
# Project: Python siril script to run SCUNet denoiser via spandrel
#          using model from https://github.com/cszn/SCUNet
#          and https://ubersmooth.com/
#
#          Now supports:
#           - GUI Framework: PyQt6
#           - Single Image and Sequence Processing
#           - Model Management: models are stored in a dedicated folder
#           - Mono image support
# ------------------------------------------------------------------------------
#
# Version History
# 1.0.3 - Original release by Nicolas CASTEL
# 2.0.0 - Ported to PyQt6 by Carlo Mollicone - AstroBOH
#       - Added Sequence Processing support
#       - Added Mono image support (via RGB replication), useful for Solar/Planetary workflows
#       - Improved Model Management: models are now stored in a dedicated 'scunet_models'
#         folder to keep the working directory clean. Added download progress bar.
#       - Performance:
#           Added Auto-Tile tuning to detect max safe tile size and prevent VRAM crashes.
#           Quality: Implemented Weighted Soft Blending to completely eliminate tile grid seams.
#       - Fix tile processing edge cases to avoid artifacts at image borders.
#       - Fix tile dimension mismatches to ensure accurate processing.
#       - Added support for Intel Arc / XPU devices via PyTorch
#       - Added instructions
#       - Added ROI preview
#

VERSION = "2.0.0"

import sys
import os
import numpy as np
import urllib.request
import ssl
import math
import zipfile
import traceback
import base64

# Attempt to import sirilpy. If not running inside Siril, the import will fail.
try:
    import sirilpy as s

    # Check the module version
    if not s.check_module_version('>=0.7.46'):
        print("Error: requires sirilpy module >= 0.7.46")
        sys.exit(1)

    # TODO: uncomment when Siril 1.4.1 is released and sirilpy version is definitely greater than or equal to 1.0.10
    # # Check the module version
    # if not s.check_module_version('>=1.0.10'):
    #     print("Error: requires sirilpy module >= 1.0.10")
    #     sys.exit(1)

    # Import Siril GUI related components
    from sirilpy import SirilError

    print("Warning: a significant size of packages are about to be downloaded and installed and it will take some time")

    s.ensure_installed("PyQt6", "spandrel", "torch")

    # --- PyQt6 Imports ---
    from PyQt6.QtWidgets import (
        QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
        QPushButton, QComboBox, QCheckBox, QMessageBox, QGroupBox, 
        QProgressBar, QDoubleSpinBox, QLineEdit, QFormLayout,
        QRadioButton, QSlider, QFrame, QStyle, QSizePolicy
    )
    from PyQt6.QtGui import QCloseEvent, QIcon, QPixmap, QImage
    from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject

    # --- Torch & Spandrel ---
    # Determine the correct torch package based on OS and hardware,
    # and ensure it is installed
    s.TorchHelper().ensure_torch()
    import torch
    from spandrel import ImageModelDescriptor, ModelLoader

except ImportError as e:
    print("Warning: sirilpy not found. The script is not running in the Siril environment.")
    sys.exit(1)

# --- Models List ---
# Format: [Name, URL, Description]
models_list = [
    ["SCUNet Color Real PSNR", "https://github.com/cszn/KAIR/releases/download/v1.0/scunet_color_real_psnr.pth", "Best all around model but can be too aggressive on stars"],
    ["SCUNet Color Real GAN", "https://github.com/cszn/KAIR/releases/download/v1.0/scunet_color_real_gan.pth", "Less aggressive denoise"],
    ["SCUNet Color 15", "https://github.com/cszn/KAIR/releases/download/v1.0/scunet_color_15.pth", "Gaussian noise level 15"],
    ["SCUNet Color 25", "https://github.com/cszn/KAIR/releases/download/v1.0/scunet_color_25.pth", "Gaussian noise level 25"],
    ["SCUNet Color 50", "https://github.com/cszn/KAIR/releases/download/v1.0/scunet_color_50.pth", "Gaussian noise level 50"],
    ["UberSmooth dso stars 0.1", "https://ubersmooth.com/uberSmooth-dso-stars-v0.1.zip", "Pretty good on stars but too aggressive on Hii regions"],
    ["UberSmooth dso stars 0.2", "https://ubersmooth.com/uberSmooth-dso-stars-v0.2.zip", "Not as aggressive as UberSmooth 0.1, but also not great"],
    ["UberSmooth planetary 0.1", "https://ubersmooth.com/uberSmooth-planetary-v0.1.zip", "Only denoise/deblur no extra star treatment"]
]

# --- Core Logic Functions (Device & Tiling) ---

# suppported for SCUNet : Nvidia GPU / Apple MPS / DirectML on Windows / CPU / Intel XPU
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available() :
        return torch.device("mps")
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        return torch.device("xpu")  # Intel Arc / XPU Support
    else:
        return torch.device("cpu")

def image_to_tensor(device: torch.device, img: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(img)
    return tensor.to(device)

def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    return (np.rollaxis(tensor.cpu().detach().numpy(), 1, 4).squeeze(0)).astype(np.float32)

def image_inference_tensor(model: ImageModelDescriptor, tensor: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return model(tensor)

def determine_optimal_tile_size(model, device, start_size=512):
    """
    Tries to run a dummy inference to find the maximum safe tile size.
    Returns the determined size (e.g., 512, 384, 256, or 128).
    """
    # If we are on CPU, we avoid heavy tests and play it safe (or we stay at 512 if there is RAM)
    if device.type == 'cpu':
        return 256  # Safe default for CPU

    # Sizes to test in descending order
    test_sizes = [512, 384, 256, 128]
    
    # Filter sizes larger than start_size (if user manually requested a max cap, logic logic usually starts at 512)
    test_sizes = [s for s in test_sizes if s <= start_size]

    for size in test_sizes:
        try:
            # Create a dummy tensor: (Batch=1, Channels=3, H=size, W=size)
            dummy_input = torch.zeros(1, 3, size, size).to(device)
            
            # Dry run inference (no_grad is already in image_inference_tensor context usually, but explicit here)
            with torch.no_grad():
                model(dummy_input)
            
            # If we are here, it worked!
            # Clear cache to free the dummy memory immediately
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            elif device.type == 'xpu':
                torch.xpu.empty_cache()
            
            return size

        except RuntimeError as e:
            # Check if it is an Out Of Memory error
            if "out of memory" in str(e).lower():
                # Clear cache and try next smaller size
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                elif device.type == 'xpu':
                    torch.xpu.empty_cache()
                continue
            else:
                # If it's another error, re-raise it
                raise e
    
    # Fallback if even 128 fails (very unlikely)
    return 128

def get_tile_weight(h, w, device):
    """
    Create a 2D weight mask that fades at the edges (from 0 to 1 and then back to 0).
    Use a linear ramp (Pyramid).
    """
    # Create 1D gradients
    # Example: [0, 0.1, ... 1 ... 0.1, 0]
    # Use linspace + minimum to create a pyramid
    x = torch.linspace(0, 1, w, device=device)
    y = torch.linspace(0, 1, h, device=device)
    
    # Ramp: min(x, 1-x) * 2 makes a pyramid 0->1->0
    wx = torch.min(x, 1 - x) * 2
    wy = torch.min(y, 1 - y) * 2
    
    # Avoid absolute zero at the edges to avoid dividing by zero (let's say an epsilon)
    wx = torch.clamp(wx, min=0.1)
    wy = torch.clamp(wy, min=0.1)
    
    # External product to make 2D mask
    weight = wy.unsqueeze(1) * wx.unsqueeze(0)
    
    # Add channel size: (1, H, W) or (C, H, W) for broadcasting
    return weight.unsqueeze(0)

def tile_process(device: torch.device, model: ImageModelDescriptor, data: np.ndarray, scale, tile_size, yield_extra_details=False):
    """
    Process data [height, width, channel] into tiles of size [tile_size, tile_size, channel],
    feed them one by one into the model, then yield the resulting output tiles.
    """
    tile_pad = 144
    
    # [height, width, channel] -> [1, channel, height, width]
    data = np.rollaxis(data, 2, 0)
    data = np.expand_dims(data, axis=0)

    batch, channel, height, width = data.shape

    tiles_x = width // tile_size
    if tiles_x * tile_size < width: tiles_x += 1
    tiles_y = height // tile_size
    if tiles_y * tile_size < height: tiles_y += 1

    for i in range(tiles_x * tiles_y):
        x = math.floor(i / tiles_y)
        y = i % tiles_y

        if x < tiles_x - 1: input_start_x = x * tile_size
        else: input_start_x = width - tile_size
        if y < tiles_y - 1: input_start_y = y * tile_size
        else: input_start_y = height - tile_size

        if input_start_x < 0: input_start_x = 0
        if input_start_y < 0: input_start_y = 0

        input_end_x = min(input_start_x + tile_size, width)
        input_end_y = min(input_start_y + tile_size, height)

        input_start_x_pad = max(input_start_x - tile_pad, 0)
        input_end_x_pad = min(input_end_x + tile_pad, width)
        input_start_y_pad = max(input_start_y - tile_pad, 0)
        input_end_y_pad = min(input_end_y + tile_pad, height)

        input_tile_width = input_end_x - input_start_x
        input_tile_height = input_end_y - input_start_y

        input_tile = data[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad].astype(np.float32)
        output_tile = image_inference_tensor(model, image_to_tensor(device, input_tile))
        
        progress = (i + 1) / (tiles_y * tiles_x)

        output_start_x_tile = (input_start_x - input_start_x_pad) * scale
        output_end_x_tile = output_start_x_tile + (input_tile_width * scale)
        output_start_y_tile = (input_start_y - input_start_y_pad) * scale
        output_end_y_tile = output_start_y_tile + (input_tile_height * scale)

        output_tile = output_tile[:, :, output_start_y_tile:output_end_y_tile, output_start_x_tile:output_end_x_tile]
        output_tile = tensor_to_image(output_tile)

        if yield_extra_details:
            yield (output_tile, input_start_y, input_start_x, input_tile_width, input_tile_height, progress)
        else:
            yield output_tile
    yield None

def process_image_buffer(image_data, model, device, strength, tile_size, progress_callback=None):
    original_dtype = image_data.dtype
    
    #1. Normalization (We handle 8-bit, 16-bit and Float)
    if original_dtype == np.uint8:
        # 8-bit (0-255) -> Float (0.0-1.0)
        pixel_data = image_data.astype(np.float32) / 255.0
    elif original_dtype == np.uint16:
        # 16-bit (0-65535) -> Float (0.0-1.0)
        pixel_data = image_data.astype(np.float32) / 65535.0
    else:
        # Float (presumably 0.0-1.0)
        pixel_data = image_data

    # 2. Mono -> RGB Hack
    # MONO Image Management -> RGB Fake (because the model requires 3 channels)
    # If the image is 2D (H, W), we make it (3, H, W) by duplicating the channels
    is_mono = False
    if pixel_data.ndim == 2:
        is_mono = True
        # (H, W) -> (3, H, W)
        pixel_data = np.stack((pixel_data,)*3, axis=0)
    elif pixel_data.ndim == 3 and pixel_data.shape[0] == 1:
        # If it is (1, H, W), we take it to (3, H, W)
        is_mono = True
        pixel_data = np.repeat(pixel_data, 3, axis=0)

    # Preparing the Accumulation Buffer on the GPU (for speed) or CPU
    # We use the same device as the model to avoid continuous transfers.
    # Warning: If VRAM is limited, it's best to keep large buffers on the CPU.
    # To be safe, we use the CPU for the final accumulation here, to avoid OOMs on huge images.
    c, h, w = pixel_data.shape
    
    # Note: tile_process expects (H, W, C) as input.
    # (C, H, W) -> (H, W, C) for compatibility with existing tile_process logic
    pixel_data_hwc = np.transpose(pixel_data, (1, 2, 0))

    # Sum buffer and Weight buffer
    # We keep them in float32 on the CPU
    output_sum = torch.zeros((c, h, w), dtype=torch.float32, device='cpu')
    output_weight = torch.zeros((c, h, w), dtype=torch.float32, device='cpu')

    # We generate the weight mask for the standard tile size
    # If the edge tiles are smaller, we'll regenerate it on the fly,
    # but we'll delete the main one
    base_weight_mask = get_tile_weight(tile_size, tile_size, 'cpu') # On CPU to match buffers

    scale = 1 # SCUNet does not upscale, it scales 1:1

    # Let's use tile_process.
    # WARNING: tile_process returns a numpy array. We'll use torch for fast accumulation.
    for i, tile_info in enumerate(tile_process(device, model, pixel_data_hwc, scale, tile_size, yield_extra_details=True)):
        if tile_info is None: break
        
        # Ignore the calculated h_tile and w_tile (they may be incorrect at the edges)
        tile_data_numpy, y_start, x_start, _, _, p = tile_info
        
        # Let's create the tensor
        tile_tensor = torch.from_numpy(tile_data_numpy.transpose(2, 0, 1))

        # --- Let's read the REAL dimensions from the tensor ---
        c_real, h_real, w_real = tile_tensor.shape
        
        if h_real <= 0 or w_real <= 0: continue

        # Calculate final coordinates based on REAL dimensions
        y_end = y_start + h_real
        x_end = x_start + w_real

        # 4. SAFETY CLIP: Calculate the REAL space available in the target image
        # Clamp to the maximum image size output_sum
        y_end_safe = min(y_end, h)
        x_end_safe = min(x_end, w)
        y_start_safe = max(0, y_start)
        x_start_safe = max(0, x_start)
        
        # Let's calculate how much space we REALLY have to write
        write_h = y_end_safe - y_start_safe
        write_w = x_end_safe - x_start_safe

        if write_h <= 0 or write_w <= 0: continue

        # 5. ADAPTATION: If the tensor is larger than the space (e.g. we are at the edge), we cut it
        if write_h != h_real or write_w != w_real:
            tile_tensor = tile_tensor[:, :write_h, :write_w]
            # Update the dimensions for the mask
            h_real = write_h
            w_real = write_w

        # Weight Mask Management (Now safe because it uses adapted dimensions)
        if h_real == tile_size and w_real == tile_size:
             mask = base_weight_mask
        else:
             # Generate custom mask
             mask = get_tile_weight(h_real, w_real, 'cpu')

        #7. Weighted Accumulation (Now the dimensions match mathematically)
        output_sum[:, y_start_safe:y_end_safe, x_start_safe:x_end_safe] += tile_tensor * mask
        output_weight[:, y_start_safe:y_end_safe, x_start_safe:x_end_safe] += mask

        if progress_callback:
            progress_callback(p)

    # 3. Final Normalization
    # Result = Sum / Weight
    # Add an epsilon to avoid division by zero (unlikely but safe)
    output_image_tensor = output_sum / (output_weight + 1e-8)
    
    # Back to Numpy
    output_image = output_image_tensor.numpy()

    #4. Mono Restore
    if is_mono:
        output_image = output_image[0, :, :]

    #5. De-normalization
    final_dtype = original_dtype

    if original_dtype == np.uint8:
        # Float -> 8-bit (0-255)
        output_image = np.clip(output_image, 0, 1) * 255.0
        output_image = output_image.astype(np.uint8)
    elif original_dtype == np.uint16:
        # Float -> 16-bit (0-65535)
        output_image = np.clip(output_image, 0, 1) * 65535.0
        output_image = output_image.astype(np.uint16)
    # If float, stay float

    #6. Blend with original
    if strength != 1.0:
        # Background preparation for the blend 
        # If is_mono is true, image_data could be 2D or 3D. Let's make sure they match.
        if is_mono and image_data.ndim == 3:
             # If image_data was (1, H, W) and now output is (H, W), we use image_data[0]
             bg = image_data[0]
        else:
             bg = image_data
        
        blended = output_image * strength + bg * (1 - strength)
        return blended.astype(final_dtype)
    else:
        return output_image

# --- Worker Thread ---
class ProcessingWorker(QObject):
    """
    A worker that performs model download and processing (single or sequential)
    in a separate thread to avoid blocking the GUI.
    """
    finished = pyqtSignal()
    progress_update = pyqtSignal(int, str) # percent, text
    error_occurred = pyqtSignal(str)

    def __init__(self, siril, params):
        super().__init__()
        self.siril = siril
        self.params = params
        self._is_running = True

    def run(self):
        try:
            model_url = self.params['model_url']
            strength = self.params['strength']
            is_sequence = self.params['is_sequence']
            seq_prefix = self.params['seq_prefix']

            #1. Setup Model Directory (Folder: scunet_models)
            self.progress_update.emit(0, "Checking Model...")
            
            # Get the user data folder and create a subfolder 'scunet_models'
            user_dir = self.siril.get_siril_userdatadir()
            models_dir = os.path.join(user_dir, "scunet_models")
            
            # Create the folder if it doesn't exist
            if not os.path.exists(models_dir):
                os.makedirs(models_dir)

            # Define the full path to the file
            model_filename = os.path.basename(model_url)
            modelpath = os.path.join(models_dir, model_filename)

            # Hook function for download progress
            def download_progress_hook(block_num, block_size, total_size):
                if not self._is_running: raise Exception("Download cancelled")
                if total_size > 0:
                    downloaded = block_num * block_size
                    percent = int((downloaded / total_size) * 100)
                    # Update every 2% for smoothness
                    if percent % 2 == 0: 
                        self.progress_update.emit(percent, f"Downloading Model: {percent}%")
                else:
                    self.progress_update.emit(0, "Downloading Model... (size unknown)")

            # Download if it doesn't exist
            if not os.path.isfile(modelpath):
                self.siril.log(f"Downloading model to: {modelpath}", s.LogColor.BLUE)
                ssl._create_default_https_context = ssl._create_stdlib_context
                
                try:
                    urllib.request.urlretrieve(model_url, modelpath, reporthook=download_progress_hook)
                    self.siril.log("Model download completed.", s.LogColor.GREEN)
                except Exception as e:
                    # If the download fails or is canceled, remove the partial file
                    if os.path.exists(modelpath):
                        os.remove(modelpath)
                    raise e
            else:
                self.siril.log(f"Using existing model at: {modelpath}", s.LogColor.BLUE)
            
            # ZIP Management (UberSmooth)
            if zipfile.is_zipfile(modelpath):
                with zipfile.ZipFile(modelpath, 'r') as zip_ref:
                    # EXPLICITLY extract to the models folder
                    zip_ref.extractall(models_dir)
                    # Update the path to the extracted .pth file
                    modelpath = modelpath.replace(".zip", ".pth")

            # 2. Load Model
            self.progress_update.emit(0, "Loading Model into Memory...")
            device = get_device()

            # Log device info to Siril console
            if device.type == 'cuda':
                device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Unknown NVIDIA GPU"
                self.siril.log(f"Acceleration: CUDA ({device_name})", s.LogColor.GREEN)
            elif device.type == 'mps':
                self.siril.log("Acceleration: Apple Metal Performance Shaders (MPS)", s.LogColor.GREEN)
            elif device.type == 'xpu':
                self.siril.log("Acceleration: Intel XPU (Arc GPU detected)", s.LogColor.GREEN)
            else:
                self.siril.log("Acceleration: CPU (No GPU detected)", s.LogColor.GREEN)
            
            # Load the model from the correct path
            model = ModelLoader().load_from_file(str(modelpath)).eval().to(device)

            # --- CHECK ARCHITECTURE ---
            # Spandrel wraps the PyTorch model in model.model
            # Let's check the architecture class name.
            architecture_name = type(model.model).__name__
            
            # SCUNet in Spandrel is usually called "SCUNet"
            if "SCUNet" not in architecture_name:
                raise RuntimeError(
                    f"Invalid model selected. Expected a SCUNet model, but detected: '{architecture_name}'.\n"
                    "Please select a valid SCUNet .pth file."
                )

            assert isinstance(model, ImageModelDescriptor)

            # --- DETERMINE TILE SIZE ---
            req_tile = self.params['tile_size']
            
            if req_tile == "Auto":
                self.progress_update.emit(0, "Auto-tuning Tile Size...")
                final_tile_size = determine_optimal_tile_size(model, device)
                self.siril.log(f"Auto-Tuning: Selected Tile Size {final_tile_size}px", s.LogColor.GREEN)
            else:
                final_tile_size = req_tile
                self.siril.log(f"Manual Tile Size: {final_tile_size}px", s.LogColor.BLUE)

            # 3. Processing
            if is_sequence:
                if not self.siril.is_sequence_loaded():
                    raise RuntimeError("No sequence loaded in Siril.")

                current_sequence = self.siril.get_seq()
                num_images = current_sequence.number
                
                for i in range(num_images):
                    if not self._is_running: break

                    if not current_sequence.imgparam[i].incl:
                        continue

                    # 1. Get filename
                    filename = self.siril.get_seq_frame_filename(i)
                    
                    # Extract the extension from the original file (e.g., ".fit", ".fits")
                    _, file_extension = os.path.splitext(filename)
                    # If for some reason there is no extension, use .fit by default
                    if not file_extension: file_extension = ".fit"

                    # 2. Load with pixels
                    ffit_image = self.siril.load_image_from_file(filename, with_pixels=True)
                    image_data = ffit_image.data
                    header = ffit_image.header

                    def callback(tile_p):
                        if not self._is_running: return
                        global_p = (i + tile_p) / num_images
                        self.progress_update.emit(int(global_p * 100), f"Processing frame {i+1}/{num_images}")

                    # 3. Process
                    processed_data = process_image_buffer(image_data, model, device, strength, final_tile_size, callback)

                    # 4. Save (use i+1 to start the sequence counter from 00001)
                    new_filename = f"{seq_prefix}{i+1:05d}{file_extension}"
                    self.siril.save_image_file(processed_data, header=header, filename=new_filename)

                self.progress_update.emit(100, f"Sequence saved: {seq_prefix}...")

            else:
                # Single Image
                if not self.siril.is_image_loaded():
                    raise RuntimeError("No image loaded in Siril.")

                image = self.siril.get_image()
                image_data = image.data

                def callback(tile_p):
                    if not self._is_running: return
                    self.progress_update.emit(int(tile_p * 100), "Denoising Image...")

                processed_data = process_image_buffer(image_data, model, device, strength, final_tile_size, callback)
                
                self.siril.undo_save_state("SCUnet denoise")
                with self.siril.image_lock(): 
                    self.siril.set_image_pixeldata(processed_data)
                
                    # --- RESET SLIDERS (0 - MAX) ---
                    # Set display cursors to full range based on data type
                    try:
                        if processed_data.dtype == np.uint8:
                            # 8-bit (JPG/PNG): Range 0-255
                            self.siril.set_siril_slider_lohi(0, 255)
                        elif processed_data.dtype == np.uint16:
                            # 16-bit (FITS): Range 0-65535
                            self.siril.set_siril_slider_lohi(0, 65535)
                        elif np.issubdtype(processed_data.dtype, np.floating):
                            # Float: Range 0.0-1.0
                            self.siril.set_siril_slider_lohi(0.0, 1.0)
                    except Exception:
                        pass

                self.progress_update.emit(100, "Done.")

            self.finished.emit()

        except Exception as e:
            traceback.print_exc()
            self.error_occurred.emit(str(e))

    def stop(self):
        self._is_running = False

# --- Image Conversion Helper for QT ---
def numpy_to_qpixmap(img_data):
    """ Converts a numpy (C, H, W) or (H, W) float32 array to a QPixmap for display. """
    # 1. Vertical Flip (standard FITS is bottom-up, screens are top-down)
    # Flip on the Y-axis (H), which is axis 1 for (C,H,W) or axis 0 for (H,W)
    if img_data.ndim == 3:
        img_data = np.flip(img_data, axis=1)
    else:
        img_data = np.flip(img_data, axis=0)

    # 1. Normalize [0, 255] based on data type
    if img_data.dtype == np.uint16:
        # If it is 16 bits (0-65535), scale to 8 bits
        img_disp = (np.clip(img_data, 0, 65535) / 256.0).astype(np.uint8)
    elif img_data.dtype == np.float32 or img_data.dtype == np.float64:
        # If it is float (0.0-1.0), scale to 255
        img_disp = (np.clip(img_data, 0.0, 1.0) * 255.0).astype(np.uint8)
    else:
        # Already assume uint8
        img_disp = np.clip(img_data, 0, 255).astype(np.uint8)

    # 2. Format and Size Management
    if img_disp.ndim == 3:
        # (C, H, W) -> (H, W, C) for view
        # Warning: Transpose creates a non-contiguous view!
        img_disp = np.transpose(img_disp, (1, 2, 0))
        h, w, c = img_disp.shape
        # Let's make sure the data is contiguous in memory (essential for QImage)
        img_disp = np.ascontiguousarray(img_disp)

        fmt = QImage.Format.Format_RGB888
        bytes_per_line = c * w
    else:
        # Mono (H, W)
        h, w = img_disp.shape
        # Let's make sure the data is contiguous
        img_disp = np.ascontiguousarray(img_disp)

        fmt = QImage.Format.Format_Grayscale8
        bytes_per_line = w

    # 3. Creating a QImage
    # PyQt6 requires data to be passed as bytes or voidptr.
    # img_disp.data is a memoryview, which sometimes fails.
    # We use img_disp.tobytes() or pass the array directly if PyQt supports it (but tobytes is safer for QImage(bytes,...))
    # HOWEVER: QImage only copies data if the correct constructor or .copy() is used.
    # To avoid garbage collection problems, we pass the data and keep a reference.
    qimg = QImage(img_disp.data, w, h, bytes_per_line, fmt)
    
    # IMPORTANT: qimg shares memory with numpy array.
    # If the array dies, qimg dies. Let's do a deep copy (.copy()) to separate it.
    return QPixmap.fromImage(qimg.copy())

# --- Preview Worker ---
class PreviewWorker(QObject):
    finished = pyqtSignal(object) # Returns the processed array
    error = pyqtSignal(str)
    progress_update = pyqtSignal(int, str)  # Signal to refresh the main bar

    def __init__(self, siril, img_data, params):
        super().__init__()
        self.siril = siril
        self.img_data = img_data
        self.params = params

    def run(self):
        try:
            # 1. Setup Model
            model_url = self.params['model_url']
            strength = self.params['strength']
            
            user_dir = self.siril.get_siril_userdatadir()
            models_dir = os.path.join(user_dir, "scunet_models")
            if not os.path.exists(models_dir): os.makedirs(models_dir)
            
            model_filename = os.path.basename(model_url)
            modelpath = os.path.join(models_dir, model_filename)

            # Hook function for download progress (Copied from ProcessingWorker)
            def download_progress_hook(block_num, block_size, total_size):
                if total_size > 0:
                    downloaded = block_num * block_size
                    percent = int((downloaded / total_size) * 100)
                    # Update every 2% for smoothness
                    if percent % 2 == 0: 
                        self.progress_update.emit(percent, f"Downloading Model: {percent}%")
                else:
                    self.progress_update.emit(0, "Downloading Model... (size unknown)")

            # If not, download it with progress updates
            if not os.path.isfile(modelpath):
                self.progress_update.emit(0, "Starting Download for Preview...")
                ssl._create_default_https_context = ssl._create_stdlib_context
                urllib.request.urlretrieve(model_url, modelpath, reporthook=download_progress_hook)
                self.progress_update.emit(100, "Download Complete.")
            
            # ZIP Management (UberSmooth)
            if zipfile.is_zipfile(modelpath):
                with zipfile.ZipFile(modelpath, 'r') as zip_ref:
                    # EXPLICITLY extract to the models folder
                    zip_ref.extractall(models_dir)
                    # Update the path to the extracted .pth file
                    modelpath = modelpath.replace(".zip", ".pth")

            # 2. Load Model
            device = get_device()

            # Log device info to Siril console
            if device.type == 'cuda':
                device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Unknown NVIDIA GPU"
                self.siril.log(f"Acceleration: CUDA ({device_name})", s.LogColor.GREEN)
            elif device.type == 'mps':
                self.siril.log("Acceleration: Apple Metal Performance Shaders (MPS)", s.LogColor.GREEN)
            elif device.type == 'xpu':
                self.siril.log("Acceleration: Intel XPU (Arc GPU detected)", s.LogColor.GREEN)
            else:
                self.siril.log("Acceleration: CPU (No GPU detected)", s.LogColor.GREEN)

            # Load the model from the correct path
            model = ModelLoader().load_from_file(str(modelpath)).eval().to(device)

            # 3. Process ROI (Auto Tile Size for safety, or fixed small???)
            # Takes the actual dimensions of the cropped image
            if self.img_data.ndim == 3:
                h_img, w_img = self.img_data.shape[1], self.img_data.shape[2]
            else:
                h_img, w_img = self.img_data.shape[0], self.img_data.shape[1]
            
            # If the ROI is small (e.g., 250px), use 250 as the tile_size.
            # If it is large (e.g., 1000px), use max 512.
            tile_size = min(512, h_img, w_img)
            self.siril.log(f"Tile Size {tile_size}px", s.LogColor.GREEN)
            
            # Let's use the existing core function
            processed_roi = process_image_buffer(self.img_data, model, device, strength, tile_size)
            
            self.finished.emit(processed_roi)

        except Exception as e:
            traceback.print_exc()
            self.error.emit(str(e))

class PreviewWindow(QWidget):
    # Signal to forward progress to main window
    progress_update = pyqtSignal(int, str)
    # Signal: Notify when window is closed
    window_closed = pyqtSignal()

    def __init__(self, siril, main_window_params):
        super().__init__()
        self.setWindowTitle(f"SCUNet Denoise - Preview & Blink - v{VERSION}")
        self.resize(600, 650)
        
        # --- Window always on top ---
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)

        self.siril = siril
        self.params = main_window_params # Dictionary with url, force, etc.
        
        self.original_data = None
        self.processed_data = None
        self.pixmap_orig = None
        self.pixmap_proc = None
        
        self.thread = None
        self.worker = None

        self.current_poly = None        # Siril Polygon Object
        self.current_poly_coords = None # List [x, y, w, h] of the last polygon drawn

        self.setup_ui()
        
        # Start the first fetch now
        self.fetch_and_process()

    def setup_ui(self):
        layout = QVBoxLayout(self)

        # Image Container
        self.lbl_image = QLabel("Processing Preview...")
        self.lbl_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_image.setStyleSheet("background-color: #222; border: 1px solid #444;")
        self.lbl_image.setMinimumSize(500, 500)
        layout.addWidget(self.lbl_image, 1) # Expandable

        # Controls
        ctrl_layout = QHBoxLayout()
        
        self.btn_update = QPushButton("Update ROI")
        self.btn_update.setToolTip("Update preview with current selection from Siril.")
        self.btn_update.clicked.connect(self.fetch_and_process)
        
        self.btn_blink = QPushButton("Hold to BLINK (Show Original)")
        self.btn_blink.setCursor(Qt.CursorShape.PointingHandCursor)
        # Blink Logic: Pressed = Original, Released = Processed
        self.btn_blink.pressed.connect(self.show_original)
        self.btn_blink.released.connect(self.show_processed)
        self.btn_blink.setEnabled(False)

        ctrl_layout.addWidget(self.btn_update)
        ctrl_layout.addWidget(self.btn_blink)
        layout.addLayout(ctrl_layout)
        
        lbl_info = QLabel("<i>Move selection in Siril window and click <b>Update ROI</b>.</i>")
        lbl_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(lbl_info)

    def fetch_and_process(self):
        """ Retrieves the pixels of the current selection, updates the overlay, and starts processing. """
        try:
            # 1. Retrieve the current selection coordinates [x, y, w, h]
            #    and force it to LIST for consistent comparisons
            raw_selection = self.siril.get_siril_selection()
            if raw_selection:
                selection = list(raw_selection)
            else:
                selection = []
            
            # If there is no valid selection (or it is empty), use the default 500x500 in the center
            if not selection or selection[2] <= 0 or selection[3] <= 0:
                self.lbl_image.setText("Invalid Selection. Creating default...")
                # Create center selection (usually handled by the main window, but let's try again here)
                shape = self.siril.get_image_shape() # (3, H, W)
                
                if len(shape) == 3:
                    H, W = shape[1], shape[2]
                else:
                    H, W = shape[0], shape[1]
                
                # Let's use the same 50px padding you set in the main window
                # to avoid crashing at the edges of the image.
                safe_size = min(500, W - 50, H - 50)

                roi_w, roi_h = safe_size, safe_size
                cx, cy = W // 2, H // 2
                # Calculate top-left coordinates
                x = max(0, cx - roi_w // 2)
                y = max(0, cy - roi_h // 2)
                
                # Set selection in Siril (standard selection rectangle)
                self.siril.set_siril_selection(x, y, roi_w, roi_h)
                selection = [x, y, roi_w, roi_h]    # It's already a list

                self.siril.log(f"Preview ROI set at x={x}, y={y} ({roi_w}x{roi_h})", s.LogColor.BLUE)

            # --- LOGIC to make selections square ---
            # Get user selection [x, y, w, h]
            x, y, w, h = selection
            
            # Find the smallest side
            square_size = min(w, h)
            
            # Create the square keeping the x,y origin
            final_selection = [x, y, square_size, square_size]
            
            # Apply square selection in Siril
            if final_selection != selection:
                self.siril.set_siril_selection(*final_selection)

            # Align the internal selection to the actual applied selection
            selection = final_selection

            # Update the polygon ONLY if the coordinates have changed or if it doesn't exist.
            # Compare with the saved coordinates (which are a list).
            # If they are different or if the polygon doesn't exist, regenerate.
            if self.current_poly is None or selection != self.current_poly_coords:
                # 1. Delete the old one if it exists
                if self.current_poly is not None:
                    try:
                        self.siril.overlay_delete_polygon(self.current_poly.polygon_id)
                    except Exception:
                        pass
                    self.current_poly = None

                # 2. Create the new
                try:
                    x, y, w, h = selection
                    poly = s.Polygon.from_rectangle((x, y, w, h), color=0x00FF0080, fill=False, legend="ROI Preview")
                    self.current_poly = self.siril.overlay_add_polygon(poly)
                    # self.siril.set_siril_selection(0, 0, 0, 0)
                    # Save the new coordinates for the next comparison
                    self.current_poly_coords = selection 
                    
                except AttributeError:
                    pass
            
            # 2. Get the pixels ONLY of the selected region using the 'shape' parameter
            # shape accepts a list [x, y, w, h]
            roi_data = self.siril.get_image_pixeldata(preview=False, shape=selection)
            
            if roi_data is None:
                raise ValueError("Could not retrieve pixel data from Siril.")

            #3. Store data (no need for manual slicing anymore, we already have the clipping)
            self.original_data = roi_data.copy()

            # Prepare original view (helper function handles flipping and normalization)
            self.pixmap_orig = numpy_to_qpixmap(self.original_data)
            
            #4. Start Thread Processing
            self.lbl_image.setText("Processing ROI...")
            self.btn_blink.setEnabled(False)
            self.btn_update.setEnabled(False)
            
            self.thread = QThread()
            self.worker = PreviewWorker(self.siril, self.original_data, self.params)
            self.worker.moveToThread(self.thread)
            
            self.thread.started.connect(self.worker.run)
            self.worker.finished.connect(self.on_process_done)
            self.worker.error.connect(self.on_process_error)

            # Connect the worker signal to the window signal
            self.worker.progress_update.connect(self.progress_update)
            
            self.worker.finished.connect(self.thread.quit)
            self.worker.finished.connect(self.worker.deleteLater)
            self.thread.finished.connect(self.thread.deleteLater)
            
            self.thread.start()

        except Exception as e:
            self.lbl_image.setText(f"Error fetching ROI:\n{e}")
            print(f"Preview Error: {e}")
            traceback.print_exc()

    def on_process_done(self, processed_array):
        self.processed_data = processed_array
        self.pixmap_proc = numpy_to_qpixmap(self.processed_data)
        
        self.show_processed()
        self.btn_blink.setEnabled(True)
        self.btn_update.setEnabled(True)

    def on_process_error(self, err_msg):
        self.lbl_image.setText(f"Processing Failed:\n{err_msg}")
        self.btn_update.setEnabled(True)

    def show_original(self):
        if self.pixmap_orig:
            self.lbl_image.setPixmap(self.pixmap_orig.scaled(self.lbl_image.size(), Qt.AspectRatioMode.KeepAspectRatio))

    def show_processed(self):
        if self.pixmap_proc:
            self.lbl_image.setPixmap(self.pixmap_proc.scaled(self.lbl_image.size(), Qt.AspectRatioMode.KeepAspectRatio))
    
    def resizeEvent(self, event):
        # Redraw the current image when resizing the window to fit
        if self.btn_blink.isDown():
            self.show_original()
        else:
            self.show_processed()
        super().resizeEvent(event)

    def closeEvent(self, event: QCloseEvent):
        # Clears polygons in Siril when closing the preview window
        try:
            self.siril.set_siril_selection(0, 0, 0, 0)  # Removes the active selection (rectangle)
            if self.current_poly is not None:
                self.siril.overlay_delete_polygon(self.current_poly.polygon_id)
            else:
                # Fallback: Clean up everything if we lost the reference
                self.siril.overlay_clear_polygons()

            self.siril.log("Preview Window (ROI) closed by User. Selection cleared.", s.LogColor.BLUE)
        except Exception as e:
            print(f"Error clearing selection: {e}")

        self.window_closed.emit() # Notify main windows that the preview window is closed
        super().closeEvent(event)

# --- Main GUI Class ---
class ScunetWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"SCUNet Denoise - v{VERSION}")

        # --- Window always on top ---
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)
        
        # --- Siril Connection ---
        # Initialize Siril connection
        self.siril = s.SirilInterface()
        try:
            self.siril.connect()
        except Exception:
            QMessageBox.critical(self, "Connection Error", "Connection to Siril failed. Make sure Siril is open and ready.")
            sys.exit(1)

        try:
            self.siril.cmd("requires", "1.4.0")
            
            # TODO: Uncomment when Siril 1.4.1 is released
            #self.siril.cmd("requires", "1.4.1")  # For selection support when sequence is loaded
        except s.CommandError:
            sys.exit(1)
        
        self.siril.set_siril_selection(0, 0, 0, 0)  # Clear any existing selection

        # Check if an image or sequence is loaded
        image_loaded = self.siril.is_image_loaded()
        seq_loaded = self.siril.is_sequence_loaded()

        if seq_loaded:
            self.siril.log("Context: Sequence loaded.", s.LogColor.BLUE)
            self.is_sequence_context = True
        elif image_loaded:
            self.siril.log("Context: Single image loaded.", s.LogColor.BLUE)
            self.is_sequence_context = False
        else:
            self.siril.error_messagebox("No image or sequence loaded")
            sys.exit(1)

        self.preview_poly = None    # Initialize the variable to draw the preview polygon

        # --- GUI Setup ---
        self.setup_ui()
        self.thread = None
        self.worker = None
        self.center_window()

    def center_window(self):
        """ Center window using PyQt methods """
        screen_geometry = self.screen().availableGeometry()
        self.resize(400, 500)
        self.move(
            int((screen_geometry.width() - self.width()) / 2),
            int((screen_geometry.height() - self.height()) / 2)
        )

    def setup_ui(self):
        layout = QVBoxLayout(self)

        # --- CREDITS HEADER ---
        lbl_credits = QLabel(
            "<span style='color:#f4d742;'><b>Original version by Nicolas CASTEL</b></span><br>"
            "Refactoring by Carlo Mollicone AstroBOH"
        )
        lbl_credits.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_credits.setStyleSheet("color: #888; font-size: 12px; margin-bottom: 5px;")
        layout.addWidget(lbl_credits)

        # Separator line
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(line)

        # --- INSTRUCTIONS HEADER ---
        # Horizontal container
        instr_container = QWidget()
        instr_layout = QHBoxLayout(instr_container)
        instr_layout.setContentsMargins(0, 0, 0, 0)
        instr_layout.setSpacing(8)

        # Warning icon
        icon_label = QLabel()
        icon = instr_container.style().standardIcon(QStyle.StandardPixmap.SP_MessageBoxWarning)
        icon_label.setPixmap(icon.pixmap(20, 20))  # dimensione icona
        icon_label.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Vertical container to prevent the text from getting too wide
        text_container = QWidget()
        text_layout = QVBoxLayout(text_container)
        text_layout.setContentsMargins(0, 0, 0, 0)
        text_layout.setSpacing(0)

        # Text
        lbl_instructions = QLabel(
            "<span style='color:#d0d0d0;'><b>SCUNet denoiser</b></span> works best on fully processed and stretched non-linear images.<br>"
            "Run <span style='color:#d0d0d0;'><b>SCUNet denoiser</b></span> as the last step before publishing."
        )
        lbl_instructions.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        lbl_instructions.setStyleSheet("color: #888; font-size: 11px; margin-bottom: 0px;")
        lbl_instructions.setWordWrap(True)

        # Adding to layout
        text_layout.addWidget(lbl_instructions)
        instr_layout.addWidget(icon_label, 0)
        instr_layout.addWidget(text_container, 1)

        layout.addWidget(instr_container)

        # Separator line
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(line)

        # 1. Model Selection (Radio Buttons)
        gb_model = QGroupBox("Model Selection")
        layout_model = QVBoxLayout()
        
        self.model_buttons = [] # Store buttons to access them later
        
        # Create radio buttons from models_list
        for i, m in enumerate(models_list):
            rb = QRadioButton(m[0])
            rb.setToolTip(m[2]) # Description as tooltip
            
            # Store the URL as a property of the button
            rb.setProperty("url", m[1])
            
            if i == 0: rb.setChecked(True) # Select first by default
            
            layout_model.addWidget(rb)
            self.model_buttons.append(rb)
        
        gb_model.setLayout(layout_model)
        layout.addWidget(gb_model)

        # 2. Parameters (Strength + Tile Size)
        gb_params = QGroupBox("Parameters")
        layout_params = QVBoxLayout()
        
        # --- Strength Slider ---
        # Horizontal layout for slider and value label
        h_slider = QHBoxLayout()
        self.slider_strength = QSlider(Qt.Orientation.Horizontal)
        self.slider_strength.setRange(0, 100) # 0 to 100 for float 0.0 - 1.0
        self.slider_strength.setValue(50)     # Default 0.5

        self.lbl_strength_val = QLabel("0.50")

        # Connect slider change to update label
        self.slider_strength.valueChanged.connect(
            lambda val: self.lbl_strength_val.setText(f"{val/100:.2f}")
        )
        
        h_slider.addWidget(QLabel("Strength:"))
        h_slider.addWidget(self.slider_strength)
        h_slider.addWidget(self.lbl_strength_val)
        layout_params.addLayout(h_slider)

        # --- Tile Size Combo ---
        h_tile = QHBoxLayout()
        self.combo_tile = QComboBox()
        # Options: Auto and manual
        self.combo_tile.addItems(["Auto", "512", "384", "256", "128"])
        self.combo_tile.setToolTip("Tile size for processing.\n'Auto' tests VRAM to find the best size.\nLower values save memory but may be slower.")
        
        h_tile.addWidget(QLabel("Tile Size:"))
        h_tile.addWidget(self.combo_tile)
        layout_params.addLayout(h_tile)

        gb_params.setLayout(layout_params)
        layout.addWidget(gb_params)

        # 3. Sequence Options
        gb_seq = QGroupBox("Sequence Options")
        layout_seq = QVBoxLayout()
        
        self.chk_seq = QCheckBox("Process Sequence")
        
        self.layout_prefix = QHBoxLayout()
        self.lbl_prefix = QLabel("Output Prefix:")
        self.txt_prefix = QLineEdit("scunet_")
        
        # CONTEXT-BASED GUI FORCING LOGIC
        if self.is_sequence_context:
            # If it is a sequence: Checkbox active, checked and NOT editable
            self.chk_seq.setChecked(True)
            self.chk_seq.setEnabled(False) # User can't remove it
            self.chk_seq.setText("Process Sequence (Detected)")
            self.txt_prefix.setEnabled(True) # Prefix enabled
        else:
            # If it is a single image: Checkbox off and NOT editable
            self.chk_seq.setChecked(False)
            self.chk_seq.setEnabled(False) # User can't put it
            self.chk_seq.setText("Process Sequence (Single Image Mode)")
            self.txt_prefix.setEnabled(False) # Prefix disabled

        layout_seq.addWidget(self.chk_seq)
        
        self.layout_prefix.addWidget(self.lbl_prefix)
        self.layout_prefix.addWidget(self.txt_prefix)
        layout_seq.addLayout(self.layout_prefix)

        gb_seq.setLayout(layout_seq)
        layout.addWidget(gb_seq)

        # 4. Progress
        self.lbl_status = QLabel("Ready")
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        layout.addWidget(self.lbl_status)
        layout.addWidget(self.progress_bar)

        layout.addSpacing(15)

        # 5. Buttons
        btn_layout = QHBoxLayout()

        preview_icon = self.style().standardIcon(QStyle.StandardPixmap.SP_FileDialogDetailedView)
        self.btn_preview = IconTextButton("Open ROI Preview", preview_icon, self)
        self.btn_preview.setProperty("class", "ROIButton")
        self.btn_preview.setToolTip(
            "Open a preview window to test denoise on a small area.\n"
            "If the window is already open, click again to update the preview with the new settings."
        )
        self.btn_preview.clicked.connect(self.open_preview)
        
        btn_layout.addWidget(self.btn_preview)

        # Flexible space to separate left/right buttons
        btn_layout.addStretch()

        # Icon: Standard check mark
        apply_icon = self.style().standardIcon(QStyle.StandardPixmap.SP_DialogApplyButton)
        self.btn_apply = IconTextButton("Apply", apply_icon, self)
        self.btn_apply.setProperty("class", "accent")
        self.btn_apply.clicked.connect(self.start_processing)
        
        btn_layout.addWidget(self.btn_apply)

        # 3. Close Button (Standard, a destra)
        close_icon = self.style().standardIcon(QStyle.StandardPixmap.SP_DialogCancelButton)
        self.btn_close = IconTextButton("Close", close_icon, self)
        self.btn_close.setProperty("class", "secondary")
        self.btn_close.clicked.connect(self.close)
        btn_layout.addWidget(self.btn_close)

        layout.addLayout(btn_layout)

    def toggle_seq_options(self, checked):
        self.txt_prefix.setEnabled(checked)

    def open_preview(self):
        # if not self.siril.is_image_loaded():
        #     QMessageBox.warning(self, "Warning", "No image loaded to preview.")
        #     return

        # 1. Retrieve current parameters from the GUI
        selected_url = ""
        for rb in self.model_buttons:
            if rb.isChecked():
                selected_url = rb.property("url")
                break
        strength_val = self.slider_strength.value() / 100.0
        
        preview_params = {
            'model_url': selected_url,
            'strength': strength_val,
            # Tile size for preview is handled internally by PreviewWindow
        }

        # 2. Create smart selection based on image size (Redundant with the preview window)
        # try:
        #     # Get image size
        #     dims = self.siril.get_image_shape() # Returns (C, H, W) or (H, W)
            
        #     if len(dims) == 3:
        #         H, W = dims[1], dims[2]
        #     else:
        #         H, W = dims[0], dims[1]

        #     # The ROI will be 500 OR the minimum image size if it is smaller
        #     safe_size = min(500, W - 50 , H - 50) # Leave some margin
            
        #     roi_w, roi_h = safe_size, safe_size
        #     cx, cy = W // 2, H // 2
            
        #     # Calculate top-left coordinates
        #     x = max(0, cx - roi_w // 2)
        #     y = max(0, cy - roi_h // 2)
            
        #     # Set selection in Siril (standard selection rectangle)
        #     self.siril.set_siril_selection(x, y, roi_w, roi_h)
        #     self.siril.log(f"Preview ROI set at x={x}, y={y} ({roi_w}x{roi_h})", s.LogColor.BLUE)
            
        # except Exception as e:
        #     print(f"Error setting selection: {e}")
        #     # We don't block, we try to open anyway

        #3. Open Window
        if not hasattr(self, 'preview_window') or not self.preview_window.isVisible():
            self.btn_preview.setText("Update ROI Preview") # Change text when open
            self.preview_window = PreviewWindow(self.siril, preview_params)

            # --- Link preview update to main toolbar ---
            self.preview_window.progress_update.connect(self.update_progress)

            # --- Connect the closure to the text reset ---
            self.preview_window.window_closed.connect(self.reset_preview_button)
            self.preview_window.show()
        else:
            # If it's already open, just update the parameters and run the fetch
            self.preview_window.params = preview_params
            self.preview_window.fetch_and_process()
            self.btn_preview.setText("Update ROI Preview")  # To be safe, let's make sure the text is Update

    def reset_preview_button(self):
        """ Called when ROI closes to reset the button text. """
        self.btn_preview.setText("Open ROI Preview")

    def start_processing(self):
        # Clear any remaining preview overlay
        try:
            self.siril.overlay_clear_polygons()
            self.preview_poly = None # Reset variable
        except Exception:
            pass

        # 1. Get Selected Model URL
        selected_url = ""
        for rb in self.model_buttons:
            if rb.isChecked():
                selected_url = rb.property("url")
                break
        
        if not selected_url:
            QMessageBox.warning(self, "Warning", "Please select a model.")
            return

        # 2. Get Strength (Slider 0-100 -> Float 0.0-1.0)
        strength_val = self.slider_strength.value() / 100.0

        # 3. Read Tile Size
        tile_choice = self.combo_tile.currentText()
        # If it is "Auto", we pass None or "Auto", otherwise the integer
        tile_param = "Auto" if tile_choice == "Auto" else int(tile_choice)

        # Prepare Params
        params = {
            'model_url': selected_url,
            'strength': strength_val,
            'tile_size': tile_param,
            'is_sequence': self.chk_seq.isChecked(),
            'seq_prefix': self.txt_prefix.text()
        }

        # Setup Thread
        self.thread = QThread()
        self.worker = ProcessingWorker(self.siril, params)
        self.worker.moveToThread(self.thread)

        # Connect Signals
        self.thread.started.connect(self.worker.run)
        self.worker.progress_update.connect(self.update_progress)
        self.worker.error_occurred.connect(self.handle_error)
        self.worker.finished.connect(self.process_finished)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)

        # UI State
        self.btn_apply.setEnabled(False)
        self.btn_close.setEnabled(False)
        self.lbl_status.setText("Starting...")
        
        self.thread.start()

    def update_progress(self, val, text):
        self.progress_bar.setValue(val)
        self.lbl_status.setText(text)

    def handle_error(self, msg):
        QMessageBox.critical(self, "Processing Error", msg)
        self.lbl_status.setText("Error occurred.")
        self.btn_apply.setEnabled(True)
        self.btn_close.setEnabled(True)

    def process_finished(self):
        self.btn_apply.setEnabled(True)
        self.btn_close.setEnabled(True)
        self.lbl_status.setText("Completed.")
        # QMessageBox.information(self, "Info", "Processing Complete!")

    def closeEvent(self, event: QCloseEvent):
        """
        Handle dialog close - Called when the window is closed via the 'X' button.
        Close the dialog and disconnect from Siril
        """
        # Close the preview window if it exists
        if hasattr(self, 'preview_window') and self.preview_window.isVisible():
            self.preview_window.close()

        # Stop the worker if it is running
        if self.worker:
            self.worker.stop()
        
        # Check the thread safely
        if self.thread:
            try:
                if self.thread.isRunning():
                    self.thread.quit()
                    self.thread.wait()
            except RuntimeError:
                # Thread has already been deleted (C++ object deleted), ignore.
                pass
        
        # Disconnect Siril
        try:
            if self.siril:
                self.siril.set_siril_selection(0, 0, 0, 0)  # Removes the active selection (rectangle)
                self.siril.overlay_clear_polygons()         # Removes graphic overlays (green polygon)
                self.preview_poly = None # Reset variabile
                self.siril.log("Window closed. Script cancelled by user.", s.LogColor.BLUE)
                self.siril.disconnect()
        except Exception as e:
            print(f"An error occurred during cleanup: {e}")
                
        event.accept()

class IconTextButton(QPushButton):
    """
    A custom QPushButton that uses an internal layout and has a proper
    size policy to behave correctly within toolbars and other layouts.
    """
    def __init__(self, text, icon=None, parent=None):
        super().__init__(parent)
        
        # Set the size policy to match a standard button.
        # This prevents the button from expanding greedily and compressing others.
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)

        # The internal layout for aligning the icon and text
        layout = QHBoxLayout()
        layout.setContentsMargins(8, 0, 8, 0) # left, top, right, bottom
        layout.setSpacing(5)

        # Icon Label
        icon_label = QLabel()
        if icon:
            pixmap = icon.pixmap(self.iconSize())
            icon_label.setPixmap(pixmap)
        
        # Text Label (Saved as self.text label for later updating!)
        self.text_label = QLabel(text)

        # Add widgets to the internal layout
        layout.addWidget(icon_label, 0, Qt.AlignmentFlag.AlignVCenter)
        layout.addStretch(1)
        layout.addWidget(self.text_label)
        layout.addStretch(1)
        
        self.setLayout(layout)

    def setText(self, text):
        """ Override standard setText to update the internal label. """
        super().setText(text) # Update the base button text (best practice)
        self.text_label.setText(text) # Update the VISIBLE label

# --- Main Execution Block ---
# Entry point for the Python script in Siril
# This code is executed when the script is run.
def main():
    try:
        qapp = QApplication(sys.argv)
        qapp.setApplicationName(f"SCUNet Denoise - v{VERSION}")

        icon_data = base64.b64decode("""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""")
        pixmap = QPixmap()
        pixmap.loadFromData(icon_data)
        app_icon = QIcon(pixmap)
        qapp.setWindowIcon(app_icon)

        qapp.setStyle("Fusion")

        # Define a Qt Style Sheet (QSS)
        stylesheet = """
            QPushButton[class="accent"] {
                background-color: #3574F0;  /* A nice blue color */
                color: white;
                font-weight: bold;
                border-radius: 4px;
                padding: 5px;
                min-width: 90px;
            }
            QPushButton[class="accent"]:hover {
                background-color: #4E8AFC; /* A slightly lighter blue for hover */
            }

            QPushButton[class="secondary"] {
                background-color: #e95767; /* Dark gray for secondary actions */
                color: #dddddd;
                font-weight: bold;
                border-radius: 4px;
                padding: 5px;
                min-width: 90px;
            }
            QPushButton[class="secondary"]:hover {
                background-color: #d64e5d; /* Slightly lighter on hover */
                color: white;
            }

            QPushButton[class="ROIButton"] {
                background-color: #f0f0f0; /* Light gray */
                border-radius: 4px;
                padding: 5px;
                min-width: 150px;
            }
            QPushButton[class="ROIButton"]:hover {
                background-color: #e0e0e0; /* Darker on hover */
            }
            
            /* Style specifically for the TEXT inside the CUSTOM HELP button */
            QPushButton[class="ROIButton"] QLabel {
                color: #005A9C; /* A professional dark blue for readability */
                font-weight: bold;
                background-color: transparent; /* Ensure label background is clear */
                border: none;
            }
        """
        # Apply the stylesheet to the entire application
        qapp.setStyleSheet(stylesheet)

        app = ScunetWindow()
        app.show()
        
        sys.exit(qapp.exec())

    except Exception as e:
        print(f"Error initializing application: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()