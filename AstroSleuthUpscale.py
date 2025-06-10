# -*- coding: UTF-8 -*-
# ==============================================================================

# ------------------------------------------------------------------------------
# Project: Python siril script to run AstroSleuth Upscaler
# using model from https://github.com/Aveygo/AstroSleuth
#
# ------------------------------------------------------------------------------
#    Author:  Nicolas CASTEL <nic.castel (at) gmail.com>
#    Author:  Gregory Taylor <gregory.taylor (at) gmail.com>
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

import sys, os, asyncio, urllib.request, ssl, math, base64
import sirilpy as s
from sirilpy import tksiril

s.ensure_installed("numpy")
import numpy as np

s.ensure_installed("ttkthemes")
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from ttkthemes import ThemedTk

s.ensure_installed("torch")
import torch
from torch.nn import functional as F
from torch import nn as nn

# ------------------------------------------------------------------------------
# Config
# ------------------------------------------------------------------------------

VERSION = "1.0.1"
MODEL_SRC = "https://github.com/Aveygo/AstroSleuth/releases/download/v4/AstroSleuthNEXT.pth"

# ------------------------------------------------------------------------------
# Neural Network Components
# ------------------------------------------------------------------------------

class LayerNorm(nn.Module):
    """Custom Layer Normalization module, taken from ConvNeXtv2

    Args:
        normalized_shape (int): The shape of the input dimension to normalize.
        eps (float, optional): Small value to avoid division by zero. Defaults to 1e-6.
        data_format (str, optional): Format of input data, either 'channels_last' or 'channels_first'.
            Defaults to 'channels_last'.
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            # layernorm not implemented in ncnn...
            # return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
            mean = x.mean(dim=-1, keepdim=True)
            var = x.var(dim=-1, unbiased=False, keepdim=True)
            x = (x - mean) / torch.sqrt(var + self.eps)
            x = x * self.weight + self.bias
            return x

        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class GRN(nn.Module):
    """Global Response Normalization module, taken from ConvNeXtv2

    Args:
        dim (int): The dimension of the input tensor to normalize.
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

class Block(nn.Module):
    """Primary ConvNeXtV2 block

    Args:
        dim (int): Number of input and output channels.
        drop_path (float, optional): Drop path rate (not used). Defaults to 0.
    """
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)

    def forward(self, x):
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        return x

class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat=64, num_grow_ch=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x

class RRDB(nn.Module):
    """Residual in Residual Dense Block.

    Args:
        num_feat (int): Number of feature channels.
        num_grow_ch (int, optional): Number of growth channels. Defaults to 32.
    """
    def __init__(self, num_feat, num_grow_ch=32):
        super(RRDB, self).__init__()
        self.num_feat = num_feat

        # NOTE: Does not play nice with NCNN - maybe GRN norm?
        self.block = Block(num_feat)
        self.l1 = nn.Linear(num_feat, num_feat)

        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x:list):
        """Forward pass through the RRDB.
        Arguments are passed as a list as pytorch sequential layers don't support multi-argument passing.

        Args:
            x (list): List containing [input tensor, condition tensor, condition strength].

        Returns:
            list: [Output tensor, condition tensor, condition strength].
        """
        x, c, s = x[0], x[1], x[2]

        if not c is None:
            condition = self.l1(c).view(-1, self.num_feat, 1, 1)
            features = self.block(x + condition) * 0.1 * s
        else:
            features = torch.zeros_like(x)

        out = self.rdb1(x + features)
        out = self.rdb2(out + features)
        out = self.rdb3(out + features)

        return [out * 0.2 + x, c, s]

def make_layer(basic_block, num_basic_block, **kwarg):
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)

def pixel_unshuffle(x, scale):
    b, c, hh, hw = x.size()
    out_channel = c * (scale**2)
    assert hh % scale == 0 and hw % scale == 0
    h = hh // scale
    w = hw // scale
    x_view = x.view(b, c, h, scale, w, scale)
    return x_view.permute(0, 1, 3, 5, 2, 4).reshape(b, out_channel, h, w)

class Network(nn.Module):
    """Main neural network for AstroSleuth image upscaling.

    Args:
        num_in_ch (int, optional): Number of channels within the input image.
        num_out_ch (int, optional): Number of output channels for the output image.
        scale (int, optional): Upscale factor (1, 2, or 4).
        num_feat (int, optional): Number of feature channels.
        num_block (int, optional): Number of RRDB blocks.
        num_grow_ch (int, optional): Number of growth channels in RDB.
    """
    def __init__(self, num_in_ch=3, num_out_ch=3, scale=4, num_feat=64, num_block=6, num_grow_ch=32):
        super(Network, self).__init__()
        self.scale = scale
        self.num_feat = num_feat

        if scale == 2:
            num_in_ch = num_in_ch * 4
        elif scale == 1:
            num_in_ch = num_in_ch * 16

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(RRDB, num_block, num_feat=num_feat, num_grow_ch=num_grow_ch)
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.l1 = nn.Linear(512, num_feat)

        # AstroSleuth was trained with CLIP embeddings
        # However, because CLIP is quite heavy, the following embeddings were precalculated for end-user convenience
        self.average = self.bytes2torch(b'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')
        self.detail = self.bytes2torch(b'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')
        self.stars = self.bytes2torch(b'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')
        self.spikes = self.bytes2torch(b'Ol1BvpgZHz6+B8s92JujvV5CKr9jRBe/ACBCPVhGWT/AMEi/5XWrvudjOL86khY/s5wZv4qSzb/4FYQ/HzI/vuU43L5U1sW+HpzePqVaEr7Ov4U+vMGWvj7e9j0MXLA82N56PuFiAz9Y6aQ+rJC0vt2OTr97Qxs+8JdfvoJ8+b5q9+M9lQrCvvpROr60BDG9Zgu1vlZxQj2tOK4+AIDuNjTzFb61J/E+EY0Zvyymj770G7s/jS9kvjRGOD54xQs+BtgIv1Ovob8jMLu+8ubQPvt7AT+M9jO+JpRvP3v6fT7CFpE/KaeRPXQMYT9jAYi+7MAzv4xkA718qA8/mPquPQMXKz82CTQ+Mv+bPTIcTr9POSM/fJIyP9BOUL5wAV+9yMZwv5xshj68Zge/7Godvf4AC7/5jmW+JI1XPdeHnD67JmM+J0ZPv3smJT/CG46+9CInPdF6PD7qMos+43XhviDT1z44bZC8kvbavjAePbxoyHs+GG8EvwBDCLu8+Zg/M0LdvsI/f7/ETek9ufLGvfiq5L2Xv40/Yl2VP3cOCz/E1xK9FB77PjqZLj/mXRi/4FeovfstrL68zWu+8oBjP9qOAj8CXEE/Bv8tPsCNmzzA8ys/8gaYPoQBBT5uVq297Fo2PspMk75pg/4+oKyEvmshwL7spzA/XNUnvTJ5nz+EQ2k9AK7qPCSBlb4fzIq+RCa+vMaS+j7ATJu/e+PePfwjkT4n+/6938wGPjHsKz8tz7q+Mjm1vhLcvj5IptI8oFRsvEprJT4QJom8thY6v8X5Oz9TOD4+hFoRwP6py75Q/oQ9TgsAP49qqD7wVSE/nv2hPljRmb90MQy+AhgzvwTm+j4tzF8+6CMDv+N6fz9LPRy/sG5GPz3tzz58hMW+v/ICv6is5z5zMgA/8g++vgo4S0A2lQK/IB7MvgTjPz4pbpI/TlK/Pt9hND7VhOo9RbIgPj1Ax70LvWq/3WSJvvTxHD2040Q/yYOEPuAqGrx8TRK9AzhIv0vu6r4ktPS+AHXxO8iBAD+wfrC9thClvsi+cj44zTk+1qURPkIM9T4qnRA/UHBcP1AGAz8kTro8Jn0VP3SfRb9QiIO+9lsdP3KRbb4QMOg+wKEQP3gPIT/2aT0+P3FCvnWhWj3bB0i/ueqaP+CCMT3wDVg+JAylveW85z7QjVE9HnQav2H9hT4JRsU+EqqfvW50FT7PJD2+lNS/vdSNPT8eqre+QM20PMqv6j3COhQ/JnvcPhGauT75awQ/VgiKvqNCGD+KxB4+os6UvlsGX77oggy/JjyBP+hOzz6bxhM+N5PBvrNHAD5u88Y+1UvbvsJkM74c7ps+wCEqPUGPnj9g1+u+IZKqPsRpuj41kJS+MZIpv+XDND5g0h49ujWKPvvOzT+GNJW//qXxPS1vDj+QINY+vrI0Pmapmj6V+X6+52emvrIvaj8HeYO/VHp5PuhiO7657KO+aFxvPdFWSD6Blii/JKj3vdRAtL+IeLQ86e8TP8JQjL5k9Py+9A0tvvJFVD+iLmq/APEyuurAJ79k3jY9Tsh2P2NoSL8U9Jm947eHv1DC5Ty0n9S+pGGzvnO8MT49E/E/T+UtvvpCAz6UaYC/HJd5PoCwSr0kkxQ/W9U/P27rN72cJBs9OILlPjqOd78ouv69AFN4viJkFb76UyI+Dq0JvmC0Gr0aAIO/YMS2Pe4kvr6fl3S/XwpePj5IBz9UqIY+PIfCvYT+9r5wQtI+9kJgvvBZWz9uSgY/KGMfP4zESr8QnxS/Rs/KvoIQEr9Rr64+PJYlv3AdlrwM4Vk+4ty7P/jCo74t6jK/BPNLP8Y8KD+gIQU/4Kapvfgajj2Ct4499ECZv46t/b0Gizi/2F6ovoJwq74SM1S/QsMCP2hEXL704JM/j7XUvYCayj3/THc+Cdgmv/Gfb7/1jCLA+kHXvn0X4b7ifNK+JmRfv6hgBL3Oe8M+NodIP4N9MD/uxCs/5ONKP73M4z4Q9+G+GO67vH9MIr7GDfs+7q8KPV6Eg72WMGg/BainvL0A+D6bNwc/AGINuxSMgD9cC4c/zb9NP0CzZLsLK6C+YZ+GvngIWD9+pVy+mopDP70RsD+oLCg/2Ks3vY4dCL7MXjY+wFWwPmGJmL5Ykvc+gJh1Px7bOT5E9J+8dLImvdgwoj+QxgM9CIQxvrkTMT+2QYc+phh5PrC9Lj88n7u/TPMrP3augL6Huja/1rk6PjAyFL/QtoS/evT8PueZWD6AnAQ/v44cPgjmp71o4HS98Kx7Ps5Inj4WbwK+fAXXvgBJeb+gyZa/DdsMvmE2Q79jIx4+/OmivjkSAD4JZwu/NC1svyQUS765F9o+hB9wv3P6I75wBpM/bgtAv12+Db8iLNG+3ioUvvgtHL/NVD0/b3Uqv6Ce8j04WAs/WvYXPxEqQb8qW7Y/uN6evn436r3wwwo/YdU+v0v3AL/weTy/PFscPj0dxb2UD0O/CCuwPvh9NL8q3BO/JcJZP849tj6atq+9DAIkvlB9vLx1b7a+VM4Fv7wQtb1uhfo92KGJvgRpCD8b088+KsakPTeZtL44AHk/1KdBvi29ED+4jpw+mLLmvT27WT781Xg9eZY2P05Ac79NZQA/jJv5PVBJS78Q0w69Qrj9vv0pPL+dmww/OUyYPrk6sr7afb8+YM0fv6APfj2aI3E+sMAvvyz2Bb90LT6/J07VPvBlWj2gVJO+5EWEv1Zx4T0=')

    def bytes2torch(self, x:bytes):
        x = np.frombuffer(base64.decodebytes(x), dtype=np.float32).copy()
        return torch.from_numpy(x)

    def rgb_to_ycbcr(self, image: torch.Tensor) -> torch.Tensor:
        r: torch.Tensor = image[..., 0, :, :]
        g: torch.Tensor = image[..., 1, :, :]
        b: torch.Tensor = image[..., 2, :, :]
        y: torch.Tensor = .299 * r + .587 * g + .114 * b
        cb: torch.Tensor = (b - y) * .564 + .5
        cr: torch.Tensor = (r - y) * .713 + .5
        return y, cb, cr

    def ycbcr_to_rgb(self, y: torch.Tensor, cb: torch.Tensor, cr: torch.Tensor) -> torch.Tensor:
        r: torch.Tensor = y + 1.403 * (cr - 0.5)
        g: torch.Tensor = y - 0.714 * (cr - 0.5) - 0.344 * (cb - 0.5)
        b: torch.Tensor = y + 1.773 * (cb - 0.5)
        rgb_image: torch.Tensor = torch.stack([r, g, b], dim=-3)
        return torch.clamp(rgb_image, 0, 1)

    def color_matching(self, src, trg, match_strength=1):
        """Matches the colors of the source image to the target image.
        (This version of AstroSleuth was not trained for realism at the cost of accuracy)

        Args:
            src (torch.Tensor): Source RGB image tensor.
            trg (torch.Tensor): Target RGB image tensor.
            match_strength (float, optional): Strength of color matching (0 to 1). Defaults to 1.

        Returns:
            torch.Tensor: Target image with matched colors.
        """
        src_y, src_cb, src_cr = self.rgb_to_ycbcr(src)
        trg_y, trg_cb, trg_cr = self.rgb_to_ycbcr(trg)
        src_cb = F.interpolate(src_cb.view(1, 1, src.shape[2], src.shape[3]), scale_factor=4, mode='bilinear').view(1, src.shape[2]*4, src.shape[3]*4)
        src_cr = F.interpolate(src_cr.view(1, 1, src.shape[2], src.shape[3]), scale_factor=4, mode='bilinear').view(1, src.shape[2]*4, src.shape[3]*4)
        trg_cb = trg_cb * (1-match_strength) + src_cb * match_strength
        trg_cr = trg_cr * (1-match_strength) + src_cr * match_strength
        trg = self.ycbcr_to_rgb(trg_y, trg_cb, trg_cr)
        return trg

    def forward(self, x, star_strength=0, detail_strength=0, spikes_strength=0, use_cond=False, cond_strength=1, color_matching=False, scale=4):
        """Forward pass through the upscaling network.

        Args:
            x (torch.Tensor): Input image tensor.
            star_strength (float, optional): Strength of star enhancement. Defaults to 0.
            detail_strength (float, optional): Strength of detail enhancement. Defaults to 0.
            spikes_strength (float, optional): Strength of spike enhancement. Defaults to 0.
            use_cond (bool, optional): Whether to use conditional input. Defaults to False.
            cond_strength (float, optional): Strength of conditional input. Defaults to 1.
            color_matching (bool, optional): Whether to apply color matching. Defaults to False.
            scale (float, optional): Upscale factor. Defaults to 4.

        Returns:
            torch.Tensor: Upscaled image tensor.
        """
        c = self.average + self.stars * star_strength + self.detail * detail_strength + self.spikes * spikes_strength
        c = self.lrelu(self.l1(c.to(x.device).float())) if use_cond else None

        x_primary = self.conv_first(x)

        skip = self.conv_body(self.body([x_primary, c, cond_strength])[0])
        x_primary = x_primary + skip

        x_primary = self.lrelu(self.conv_up1(F.interpolate(x_primary, scale_factor=2, mode='bilinear')))
        x_primary = self.lrelu(self.conv_up2(F.interpolate(x_primary, scale_factor=2, mode='bilinear')))

        out = self.conv_last(self.lrelu(self.conv_hr(x_primary)))
        if color_matching:
            out = self.color_matching(x, out)

        if scale == 4:
            return out

        return torch.nn.functional.interpolate(out, scale_factor=scale / 4, mode="bilinear", antialias=True)

# ------------------------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------------------------

def get_device() -> torch.device:
    """Determines the best available device for PyTorch computations.

    Returns:
        torch.device: The selected device ('cuda', 'mps', or 'cpu').
    """
    if torch.cuda.is_available():
        print("cuda acceleration used")
        return torch.device("cuda")
    if torch.backends.mps.is_available() :
        print("mps acceleration used")
        return torch.device("mps")
    else:
        print("cpu used")
        return torch.device("cpu")

def image_to_tensor(device: torch.device, img: np.ndarray) -> torch.Tensor:
    """Converts a NumPy image array to a PyTorch tensor on the specified device.

    Args:
        device (torch.device): Target device for the tensor.
        img (np.ndarray): Input image array.

    Returns:
        torch.Tensor: Tensor representation of the image.
    """
    tensor = torch.from_numpy(img)
    return tensor.to(device)

def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    """Converts a PyTorch tensor back to a NumPy image array.

    Args:
        tensor (torch.Tensor): Input tensor of shape (1, C, H, W).

    Returns:
        np.ndarray: Image array of shape (H, W, C).
    """
    return (np.rollaxis(tensor.cpu().detach().numpy(), 1, 4).squeeze(0)).astype(np.float32)

def image_inference_tensor(model: torch.nn.Module, tensor: torch.Tensor, args:dict) -> torch.Tensor:
    """Performs inference on an image tensor using the specified model.

    Args:
        model (torch.nn.Module): The neural network model.
        tensor (torch.Tensor): Input image tensor.
        args (dict): Model configuration parameters.

    Returns:
        torch.Tensor: Output tensor after model inference.
    """
    with torch.no_grad():
        return model(tensor, **args)

def tile_process(device: torch.device, model: torch.nn.Module, data: np.ndarray, model_config:dict, scale:int, tile_size:int, yield_extra_details:bool=False):
    """Processes an image in tiles to manage memory usage during upscaling.

    Args:
        device (torch.device): Device to run the model on.
        model (torch.nn.Module): The neural network model.
        data (np.ndarray): Input image array of shape (H, W, C).
        model_config (dict): Configuration parameters for the model.
        scale (int): Upscale factor.
        tile_size (int): Size of each tile.
        yield_extra_details (bool, optional): If True, yields additional tile information. Defaults to False.

    Yields:
        tuple or np.ndarray: Either the processed tile or a tuple containing (tile, x, y, w, h, progress) depending on yield_extra_details
    """

    tile_pad = 16

    data = np.rollaxis(data, 2, 0)
    data = np.expand_dims(data, axis=0)

    batch, channel, height, width = data.shape

    tiles_x = width // tile_size
    if tiles_x*tile_size < width:
        tiles_x+=1
    tiles_y = height // tile_size
    if tiles_y*tile_size < height:
        tiles_y+=1

    p_thresh = [i*10 for i in range(11)]

    for i in range(tiles_x * tiles_y):
        x = math.floor(i/tiles_y)
        y = i % tiles_y

        if x<tiles_x-1:
            input_start_x = x * tile_size
        else:
            input_start_x = width - tile_size
        if y<tiles_y-1:
            input_start_y = y * tile_size
        else:
            input_start_y = height - tile_size

        if input_start_x < 0 : input_start_x = 0
        if input_start_y < 0 : input_start_y = 0

        input_end_x = min(input_start_x + tile_size, width)
        input_end_y = min(input_start_y + tile_size, height)

        input_start_x_pad = max(input_start_x - tile_pad, 0)
        input_end_x_pad = min(input_end_x + tile_pad, width)
        input_start_y_pad = max(input_start_y - tile_pad, 0)
        input_end_y_pad = min(input_end_y + tile_pad, height)

        input_tile_width = input_end_x - input_start_x
        input_tile_height = input_end_y - input_start_y

        input_tile = data[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad].astype(np.float32)

        output_tile = image_inference_tensor(model, image_to_tensor(device, input_tile), model_config)
        progress = (i+1) / (tiles_y * tiles_x)
        p_progress = round(progress * 100, 2)

        if p_progress >= p_thresh[0]:
            print(f"Progress: {p_thresh.pop(0)}%")

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

# ------------------------------------------------------------------------------
# Siril GUI Application
# ------------------------------------------------------------------------------

class SirilAstroSleuth:
    def __init__(self, root):
        self.root = root
        self.root.title(f"AstroSleuth Upscale - v{VERSION}")
        self.root.resizable(True, True)

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

        try:
            self.siril.cmd("requires", "1.3.6")
        except s.CommandError:
            self.close_dialog()
            return

        tksiril.match_theme_to_siril(self.root, self.siril)

        self.sliders = {}
        self.value_labels = {}
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

        # Upscale Factor Dropdown
        upscale_frame = ttk.Frame(model_frame)
        upscale_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(upscale_frame, text="Upscale Factor:").pack(side=tk.LEFT)
        self.upscale_var = tk.StringVar(value="4x")
        upscale_dropdown = ttk.Combobox(
            upscale_frame,
            textvariable=self.upscale_var,
            values=["1x", "2x", "4x"],
            state="readonly",
            width=10
        )
        upscale_dropdown.pack(side=tk.LEFT, padx=5)

        # Slider parameters
        slider_params = [
            {"name": "Strength", "from_": 0, "to": 5, "default": 0.5},
            {"name": "Detail", "from_": -1, "to": 1, "default": 0.0},
            {"name": "Stars", "from_": -1, "to": 1, "default": 0.0}
        ]

        # Create sliders
        for param in slider_params:
            # Create LabelFrame for each slider
            name = param["name"].lower()
            frame = ttk.LabelFrame(main_frame, text=param["name"], padding=10)
            frame.pack(fill=tk.X, padx=5, pady=5)

            # Create slider
            self.sliders[name] = ttk.Scale(
                frame,
                from_=param["from_"],
                to=param["to"],
                value=param["default"],
                orient=tk.HORIZONTAL,
            )
            self.sliders[name].pack(fill=tk.X, padx=5, pady=5)

            # Create value label
            self.value_labels[name] = ttk.Label(
                frame,
                text=f"{param['name']}: {self.sliders[name].get():.2f}"
            )
            self.value_labels[name].pack(pady=5)

            # Bind slider movement to update label
            def update_label(event, slider_name=name):
                self.value_labels[slider_name].configure(
                    text=f"{slider_name.capitalize()}: {self.sliders[slider_name].get():.2f}"
                )

            self.sliders[name].bind("<B1-Motion>", update_label)
            self.sliders[name].bind("<ButtonRelease-1>", update_label)

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
        """Applies the upscaling process to the loaded image.

        Handles model downloading, image processing, and updating Siril with the upscaled image.
        """
        try:
            print("AstroSleuthUpscale:begin")

            self.siril.reset_progress()
            modelpath = os.path.join(self.siril.get_siril_userdatadir(), os.path.basename(MODEL_SRC))

            if os.path.isfile(modelpath):
                print("Model found: " + modelpath)
            else:
                ssl._create_default_https_context = ssl._create_stdlib_context
                urllib.request.urlretrieve(MODEL_SRC, modelpath)
                print("Model downloaded: " + modelpath)

            device = get_device()

            model = Network().eval().to(device)
            model.load_state_dict(torch.load(modelpath, map_location=device))

            self.siril.update_progress("Model initialised", 0.05)
            image = self.siril.get_image()

            # Convert pixel data to 32 bit if 16bit mode is used
            original_data = image.data
            original_dtype = original_data.dtype
            if original_dtype == np.uint16:
                pixel_data = original_data.astype(np.float32) / 65535.0
            else:
                pixel_data = original_data

            # Handle planar format (c, h, w) -> (h, w, c)
            pixel_data = np.transpose(pixel_data, (1, 2, 0))
            original_height, original_width, channels = pixel_data.shape
            img_max = np.max(pixel_data)
            img_min = np.min(pixel_data)

            pixel_data = np.interp(pixel_data, (img_min, img_max), (0, +1))

            print(f"Original_height: {original_height}, Original_width: {original_width}")

            tile_size = 256
            scale = int(self.upscale_var.get().rstrip("x"))

            # Allocate an image to save the tiles
            imgresult = np.zeros((original_height*scale,original_width*scale,channels), dtype=np.float32)

            model_config = {
                "star_strength": self.sliders["stars"].get(),
                "detail_strength": self.sliders["detail"].get(),
                "cond_strength": self.sliders["strength"].get(),
                "use_cond": not self.sliders["detail"].get() == 0,
                "scale": float(scale)
            }

            print(f"Running with config: {model_config}")

            for i, tile in enumerate(tile_process(device, model, pixel_data, model_config, scale, tile_size, yield_extra_details=True)):

                if tile is None:
                    break

                tile_data, x, y, w, h, p = tile
                if w != 0 and h != 0 :
                    imgresult[x*scale:x*scale+tile_size*scale,y*scale:y*scale+tile_size*scale] = tile_data

                self.siril.update_progress("Image upscale ongoing", p)

            # Convert back to planar format
            output_image = np.transpose(imgresult, (2, 0, 1))
            output_image = np.interp(output_image, (pixel_data.min(), pixel_data.max()), (img_min, img_max)).astype(np.float32)

            # Scale back if needed
            if original_dtype == np.uint16:
                output_image = output_image * 65535.0
                output_image = output_image.astype(np.uint16)

            self.siril.undo_save_state("AstroSleuth upscale")
            with self.siril.image_lock(): self.siril.set_image_pixeldata(output_image)

            self.siril.update_progress("Image upscaled", 1.0)

        except Exception as e:
            print(f"Error in apply_changes: {str(e)}")
            self.siril.update_progress(f"Error: {str(e)}", 0)
        finally:
            print("AstroSleuthUpscale:end")

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
