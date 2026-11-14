# Siril-SmartScope-Scripts

## Exporting SCUNet to ONNX

`exportScunetToONNX.py` exports the SCUNet model weights used by
`SCUNet_DenoiseV2.py`; it does not convert the Siril GUI, image preprocessing,
or tiled blending workflow.

Run the exporter from Siril's Python environment, where it can install the
required ONNX packages:

```text
python exportScunetToONNX.py
```

With no checkpoint argument, it connects to the running Siril instance and
loads `scunet_color_real_psnr.pth` from Siril's `scunet_models` data folder.
To export another Spandrel-compatible checkpoint, pass its path:

```text
python exportScunetToONNX.py path\to\scunet_model.pth --output-dir path\to\output
```

By default, the exporter writes `<checkpoint>_fp16.onnx` only. Use
`--precision fp32` to write the FP32 model instead, or `--precision both` to
write both variants. FP16 conversion keeps the model's input and output tensors
in FP32. The export input is NCHW with one or three channels; height and width
default to 256 and must be positive multiples of 64. The ONNX graph has fixed
spatial dimensions by default for DirectML compatibility. Add `--dynamic-shapes`
to export dynamic batch, height, and width axes for CPU inference instead.

When running outside Siril, pass the checkpoint path and install compatible
`torch`, `spandrel`, `onnx`, and `onnxconverter-common` packages in the active
Python environment.

## Denoising with an exported ONNX model

`SCUNet_ONNX_Denoise.py` applies an exported model to the image currently loaded
in Siril. It uses ONNX Runtime's DirectML provider and NumPy, which it asks
Siril to install when needed. By default it loads
`scunet_models\scunet_color_real_psnr_fp16.onnx` from Siril's user data folder:

```text
pyscript SCUNet_ONNX_Denoise.py --strength 0.8 --tile-size 128 --provider directml
```

To select another exported model or process a linear image:

```text
pyscript SCUNet_ONNX_Denoise.py --model path\to\scunet_model_fp16.onnx --pre-stretch
```

Use the FP32 model instead of the default FP16 model with:

```text
pyscript SCUNet_ONNX_Denoise.py --precision fp32
```

The script updates the loaded image in place as an undoable Siril operation;
save the result with Siril when desired. It accepts mono or RGB images loaded
by Siril, preserves the input pixel type and image dimensions, and uses
overlapping tiles with weighted blending. `--strength` ranges from 0 (unchanged)
to 1 (full denoise); `--tile-size` accepts 64, 128, 256, 384, or 512 pixels.
Fixed shape models use the remaining model dimensions for tile context; dynamic
CPU models use 144 pixels of context and pad to SCUNet's 64-pixel size multiple.
`--pre-stretch` enables the same GHS prestretch/destretch used for linear data
in `SCUNet_DenoiseV2.py`; leave it off for images that are already stretched.

The exporter produces single-input, single-output NCHW models with FP32 input
and output (including the default FP16-weight variant). DirectML requires the
`onnxruntime-directml` package on Windows, fixed spatial model dimensions, and
sequential execution. The exporter creates fixed-shape models by default;
re-export existing dynamic models before using DirectML by running
`exportScunetToONNX.py` again without `--dynamic-shapes`. Tile size must not
exceed the exported model's smaller spatial dimension. The script refuses to
silently switch to CPU if DirectML is unavailable. Use `--provider cpu` for
either fixed- or dynamic-shape models instead. A mono model cannot process an
RGB image. This script processes one loaded image only; it does not provide the
original script's GUI, sequence processing, or preview workflow. CPU inference
on large full-resolution images can take a long time. During processing, Siril's
console and progress status report each tile's ONNX inference time and the
cumulative model-inference and total processing times; no trace file is written.