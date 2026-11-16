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
default to 384 and must be positive. The ONNX graph has fixed
spatial dimensions by default for DirectML compatibility. Add `--dynamic-shapes`
to export dynamic batch, height, and width axes for CPU inference instead.

When running outside Siril, pass the checkpoint path and install compatible
`torch`, `spandrel`, `onnx`, and `onnxconverter-common` packages in the active
Python environment.

## Denoising with an exported ONNX model

`SCUNet_ONNX_Denoise.py` applies an exported model to the image currently loaded
in Siril. It uses ONNX Runtime and NumPy, which it asks Siril to install when
needed. By default it loads
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
CPU models use 128 pixels of context and pad to SCUNet's 64-pixel size multiple.
`--pre-stretch` enables the same GHS prestretch/destretch used for linear data
in `SCUNet_DenoiseV2.py`; leave it off for images that are already stretched.

The exporter produces single-input, single-output NCHW models with FP32 input
and output (including the default FP16-weight variant). DirectML requires the
`onnxruntime-directml` package on Windows, fixed spatial model dimensions, and
sequential execution. The exporter creates fixed-shape models by default;
re-export existing dynamic models before using DirectML by running
`exportScunetToONNX.py` again without `--dynamic-shapes`. Tile size must not
exceed the exported model's smaller spatial dimension. The script refuses to
silently switch to CPU if DirectML is unavailable and disables ONNX Runtime's
CPU execution-provider fallback in DirectML sessions. CPU may still appear in
the registered-provider list, but DirectML inference will fail rather than
silently use CPU if the selected model cannot run on DirectML. Use
`--provider cpu` for either fixed- or dynamic-shape models instead. A mono
model cannot process an RGB image. This script processes one loaded image only;
it does not provide the original script's GUI, sequence processing, or preview
workflow. CPU inference on large full-resolution images can take a long time.
During processing, Siril's console and progress status report each tile's ONNX
inference time and the cumulative model-inference and total processing times;
no trace file is written.

To select ONNX Runtime's native WebGPU provider, use `--provider webgpu`:

```text
pyscript SCUNet_ONNX_Denoise.py --provider webgpu
```

This installs Microsoft's `onnxruntime-webgpu` package when needed (Python
3.11+; Siril's Python 3.12.9 is compatible). The package exposes
`WebGpuExecutionProvider` through the usual `onnxruntime` Python API; it is not
the JavaScript/browser runtime. Its published wheels cover Windows x86-64,
Linux x86-64, and macOS 14+ on Apple Silicon. A compatible GPU/driver is
required: Windows uses Direct3D 12 or Vulkan through Dawn, Linux uses Vulkan,
and macOS uses Metal. The script checks that the provider is available and
active and reports device initialization failures. ONNX Runtime may still assign
unsupported model operations to its CPU provider.

`onnxruntime-webgpu`, `onnxruntime-directml`, and `onnxruntime` are separate
distributions that provide the same `onnxruntime` Python module; they cannot be
safely mixed in one Python environment. When `--provider webgpu` is selected,
the script requests `onnxruntime-webgpu` through Siril's
`s.ensure_installed(...)`, replacing the active ONNX Runtime files with the
WebGPU build. It does not uninstall the DirectML distribution, but DirectML is
no longer available from the active runtime until restored. If the selected
runtime distribution is already installed but another provider is active, the
script reinstalls the selected runtime and asks you to restart Siril before
running again. It verifies provider availability and activation and reports an
actionable error if WebGPU still cannot be used.

DirectML remains the default and requires `onnxruntime-directml` on Windows.
Selecting `--provider directml` restores that runtime if it was replaced; a
restart may be required. The `onnxruntime-webgpu` package also provides the CPU
execution provider, so `--provider cpu` can use the active runtime without
replacing it.

To use the Intel GPU through the supported ONNX Runtime OpenVINO plugin, run:

```text
pyscript SCUNet_ONNX_Denoise.py --provider openvino
```

The script installs `onnxruntime-ep-openvino` on first use, registers its
execution-provider plugin, selects an available OpenVINO GPU explicitly, and
disables CPU execution-provider fallback. This plugin is separate from the
deprecated `onnxruntime-openvino` distribution and does not replace the active
`onnxruntime-directml` runtime. If no OpenVINO GPU is exposed, the script fails
instead of silently selecting its CPU device. The plugin supports Python 3.10+
and has Windows x64 wheels; its GPU execution requires a compatible Intel
graphics driver.
The automatic path was also checked in a fresh process using Siril's actual
ONNX Runtime 1.24.4 and a temporary isolated plugin path: OpenVINO activated
on Intel UHD, a real cached 384x384 SCUNet tile completed in 2.44s, and the
runtime's original DirectML/CPU provider list remained unchanged.

On this Intel UHD Graphics / Core i3-N305 test system (Python 3.12.9, ONNX
Runtime 1.24.4, OpenVINO EP plugin 1.8.1), same-precision FP32 SCUNet tiles
(384x384 input, 128px core) took a warmed median 2.44s on the OpenVINO GPU
versus 6.56s on PyTorch CPU. Four-tile runs took 9.85s and 29.00s respectively;
profiled ONNX nodes ran on OpenVINO GPU only. Cold OpenVINO compilation of the
FP32 model took 912.5s in this test, so the GUI's default FP16-weight model is
the practical OpenVINO choice: its compilation took 6.51s and warmed tiles
took a median 2.30s. The FP32 outputs differed by max 0.0146 and mean 0.0019
on the seeded test tile. These are measurements from a small bounded image,
not full-resolution throughput or quality guarantees.

OpenVINO MULTI (`MULTI:GPU,CPU`) was also checked. The supported
`onnxruntime-ep-openvino` plugin API used by this script selects a concrete
`OrtEpDevice`; passing GPU and CPU together warns that only the first device is
used, and its `device_type` provider option is ignored. The legacy
`onnxruntime-openvino` 1.24.1 provider does accept the proposed
`device_type`/`precision`/`cache_dir` options, but it is deprecated and replaces
the active ONNX Runtime package, so it was tested only in an isolated
environment. For four concurrent requests of the same 384x384 test tile,
legacy MULTI with FP16 took 16.06s versus 9.36s on its GPU-only provider.
Native OpenVINO 2025.4.1 MULTI reported both `GPU.0` and `CPU` as execution
devices, but its four-request batch took 11.10s versus 9.77s on GPU alone.
Neither MULTI test improved throughput, and the Siril tiler currently submits
tiles sequentially. MULTI is therefore not exposed as another GUI backend.

The Siril pipeline supplies ordinary host-memory NumPy arrays. Shared physical
RAM on an integrated GPU does not by itself make those host pointers valid
GPU-backed ONNX Runtime values. OpenVINO's ONNX Runtime plugin does not expose
a direct Siril/OpenVINO remote-tensor handoff in this Python path. IO binding
with CPU NumPy input still includes provider transfers; on this test tile it
measured 2.40s versus 2.30s with the normal session API, so it did not provide
a zero-copy speedup. A 384x384 RGB FP32 input or output is about 1.69 MiB.

## SCUNet Denoise GUI backend

`SCUNet_DenoiseV2.py` keeps its existing model picker, sequence processing, ROI
preview, GHS prestretch, strength blending, progress, and undo workflow. On
Windows it uses ONNX Runtime and automatically selects the OpenVINO GPU plugin
when an Intel GPU is detected; otherwise it uses DirectML. The startup probe
installs the separate `onnxruntime-ep-openvino` plugin alongside the existing
`onnxruntime-directml` runtime and explicitly checks the OpenVINO GPU device.
It does not replace DirectML. If OpenVINO cannot initialize the selected SCUNet
model, Automatic logs the reason and explicitly retries with DirectML; CPU
execution-provider fallback remains disabled for both paths. The GUI selector
also allows explicitly choosing DirectML or OpenVINO GPU.

The selected `.pth` checkpoint is exported on first use to a cached, fixed FP16
ONNX input sized for the selected tile core plus 128 pixels of context on each
side (for example, the 128px Auto core uses a 384x384 model). Each GUI
tile/halo configuration has its own cached model, so the previous 512x512
`_tile256_halo128_fp16.onnx` export is not reused. The export step uses PyTorch
and Spandrel on CPU (TorchHelper may install a regular CUDA-enabled PyTorch
build on NVIDIA systems); PyTorch is not used for Windows inference and
`torch-directml` is no longer installed or required. FP16 conversion reduces
model weight storage, but the ONNX input/output stay FP32 and some operations
may also use FP32, so it does not halve the complete inference memory
requirement.

The GUI logs the detected/selected backend and OpenVINO device in the Siril log.
If neither OpenVINO GPU initialization nor DirectML can run the selected model,
processing fails rather than silently running on CPU. Native DirectML errors
that ONNX Runtime cannot decode are reported with the
system ANSI code page, including the tile coordinates and fixed input shape;
the original exception is retained as the cause. HRESULT `0x887A0006`
(`DXGI_ERROR_DEVICE_HUNG`) indicates a Direct3D device failure; restart Siril,
close other GPU-intensive applications, and if it recurs try a smaller tile
core or check the graphics driver. The script does not retry the failed tile
or switch to CPU after a DirectML device error.

Windows Auto uses a fixed 128px tile core to reduce GPU timeout risk; explicit
512, 384, and 256px choices remain available for devices that support them.
Tile starts, 75%-stride overlap (96px stride and 32px nominal overlap for the
128px core), 128px context extraction, edge weighting, and weighted
normalization follow the PyTorch path. DirectML requires fixed input
dimensions: edge tiles that have less than 128px of real context are padded
to the exported shape, so the extreme image border can differ from PyTorch's
variable-size edge-tile inference. Larger tile cores require proportionally
more DirectML memory; if DirectML reports out-of-memory, choose a smaller core
explicitly or use a device with more available GPU memory. CPU fallback is
disabled. Other operating systems continue to use the existing PyTorch backend
and its CUDA, MPS, Intel XPU, or CPU device selection.