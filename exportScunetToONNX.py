import torch
import onnx

# Attempt to import sirilpy. If not running inside Siril, the import will fail.
try:
    import sirilpy as s

    # Check the module version
    if not s.check_module_version('>=1.0.10'):
        print("Error: requires sirilpy module >= 1.0.10")
        sys.exit(1)

    from sirilpy import SirilError
    # 1. PyQt6 e Spandrel prima di Torch
    s.ensure_installed("spandrel", "onnxconverter_common")

except ImportError as e:
    print(f"Warning: module imports failed: {e}")
    sys.exit(1)

from spandrel import ImageModelDescriptor, ModelLoader
from onnxconverter_common import float16
import sys, os

siril = s.SirilInterface()
try:
    siril.connect()
except Exception:
    print('Connection to Siril failed. Make sure Siril is open and ready.')
    sys.exit(1)

user_dir = siril.get_siril_userdatadir()
models_dir = os.path.join(user_dir, 'scunet_models')
model_filename = "scunet_color_real_psnr.pth"
modelpath = os.path.join(models_dir, model_filename)

# 1. Load the SCUNet checkpoint automatically using Spandrel
descriptor = ModelLoader().load_from_file(modelpath)
assert isinstance(descriptor, ImageModelDescriptor)

# 2. Extract the underlying PyTorch nn.Module and set to eval mode
pytorch_model = descriptor.model
pytorch_model.eval()

# 3. Build dummy input using Spandrel's detected input_channels (3 for RGB, 1 for Grayscale)
# Height and width MUST be multiples of 64 for SCUNet
dummy_input = torch.randn(1, descriptor.input_channels, 256, 256, dtype=torch.float32)

# 4. Export FP32 ONNX model
fp32_path =  os.path.join(models_dir, "scunet_color_real_psnr_fp32.onnx")
torch.onnx.export(
    pytorch_model,
    dummy_input,
    fp32_path,
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch_size", 2: "height", 3: "width"},
        "output": {0: "batch_size", 2: "height", 3: "width"},
    },
    dynamo=False,
)

# 5. Convert to FP16 (keeping FP32 I/O for convenience)
model_fp32 = onnx.load(fp32_path)
model_fp16 = float16.convert_float_to_float16(model_fp32, keep_io_types=True)
onnx.save(model_fp16,  os.path.join(models_dir, "scunet_color_real_psnr_fp16.onnx"))
print(f"Exported {descriptor.architecture.name} successfully!")

