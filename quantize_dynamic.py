import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
import tempfile
import shutil
import os

# Paths
model_fp32 = "model_fp32.onnx"
model_dynamic_quant = "model_dynamic_quant.onnx"

# Load the model and fix shape inference issues
print("Loading and preparing model for quantization...")
model = onnx.load(model_fp32)

# Remove all value_info entries that might have conflicting shapes
# This prevents shape inference errors while keeping the model functional
print(f"Clearing {len(model.graph.value_info)} intermediate value_info entries...")
while len(model.graph.value_info) > 0:
    model.graph.value_info.pop()

# Create a temporary file for the cleaned model
temp_dir = tempfile.mkdtemp()
temp_model_path = os.path.join(temp_dir, "cleaned_model.onnx")

try:
    # Save the cleaned model
    onnx.save(model, temp_model_path)
    print("Model prepared successfully")
    
    # Perform dynamic quantization
    print("Starting quantization...")
    quantize_dynamic(
        model_input=temp_model_path,
        model_output=model_dynamic_quant,
        weight_type=QuantType.QInt8
    )
    
    print(f"✓ Dynamic quantization complete. Quantized model saved to: {model_dynamic_quant}")
finally:
    # Clean up temporary directory
    shutil.rmtree(temp_dir, ignore_errors=True)
