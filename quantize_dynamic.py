import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

# Paths
model_fp32 = "model_fp32.onnx"
model_dynamic_quant = "model_dynamic_quant.onnx"

# Perform dynamic quantization
quantize_dynamic(
    model_input=model_fp32,
    model_output=model_dynamic_quant,
    weight_type=QuantType.QInt8
)

print(f"Dynamic quantization complete. Quantized model saved to: {model_dynamic_quant}")
