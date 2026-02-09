from onnxruntime.quantization import (
    quantize_static,
    QuantType
)
from datasets import dataset_loader
import onnx
from onnxruntime.quantization import CalibrationDataReader
import tempfile
import shutil
import os

class ImageCalibrationDataReader(CalibrationDataReader):
    def __init__(self, dataloader, input_name, num_batches=10):
        self.iterator = iter(dataloader)
        self.input_name = input_name
        self.num_batches = num_batches
        self.count = 0

    def get_next(self):
        if self.count >= self.num_batches:
            return None

        try:
            images, _ = next(self.iterator)
        except StopIteration:
            return None

        self.count += 1
        return {self.input_name: images.numpy()}


if __name__ == '__main__':
    model_fp32 = "model_fp32.onnx"
    model_static_quant = "static_model_quant.onnx"

    # Load ONNX model and fix shape inference issues
    print("Loading and preparing model for quantization...")
    onnx_model = onnx.load(model_fp32)
    input_name = onnx_model.graph.input[0].name

    # Remove all value_info entries that might have conflicting shapes
    print(f"Clearing {len(onnx_model.graph.value_info)} intermediate value_info entries...")
    while len(onnx_model.graph.value_info) > 0:
        onnx_model.graph.value_info.pop()

    # Get calibration data (use num_worker=0 to avoid multiprocessing issues on Windows)
    print("Loading calibration data...")
    train_loader, _, _ = dataset_loader(
        dataset="cifar10",
        batch_size=32,
        image_size=224,
        num_worker=0  # Set to 0 for Windows compatibility
    )

    calibration_reader = ImageCalibrationDataReader(
        dataloader=train_loader,
        input_name=input_name,
        num_batches=10
    )

    # Create a temporary file for the cleaned model
    temp_dir = tempfile.mkdtemp()
    temp_model_path = os.path.join(temp_dir, "cleaned_model.onnx")

    try:
        # Save the cleaned model
        onnx.save(onnx_model, temp_model_path)
        print("Model prepared successfully")
        
        # Perform static quantization
        print("Starting static quantization...")
        quantize_static(
            model_input=temp_model_path,
            model_output=model_static_quant,
            calibration_data_reader=calibration_reader,
            weight_type=QuantType.QInt8,
            activation_type=QuantType.QInt8
        )
        
        print(f"✓ Static quantization complete, model saved to: {model_static_quant}")
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)
