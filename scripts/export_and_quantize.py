from optimum.exporters.onnx import main_export
from onnxruntime.quantization import quantize_dynamic, QuantType
import os

model_id = "sentence-transformers/all-MiniLM-L6-v2"
output_dir = "models/all-MiniLM-L6-v2"
os.makedirs(output_dir, exist_ok=True)

# Export model to ONNX
main_export(
    model_name_or_path=model_id,
    output=output_dir,
    task="feature-extraction"
)

# Quantize model
quantize_dynamic(
    model_input=os.path.join(output_dir, "model.onnx"),
    model_output=os.path.join(output_dir, "model_quantized.onnx"),
    weight_type=QuantType.QInt8
)

print("Quantization complete. Model saved to:", output_dir)
