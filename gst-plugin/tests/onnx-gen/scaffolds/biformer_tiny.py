import torch
import sys
# Append the path so Python can find the models folder
sys.path.append('.')
from models.biformer import biformer_tiny # Or whichever variant you want

# 1. Initialize the model 
model = biformer_tiny(pretrained=False) 
model.eval()

# 2. Enforce absolute static dimensions (224x224) 
# Tracing forces PyTorch to unroll the routing attention loops for this specific size
dummy_input = torch.randn(1, 3, 224, 224)

print("Attempting ONNX export via structural tracing...")
try:
    torch.onnx.export(
        model,
        dummy_input,
        "biformer_tiny.onnx",
        export_params=True,
        opset_version=17, # Higher opsets have better luck with dynamic loops
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output']
    )
    print("Success!")
except Exception as e:
    print(f"Export failed due to custom ops: {e}")
