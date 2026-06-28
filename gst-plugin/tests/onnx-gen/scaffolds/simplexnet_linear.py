import torch
import torch.nn as nn

class UltraSimpleNet(nn.Module):
    def __init__(self):
        super(UltraSimpleNet, self).__init__()
        self.linear = nn.Linear(1, 1, bias=False)
        self.linear.weight.data.fill_(2.0)

    def forward(self, x):
        return self.linear(x)

# Instantiate and set to evaluation mode
model = UltraSimpleNet().eval()

# Dummy input: Just a single float32 number (Batch size 1, 1 feature)
dummy_input = torch.tensor([[3.0]], dtype=torch.float32)

# Export to ONNX
onnx_path = "simple_test.onnx"
torch.onnx.export(
    model, 
    dummy_input, 
    onnx_path, 
    export_params=True, 
    opset_version=11, # Highly compatible with MIGraphX
    input_names=['input'], 
    output_names=['output']
)

print(f"Successfully generated {onnx_path}!")