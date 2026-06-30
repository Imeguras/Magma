import torch
import torch.nn as nn

class AdaptedLinearNet(nn.Module):
    def __init__(self):
        super(AdaptedLinearNet, self).__init__()
        self.flatten = nn.Flatten()
        # mgmpreproc outputs [1, 3, 32, 32] = 3072 floats
        # Flatten → [1, 3072] → Linear → [1, 1]
        self.linear = nn.Linear(3 * 32 * 32, 1, bias=False)
        self.linear.weight.data.fill_(2.0)

    def forward(self, x):
        x = self.flatten(x)
        return self.linear(x)

model = AdaptedLinearNet().eval()

# Match mgmpreproc output shape: [batch=1, channels=3, height=32, width=32]
dummy_input = torch.randn(1, 3, 32, 32, dtype=torch.float32)

onnx_path = "simple_test_adapted.onnx"
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    export_params=True,
    opset_version=11,
    input_names=['input'],
    output_names=['output']
)

print(f"Generated {onnx_path}!")
print(f"Model: input [1,3,32,32] → flatten → Linear(3072,1) → [1,1]")
print(f"Weight = {model.linear.weight.data[0,0].item():.4f}")
