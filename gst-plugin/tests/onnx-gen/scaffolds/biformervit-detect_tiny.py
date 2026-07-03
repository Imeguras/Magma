import torch
import torch.nn as nn
import sys
sys.path.append('.')
from models.biformer import biformer_tiny

class BiFormerDetector(nn.Module):
    def __init__(self, num_classes=80, num_queries=100):
        super().__init__()
        backbone = biformer_tiny(pretrained=False)
        self.downsample_layers = backbone.downsample_layers
        self.stages = backbone.stages
        self.norm = backbone.norm
        feat_dim = backbone.num_features[-1]

        self.query_embed = nn.Parameter(torch.randn(1, num_queries, feat_dim))
        self.bbox_head = nn.Linear(feat_dim, 4)
        self.cls_head = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        x = self.norm(x)
        feat = x.flatten(2).mean(dim=-1)
        queries = feat.unsqueeze(1) + self.query_embed
        bboxes = self.bbox_head(queries)
        logits = self.cls_head(queries)
        return torch.cat([bboxes, logits], dim=-1)

model = BiFormerDetector(num_classes=80, num_queries=100)
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)

print("Exporting BiFormerDetector to ONNX...")
try:
    torch.onnx.export(
        model,
        dummy_input,
        "biformervit-detect_tiny.onnx",
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output']
    )
    print("Success! Output shape: [1, 100, 84] (100 queries x (4 bbox + 80 class scores))")
except Exception as e:
    print(f"Export failed: {e}")
