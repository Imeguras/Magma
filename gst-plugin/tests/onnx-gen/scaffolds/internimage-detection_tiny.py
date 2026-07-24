import torch
import torch.nn as nn
import sys
sys.path.append('path/to/InternImage/classification')
from models.intern_image import InternImage

class InternImageDetector(nn.Module):
    def __init__(self, num_classes=80, num_queries=100):
        super().__init__()
        backbone = InternImage(
            core_op='DCNv3',
            channels=64,
            depths=[4, 4, 18, 4],
            groups=[4, 8, 16, 32],
            num_classes=0,
            mlp_ratio=4.,
            drop_rate=0.,
            drop_path_rate=0.2,
            act_layer='GELU',
            norm_layer='LN',
            layer_scale=1.0,
            offset_scale=1.0,
            post_norm=False,
            cls_scale=1.5,
            with_cp=False,
        )
        feat_dim = backbone.num_features
        self.backbone = backbone

        self.query_embed = nn.Parameter(torch.randn(1, num_queries, feat_dim))
        self.bbox_head = nn.Linear(feat_dim, 4)
        self.cls_head = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        x = self.backbone.patch_embed(x)
        x = self.backbone.pos_drop(x)
        for level in self.backbone.levels:
            x = level(x)
        x = self.backbone.conv_head(x.permute(0, 3, 1, 2))
        feat = x.flatten(2).mean(dim=-1)
        queries = feat.unsqueeze(1) + self.query_embed
        bboxes = self.bbox_head(queries).sigmoid()
        logits = self.cls_head(queries)
        return torch.cat([bboxes, logits], dim=-1)

model = InternImageDetector(num_classes=80, num_queries=100)
model.eval()

dummy_input = torch.randn(1, 3, 640, 640)

print("Exporting InternImageDetector to ONNX...")
try:
    torch.onnx.export(
        model,
        dummy_input,
        "internimage-detection_tiny.onnx",
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output']
    )
    print("Success! Output shape: [1, 100, 84] (100 queries x (4 bbox + 80 class scores))")
    print("Bbox format: cx, cy, w, h (normalized 0..1)")
except Exception as e:
    print(f"Export failed: {e}")
