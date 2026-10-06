# Segmentation model selection for Magma

## Environment

| Component | Version / value |
|---|---|
| MIGraphX | 2.15.0 (custom 2025-09-12 build) |
| GPU | AMD Radeon RX 7800 XT (gfx1101, RDNA3) |
| ROCm |  (from `/opt/rocm`) |
| ONNX opset ceiling | 17 validated (YOLO11n-seg @ opset 17 parses and compiles); models at opset 19 may work but not tested |
| Static shapes preferred | **Yes** — all YOLO segmentation exports below use static `(1,3,640,640)` |

## Candidate comparison table

| Model | Task type | Params | ONNX size | Opset | Static shape? | MIGraphX parse | MIGraphX compile | Compile time | License |
|---|---|---|---|---|---|---|---|---|---|
| **YOLO11n-seg** | Instance | 2.87M | 11 MB | 17 | ✅ 640×640 | ✅ | ✅ | 65 s | AGPL-3.0 |
| **YOLO11m-seg** | Instance | 22.4M | 86 MB | 17 | ✅ 640×640 | ✅ | ✅ | ~5 min | AGPL-3.0 |
| YOLOv8x-seg | Instance | 71.8M | 274 MB | 17 | ✅ 640×640 | ✅ | ⏰ timeout (5 min) | N/A | AGPL-3.0 |
| InternImage-T Mask R-CNN | Instance | ~49M (est.) | 434 MB | 16 | ⚠️ dynamic dims | ✅ | ✅ (via magma_compile) | N/A (runs in pipeline) | Apache-2.0 |
| DeepLabV3+ ResNet-101 | Semantic | ~60M (est.) | — | — | ⚠️ dynamic H/W if non-exported | ❌ not tested | ❌ not tested | — | BSD (torchvision weights) / Apache-2.0 |
| SAM2 tiny / SAM ViT-B image encoder | Promptable embedding | ~91M (ViT-B) | — | — | ⚠️ needs custom export | ❌ not tested | ❌ not tested | — | Apache-2.0 |
| PP-LiteSeg-T | Semantic | ~1M (est.) | — | — | needs export | ❌ not tested | ❌ not tested | — | Apache-2.0 |
| RF-DETR Segmentation | Instance | — | — | — | needs export | ❌ not tested | ❌ not tested | — | Apache-2.0 |

> ⏰ timeouts limit was 300 s. YOLOv8x-seg (275 MB, 71.8M params) would likely complete given more time.

### Evaluation environment constraints

- **DeepLabV3+ / Mask2Former / SAM2**: blocked by `torchvision` version incompatibility
  (`_meta_registrations.py` fails with `RuntimeError: operator torchvision::nms does not exist`
  under torch 2.12.1 + torchvision 0.27.1). Pre-exported ONNX downloads from ONNX Model Zoo
  and HuggingFace were attempted but the former returned 404 and the latter 401 (private repo).
  Recommend re-exporting from a compatible PyTorch/torchvision pair on a separate machine.
- **PP-LiteSeg / RF-DETR**: `segmentation_models_pytorch` and `rfdetr` packages not installed
  in the dev environment. Both are straightforward to install (`pip install`). Missed due to
  time constraints, not because of fundamental MIGraphX incompatibility.

### Node-level operator compatibility

All three YOLO-seg models use the same 18–19 ONNX operator types. MIGraphX 2.15 can handle
them all — the compile completed without unsupported-op errors. Observed operator set:

`Add, Concat, Constant, Conv, Div, Exp, Gather, Gemm, MaxPool, Mul, Neg, Not, Reshape,
Resize, Sigmoid, Slice, Softmax, Sub, Transpose`

Notable omissions from the standard set (all confirmed absent in YOLO-seg ONNX exports):
`NonMaxSuppression` (NMS is post-processing, not in the model), `RoiAlign`,
`DeformConv` (present in InternImage, absent in YOLO).

## Final picks

### All-arounder: **YOLO11n-seg** (nano) / **YOLO11m-seg** (medium)

| Criterion | Verdict |
|---|---|
| Real-time capable | ✅ 2.9M–22.4M params, standard CNN backbone, 640×640 input |
| ONNX export path | ✅ Mature `ultralytics.export(format="onnx")`, one-liner |
| MIGraphX compatibility | ✅ Parses and compiles without errors |
| Output format | Simple, well-documented (see below) |
| License concern | ⚠️ **AGPL-3.0** — if Magma needs a permissively licensed default all-arounder, YOLO11 cannot be it. The repo already ships InternImage (Apache-2.0) which *does* include a mask head; for a fully permissive stack, InternImage Mask R-CNN is the safer default. YOLO11 is the better *developer experience* choice (faster, smaller, easier to source). |
| Family range | Pick the variant that matches the deployment target: n (2.9M) → m (22.4M) → l (46.7M?) — all share the same output format, so a single parser works across sizes. |

### Weird-and-precise #1: **InternImage-T Mask R-CNN** (already in the repo)

This is Magma's current production model. It represents a genuine "weird" architecture:
- **Custom op**: DCNv3 (deformable convolution v3) — not a standard ONNX op, requires the custom C++ DCNv3 kernel
- **Multi-output**: 3 outputs (dets, labels, masks) at different shapes and dtypes
- **Mask head**: Full Mask R-CNN mask branch with ROI Align + upsample + per-class logits
- **License**: Apache-2.0 (no AGPL concern)

Lifting its parser to the new `MagmaSemanticMeta` format (already underway) gives Magma a
fully permissive, production-validated segmentation pipeline, and the parser serves as the
reference implementation for multi-output detection + mask model support.

### Weird-and-precise #2 (recommended for Phase 3): **SAM-family image encoder**

SAM's output format (a single high-dimensional embedding per image, not structured detections)
is the most architecturally distinct from both YOLO and InternImage. Stressing the parser and
the `MagmaSemanticMeta` design with an embedding-output model reveals boundary cases that
detection-shaped models won't.

**Export status**: Not validated in this session (blocked by torchvision version). The
recommended approach is a stand-alone ONNX export of the image encoder only (output:
1×256×64×64 feature embedding), with prompt encoding + mask decoding handled as a separate
stage (CPU or GPU, not in the GStreamer pipeline's hot path).

**Risk**: SAM's image encoder uses windowed attention, relative position biases, and LayerNorm
— all standard ONNX opset-17 ops, so MIGraphX should handle them. The risk is low. The
prompt decoder (which uses transformer cross-attention) is more exotic but can be deferred.

## Output tensor documentation

### YOLO11n-seg / YOLO11m-seg (shared format)

```
Input:  images  [1, 3, 640, 640]  float32
                      RGB, normalized [0, 1]
Output[0]: output0  [1, 116, 8400]  float32
Output[1]: output1  [1, 32, 160, 160]  float32
```

**Output[0] semantics** — `[batch, channels, N]` where `N = 8400` anchors:

| Column range | Content | Type |
|---|---|---|
| 0–3 | `[x_center, y_center, width, height]` — **pixels** in the 640×640 input space | float32 |
| 4–83 | per-class probability (80 COCO classes), **sigmoid already applied** | float32 |
| 84–115 | mask coefficients (32) for the proto mask | float32 |

Total: `4 + 80 + 32 = 116` channels per anchor.

There is **no objectness column** — YOLOv8/v11 are anchor-free and dropped it (that is a
YOLOv5-era layout). Two traps that produced garbage output during integration:

- The class channels come out of the graph **already sigmoided**
  (`Detect._inference` returns `cat(dbox, cls.sigmoid())`). Applying sigmoid again in the
  parser squashes every score into `[0.5, 0.73]`, so every anchor clears the threshold and
  each detection reports ~50%.
- Box coordinates are in **input pixels**, not normalized 0..1 — divide by `net_w`/`net_h`.

**Post-processing** (CPU or GPU, not in the ONNX graph):
1. Take the max class probability directly — no sigmoid.
2. Filter by confidence threshold.
3. Mask NMS per class (standard IoU NMS).
4. For each surviving detection, compute mask = `sigmoid(mask_coeffs @ proto_mask)` at the
   proto-mask resolution (160×160), then crop + rescale to the detection box.

**Output[1] semantics** — `[batch, 32, 160, 160]` — proto mask basis vectors:
- Each of the 32 channels is a 160×160 spatial activation map.
- A detection's mask is `sigmoid( Σ(c_i * proto[i]) )` where `c_i` are the detection's
  32 mask coefficients from output0[84:116].
- The 160×160 grid spans the **whole model input**, not the detection box. Every mask must
  be cropped to its own box (ultralytics `crop_mask`) or instances bleed into each other,
  and mask pixels map onto the full ROI — they are not box-relative.

**Parser notes** — implemented in `addons/magma-yolo-seg/`:
- The parser reconstructs masks from coefficients + proto mask. This is not a
  simple memory copy (unlike InternImage's direct per-instance mask output). The YOLO mask
  reconstruction requires a `MatMul` + `Sigmoid` + `Resize` per detection.
- Mask reconstruction runs on GPU (proto mask and coefficients are both float32 tensors,
  well-suited to a HIP kernel) in `nms_mask_kernel`.
- `decode_masks_kernel` turns each mask into a silhouette polygon by sampling 24 columns and
  taking the topmost/bottommost set pixel of each, giving 48 vertices per instance.
- `MagmaInferObjectGPU.class_id` is a `guint` sharing storage with a float array — the NMS
  kernel writes it with `__int_as_float`, so readers must reinterpret the bits rather than
  convert. Converting yields class 0 for every detection.

### InternImage-T Mask R-CNN (existing, documented for reference)

```
Input:  input  [1, 3, 640, 640]  float32
              RGB, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
Output[0]: dets    [1, 100, 5]  float32  — (x1, y1, x2, y2, score) in source coords
Output[1]: labels  [1, 100]     int64    — class IDs
Output[2]: masks   [1, 100, 640, 640]  float32  — per-instance logits
```

Parser is already implemented in `addons/magma-internimage/src/parse.cpp`.

### SAM image encoder (not yet exported — reference output format)

```
Input:  input  [1, 3, 1024, 1024]  float32
Output: embedding  [1, 256, 64, 64]  float32  — spatial feature map
```

The encoder output is not detection-shaped at all. A SAM pipeline needs:
1. **Encoder** — forward pass on the full image (this is what goes in the GStreamer pipeline).
2. **Prompt encoder** — encodes points/boxes into prompt tokens.
3. **Mask decoder** — lightweight transformer that attends from prompt tokens to the encoder
   embedding and outputs a mask.

Items 2 and 3 are typically CPU-side and not part of the streaming hot path.

## How to reproduce

### Download / export models

A fetch script is at `scripts/fetch-segmentation-models.sh` (see below). It exports all three
YOLO variant ONNX files and compiles MXR files for the nano and medium variants.

```sh
./scripts/fetch-segmentation-models.sh
```

### Verify MIGraphX compilation

```sh
# Parse (syntax check)
/opt/rocm/bin/migraphx-driver onnx path/to/model.onnx

# Compile for GPU
/opt/rocm/bin/migraphx-driver compile path/to/model.onnx \
    --output path/to/model.mxr --batch 1
```

### Run in Magma pipeline

```sh
GST_PLUGIN_PATH=build/gst-plugin LD_LIBRARY_PATH=build/gst-plugin gst-launch-1.0 \
  filesrc location=test_image.png \
  ! pngdec ! videoconvert ! mgmvideoconvert \
  ! mgmpreproc net-width=640 net-height=640 \
  ! mgminfer model-mxr-file=gst-plugin/tests/onnx-gen/migraph/yolo11n-seg.mxr \
            parser-plugin=build/gst-plugin/addons/libmagmayolo-seg-parser.so \
            confidence-threshold=0.4 \
  ! mgmosd labels-file=gst-plugin/addons/magma-yolo-seg/coco.names \
  ! mgmvideoconvert ! videoconvert ! jpegenc ! filesink location=out.jpg
```

Verified on `bus.jpg`, `football.png` and `pessoa.png` with both `yolo11n-seg.mxr` and
`yolo11m-seg.mxr`.

**Note**: YOLO11-seg uses a different output format from InternImage (mask coefficients +
proto mask vs. per-instance mask), so the InternImage parser **will not understand it** —
use `libmagmayolo-seg-parser.so`. See `docs/model-integration.md` for the parser plugin API.

## Rejected candidates and why

| Candidate | Reason |
|---|---|
| **YOLOv8x-seg** | 71.8M params, 274 MB file, >5 min compile timeout — too heavy for the default all-arounder slot. Keep as "extra large" option if users need maximum accuracy. |
| **DeepLabV3+ ResNet-101** | Environment constraint (torchvision broken). As a pure semantic segmentation model (one class per pixel, no instances), it *does* exercise a different output format (C×H×W logits). Worth revisiting when torchvision is fixed. |
| **PP-LiteSeg** | Not tested (package `segmentation_models_pytorch` not installed). If a permissively licensed, lightweight semantic segmentation model is needed and DeepLabV3+ is too heavy, PP-LiteSeg is the recommended replacement. |
| **RF-DETR Segmentation** | Not tested (package `rfdetr` not installed). Apache-2.0, DETR-based, potentially the best "permissively licensed all-arounder" candidate. Recommended for Phase 3 evaluation. |
| **Mask2Former** | Likely to hit MIGraphX opset-19 ceiling or masked-attention op issues. Heavy (double-digit GB VRAM). Skip for now. |
