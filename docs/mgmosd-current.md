# mgmosd — Current State (Pre-Refactor)

This document describes `mgmosd` as it existed before the model-agnostic
rendering refactor. It catalogues every detection-specific and hardcoded
assumption so the refactor does not accidentally lose behaviour.

## File layout

| File | Role |
|------|------|
| `gst-plugin/include/mgmosd.hpp` | GObject struct + class declaration |
| `gst-plugin/src/mgmosd.cpp` | GstBaseTransform implementation |
| `gst-plugin/kernels/osd_kernels.hip` | Single GPU kernel (`draw_boxes_kernel`) |
| `gst-plugin/kernels/common.hip` | Shared device helpers (prepended at JIT time) |

## End-to-end data flow

```
  GstBuffer (NV12 DMABuf or system memory)
    │
    ├── MagmaInferenceMeta (attached by mgminfer)
    │     └── objects_gpu : GstMemory*
    │           └── MagmaInferObjectGPU[ N ]  (class_id, conf, x, y, w, h)
    │
    └── raw NV12 pixels
```

1. **Start** (`gst_magma_osd_start`):
   - Acquires shared HIP stream (`magma_get_shared_hip_stream`)
   - JIT-compiles `osd_kernels.hip` → `draw_boxes_kernel` via `compile_kernel`
   - Pre-allocates `hipMalloc` buffer for **100** `BoxParam` entries

2. **Transform** (`gst_magma_osd_transform_ip`):
   - Reads `MagmaInferenceMeta` from buffer
   - Maps `m->objects_gpu` (a `GstMemory` wrapping heap-allocated
     `MagmaInferObjectGPU[]` — copied from GPU to CPU in
     `magma-meta.c:attach_inference_meta`)
   - Coordinate-remaps each object from model-normalised space to source-pixel
     space using the ROI recorded by the preprocessing chain
   - Packs up to **100** objects into a stack-allocated `BoxParam params[100]`
   - Acquires GPU frame pointer (HipMeta → DMABuf import → system-memory upload)
   - Copies `params[]` to pre-allocated device buffer
   - Launches `draw_boxes_kernel` (1 thread-group per box, 64-wide)

3. **Frame pointer acquisition** (reusable):
   - Check `MagmaHipMeta` → use raw HIP pointer directly
   - Fall back: DMABuf import via `hipImportExternalMemory`
   - Last resort: `hipMalloc` + `hipMemcpy` from system memory

## Detection-specific features — inventory

### 1. COCO class-name table (`mgmosd.cpp:16-33`)
Hardcoded array of 80 COCO class name strings. Accessed via
`coco_names[obj->class_id % COCO_NUM_CLASSES]`. Any model that does not use
COCO class IDs gets wrong or misleading labels. 80 entries, null-terminated
`const char*` pointers.

### 2. `BoxParam` struct (`mgmosd.cpp:80-85`)
Tightly coupled to the detection-box case:
```c
struct BoxParam {
    int x, y, w, h;
    unsigned char y_val, u_val, v_val;   // colour (YUV, pre-computed from palette)
    unsigned char _pad;
    unsigned char label[8];              // max 7 chars + NUL
};
```
- `label[8]` limits text to 7 printable characters (in practice "person." or
  truncated names)
- `_pad` is an explicit alignment byte — fragile, unused
- No facility for filled boxes, polylines, points, masks, arrows, or any
  non-rectangular primitive

### 3. `MagmaInferObjectGPU` / `MagmaInferenceMeta` dependency
`mgmosd.cpp` `#include`s `magma-infer-meta.h` and calls
`magma_buffer_get_inference_meta()`. The element has no way to consume
inference output that does not fit the box + class_id + confidence shape.
Pose keypoints, segmentation masks, gaze vectors, audio events, and tracking
IDs are invisible to mgmosd.

### 4. 100-object hard cap (`mgmosd.cpp:139,226-228`)
Stack array `BoxParam params[100]` and device allocation
`hipMalloc(100 * sizeof(BoxParam))`. Silently drops objects beyond 100 with
no warning. Configurable ceiling does not exist.

### 5. Single kernel launch (`osd_kernels.hip:136`)
`draw_boxes_kernel` renders **only** axis-aligned rectangle outlines with an
optional label in a fixed-position background strip above the box. Rectangle
rendering, label-background fill, and bitmap font rendering are interleaved in
one kernel — no code reuse with future primitive types.

### 6. Coordinate remapping (`mgmosd.cpp:213-234`)
The model-space → source-space mapping is hardcoded for box semantics:
```c
int sx = off_x + (int)(obj->x * map_w);
int sy = off_y + (int)(obj->y * map_h);
int sw = (int)(obj->width  * map_w);
int sh = (int)(obj->height * map_h);
```
This assumes every model output has `(cx, cy, w, h)` or `(x1, y1, w, h)`
normalised 0..1 semantics. A pose model outputting keypoint coordinates or a
segmentation model outputting per-pixel masks would need different remapping.

### 7. Colour palette (`mgmosd.cpp:57-77`)
16-entry YUV palette indexed by `class_id % 16`. Single palette with no
way to select key (`track_id` vs `class_id`), which breaks tracking
use-cases where each track ID needs a stable colour across frames.

### 8. 5×7 bitmap font (`osd_kernels.hip:12-108`)
Inlined in the GPU kernel source — the `font5x7[95][7]` table. ASCII 32–126,
monochrome, 5×7 pixels. This is not a problem per se (a small inline font is
fast), but duplication with any future text-rendering path would be wasteful.
Should become a shared device symbol if more kernels draw text.

## What is reusable (will be kept)

- **HIP stream acquisition** (`magma_get_shared_hip_stream`) via
  `magma-hip-stream.hpp`
- **DMABuf import path** (`import_dmabuf` / `hipImportExternalMemory`) —
  caches the `hipExternalMemory_t` handle across frames
- **System-memory upload fallback** (map + `hipMemcpy` when no DMABuf or
  HipMeta is available)
- **ROI coordinate remapping** logic (the `roi_x/y/w/h` + `model_width/height`
  preference cascade) — must be preserved as-is for model-space → source-space
  conversion on rect primitives
- **YUV colour palette** — the 16 pre-computed YUV colours, but make the
  palette key selectable
- **Kernel JIT compilation** via `compile_kernel` — the refactored kernels
  use the same mechanism

## Summary

`mgmosd` is 496 lines of C++ (header + source) + 241 lines of GPU kernel code,
entirely specialised for one rendering task. Every data type, every code path,
and the shader itself assumes detection boxes. Seven hardcoded assumptions
(listed above) must be broken to support pose, segmentation, gaze, audio, and
tracking rendering without per-model patches to mgmosd itself.
