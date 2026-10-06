# Adding a new model to Magma

This document describes the complete workflow for adding a new model type
to the Magma pipeline: from ONNX export → parser plugin → type_id convention
→ primitive rendering → INT8 quantization with AMD Quark → validation.

## Architecture overview

Magma processes video frames through a pipeline of GPU elements:

```
source → mgmh264dec → mgmvideoconvert → mgmpreproc → mgminfer → mgmosd → sink
```

The inference element (`mgminfer`) runs a MIGraphX-compiled ONNX model and
invokes a **parser plugin** (`.so` loaded at runtime via `dlopen`) to convert
raw tensor output into structured results. Those results are carried through
the pipeline as **GStreamer metadata** and consumed by **mgmosd** for
on-screen rendering.

### The two-layer rendering graph

To support many model types (detection, pose, segmentation, gaze, audio, track)
without mgmosd itself knowing about any of them, we use a two-layer approach:

1. **Semantic layer** — Each model kind has its own `type_id` string
   (e.g. `"magma.detection.v1"`, `"magma.pose.v1"`) and its own GPU data
   layout. The data is attached to buffers via `MagmaSemanticMeta`.

2. **Primitive layer** — A small, closed vocabulary of render primitives
   (`rect`, `polyline`, `polygon`, `point`, `text`, `arrow`, `ticker`) that
   mgmosd natively renders. Every converter turns semantic data into these
   primitives.

### The bridge: `magma_to_primitives`

Each parser plugin **optionally** exports a `magma_to_primitives` function
that the rendering pipeline calls to convert semantic meta → primitives.
Parser plugins that do not export this function will still work: mgmosd
falls back to the legacy `MagmaInferenceMeta` path for detection boxes.

## Parser plugin ABI

A parser plugin is a shared library (`.so`) loaded by mgminfer at runtime.
It exports one or more of the following symbols:

### Required: `magma_parse`

```c
extern "C" int magma_parse(MagmaParseParams* params);
```

Converts raw GPU tensor output from MIGraphX into structured detection
objects (or custom GPU data). The `MagmaParseParams` struct provides:

| Field | Direction | Description |
|-------|-----------|-------------|
| `d_raw_output` / `d_raw_outputs[]` | input | GPU pointers to model output tensors |
| `output_shape` / `output_shapes[]` | input | Tensor shapes |
| `net_width`, `net_height` | input | Model input dimensions |
| `confidence_thresh`, `nms_thresh` | input | Filtering thresholds |
| `max_detections` | input | Output buffer capacity |
| `d_objects` | output | GPU buffer for results (MagmaInferObjectGPU[] or custom) |
| `d_num_detected` | output | GPU int written by parser |
| `stream` | input | HIP stream for kernel launches |

Returns 0 on success.

### Optional: `magma_compile`

```c
extern "C" int magma_compile(const MagmaCompileParams* params);
```

Compiles an ONNX model → MIGraphX program with optional quantization.
If not exported, mgminfer uses its default MIGraphX compilation path.

| Field | Description |
|-------|-------------|
| `onnx_path` | Source ONNX file |
| `mxr_output_path` | Path to write compiled `.mxr` |
| `precision` | `"FP32"`, `"FP16"`, or `"INT8"` |
| `calib_data_path` | Calibration dataset (required for INT8) |
| `batch_size` | Inference batch size |
| `extras` | Null-terminated `key=value` array for per-model overrides |

See the [AMD Quark INT8 workflow](#int8-quantization-with-amd-quark) section below.

### Optional: `magma_to_primitives`

```c
extern "C" int magma_to_primitives(
    const MagmaToPrimitivesParams* params,
    MagmaPrimitiveList* out);
```

Converts semantic meta → render primitives. Called by mgmosd at render time.

| Field | Description |
|-------|-------------|
| `type_id` | Semantic type string (e.g. `"magma.pose.v1"`) |
| `d_data` | GPU pointer to model-specific data |
| `data_size` | Bytes of GPU data |
| `cpu_data`, `cpu_data_size` | Pre-mapped CPU copy (for converters that cannot use GPU) |
| `source_width`, `source_height` | Source frame dimensions |
| `model_width`, `model_height` | Model input dimensions |
| `roi_x/y/w/h` | ROI within the source frame |
| `stream` | HIP stream |

The function fills `out` with primitives using the inline append helpers
declared in `magma-primitives.h`:

- `magma_primitive_list_add_rect()`
- `magma_primitive_list_add_polyline()`
- `magma_primitive_list_add_polygon()`
- `magma_primitive_list_add_point()`
- `magma_primitive_list_add_text()`
- `magma_primitive_list_add_arrow()`

All coordinates must be in **source-pixel space** (not model-normalised).
The `roi_*` and `source_*` fields provide the remapping context.

### Optional: `magma_semantic_type`

```c
extern "C" const char* magma_semantic_type = "magma.pose.v1";
```

A string constant that identifies the semantic type of this parser's output.
mgminfer reads this symbol at load time and uses it as the registry key when
storing the `magma_to_primitives` function pointer.

**Convention:** `"magma.<model_kind>.v<version>"` where `<model_kind>` is
one of:
- `detection` — bounding box + class_id (built-in default converter)
- `pose` — keypoint + skeleton (example in `magma-pose-example`)
- `segmentation` — per-pixel masks
- `gaze` — gaze vector + point of regard
- `audio` — audio event tickers
- `track` — bounding box + track_id

## `type_id` convention and auto-selection

Each `MagmaSemanticMeta` attached to a buffer carries a `type_id` (GQuark).
The lookup chain is:

1. mgmosd iterates all `MagmaSemanticMeta` entries on the buffer.
2. For each, it calls `magma_lookup_to_primitives(type_id_str)`.
3. The registry checks for an exact match first (parser-exported converter).
4. If no exact match, it checks the prefix: `"magma.detection.*"` maps to
   the built-in default converter.
5. Unrecognised `type_id` values are silently skipped.

This means:
- **New detection parsers** can export `magma_to_primitives` for custom
  logic, or rely on the built-in default converter if their output is
  `MagmaInferObjectGPU[]` with the standard box layout.
- **Non-detection parsers** (pose, seg, gaze, etc.) **must** export
  `magma_to_primitives` and `magma_semantic_type` — there is no default
  converter for these shapes.

## Example: `magma-pose-example`

See `gst-plugin/addons/magma-pose-example/` for a complete reference
implementation of a non-detection parser. It demonstrates:

- `magma_parse` — reads keypoint output, filters by confidence, writes
  custom GPU buffer layout
- `magma_semantic_type = "magma.pose.v1"` — declares the type
- `magma_to_primitives` — reads the custom GPU buffer, emits point
  (keypoint joint) + polyline (skeleton edge) primitives

The build is defined in `gst-plugin/addons/meson.build`.

## Registration timing contract

1. `libmagma-meta.so` constructor registers the built-in default converter
   for `"magma.detection.v1"` at library load time (before any pipeline
   elements are created).

2. When `mgminfer` moves to `PAUSED` (start), it `dlopen`s the parser .so
   and `dlsym`s `magma_parse`, `magma_compile`, `magma_semantic_type`,
   and `magma_to_primitives`. If `magma_to_primitives` is found, it calls
   `magma_register_to_primitives(type_id, func)`.

3. When `mgmosd` processes a frame, it iterates `MagmaSemanticMeta` entries
   on the buffer, looks up the converter by `type_id`, and calls it.

4. Unrecognised `type_id` values (those with no registered converter) are
   silently skipped — never fatal.

## INT8 quantization with AMD Quark

### Workflow

```
ONNX model + calibration dataset
              │
              ▼
    AMD Quark (offline PTQ)
              │
              ▼
    Quantized ONNX (QDQ nodes baked in)
              │
              ▼
    magma_compile (precision="INT8")
    → migraphx::parse_onnx → compile → save .mxr
              │
              ▼
    Deploy with mgminfer + INT8 .mxr
```

### Implementation

The `magma_compile` hook in parser addons (see `magma-internimage/src/parse.cpp`
for a reference) handles the INT8 path as:

1. **Option A (recommended):** Run Quark offline before `magma_compile`.
   Quark produces an ONNX with QDQ (Quantize/Dequantize) nodes already
   baked in. The `magma_compile` INT8 branch then skips
   `migraphx::quantize_*` entirely and goes directly to
   `migraphx::parse_onnx` + `compile`.

2. **Option B (MIGraphX in-process):** Use Quark to generate per-tensor
   scale factors, then feed them into MIGraphX's own
   `migraphx::quantize_int8` via calibration parameter maps.

**Current status:** The INT8 branch is a stub that logs "TBD" and falls
back. The AMD Quark integration is the pending work described in the next
section.

### Known tuning need

Quantization shifts confidence/logit distributions. The detection/RPN/cascade
parsers hardcode `confidence_thresh`/`nms_thresh` handling from
`MagmaParseParams`. **Do not assume FP32 thresholds are still correct for an
INT8-compiled model.** Re-tune thresholds per precision:

- Use a representative validation set
- Sweep confidence_thresh in [0.05, 0.95] at 0.05 increments
- Sweep nms_thresh in [0.3, 0.8] at 0.05 increments
- Select thresholds that maximise mAP on the validation set

## Validation workflow

### Regression test (detection, CPU-only)

```sh
GST_PLUGIN_PATH=build/gst-plugin LD_LIBRARY_PATH=build/gst-plugin \
  gst-launch-1.0 filesrc location=test_image.png \
  ! pngdec ! videoconvert ! mgmvideoconvert \
  ! mgmpreproc ! mgminfer model-onnx-file=model.onnx parser-plugin=... \
  ! mgmosd ! fakesink
```

### Unit tests

```sh
cd build && meson test test-primitives --print-errorlogs   # primitive list API
cd build && meson test test-infer-object --print-errorlogs  # legacy detection API
cd build && meson test test-inference-meta --print-errorlogs # meta API
```

### INT8 accuracy validation

For each model:
1. Compile FP16 (baseline)
2. Compile INT8 via Quark
3. Run both through the same pipeline on a validation set
4. Compare mAP (or task-specific metric)
5. Record latency improvement and accuracy delta

See `benchmarks/` for profiling scripts.
