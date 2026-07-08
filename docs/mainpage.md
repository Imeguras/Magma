# Magma v1.0.0

GPU-accelerated video analytics framework for AMD ROCm.

Magma is a suite of GStreamer plugins that enables hardware-accelerated video
decode, preprocessing, inference, visualization, and display on AMD GPUs —
a free and open-source alternative to NVIDIA DeepStream.

## Architecture

<!-- TODO: user to fill in theory of operation -->

## Plugins

| Plugin | Description |
|--------|-------------|
| `mgmh264dec` | H.264 GPU decoder via rocDecode |
| `mgmvideoconvert` | Cross-format video conversion (NV12/I420, system/DMABuf/HIP) |
| `mgmpreproc` | ROI crop, resize, and tensor normalization |
| `mgminfer` | MIGraphX model inference with parser plugin support |
| `mgmosd` | On-screen display of bounding boxes and labels |
| `mgmserialize` | Serialize inference results to JSON/Protobuf |
| `mgmtensordump` | Dump tensor data to file with RGB preview |
| `mgmkpublish` | Kafka sink for publishing results |
| `mgmdisplay` | DRM/KMS display sink with NV12→RGB GPU conversion |

## License

LGPLv3 — see LICENSE.md
