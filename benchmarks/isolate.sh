#!/usr/bin/env bash
set -euo pipefail
# Usage: ./benchmarks/isolate.sh <component> [extra rocprof args]
# Components: decode, upload, preproc, preproc-roi, infer, download, tensordump

export LC_NUMERIC=C
export PATH="/opt/rocm/bin:/usr/local/bin:/usr/bin:/bin"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TEST_DATA="$ROOT/gst-plugin/tests/test_data"
ONNX_DIR="$ROOT/gst-plugin/tests/onnx-gen/onnx-models"
MXR_DIR="$ROOT/gst-plugin/tests/onnx-gen/migraph"
ADDONS="/usr/lib/magma/addons"
PROFILE_DIR="$ROOT/profiles"
mkdir -p "$PROFILE_DIR"

VIDEO="${TEST_DATA}/ny-walking.mp4"
FRAMES=100

component="${1:-help}"
shift || true

outdir="$PROFILE_DIR/isolate_${component}_$(date +%Y%m%d_%H%M%S)"

PIPELINE=()
build_pipeline() {
  local comp=$1
  PIPELINE=()
  case "$comp" in
    decode)
      PIPELINE=(filesrc "location=$VIDEO" "!" qtdemux "!" h264parse "!" mgmh264dec
                "!" fakesink num-buffers="$FRAMES" sync=false)
      ;;
    upload)
      PIPELINE=(filesrc "location=$VIDEO" "!" qtdemux "!" h264parse "!" vah264dec
                "!" mgmvideoconvert "!" fakesink num-buffers="$FRAMES" sync=false)
      ;;
    preproc)
      PIPELINE=(filesrc "location=$VIDEO" "!" qtdemux "!" h264parse "!" vah264dec
                "!" mgmvideoconvert
                "!" mgmpreproc net-width=640 net-height=640
                "!" fakesink num-buffers="$FRAMES" sync=false)
      ;;
    preproc-roi)
      PIPELINE=(filesrc "location=$VIDEO" "!" qtdemux "!" h264parse "!" vah264dec
                "!" mgmvideoconvert
                "!" mgmpreproc net-width=640 net-height=640
                     enable-roi=true roi-x=0 roi-y=587 roi-w=719 roi-h=451
                "!" fakesink num-buffers="$FRAMES" sync=false)
      ;;
    infer)
      PIPELINE=(filesrc "location=$VIDEO" "!" qtdemux "!" h264parse "!" vah264dec
                "!" mgmvideoconvert
                "!" mgmpreproc net-width=640 net-height=640
                "!" mgminfer
                     model-onnx-file="$ONNX_DIR/yolov8n.onnx"
                     model-mxr-file="$MXR_DIR/yolov8n.mxr"
                     parser-plugin="$ADDONS/libyolov8-parser.so"
                     confidence-threshold=0.51 nms-threshold=0.45 max-detections=1000
                "!" fakesink num-buffers="$FRAMES" sync=false)
      ;;
    download)
      PIPELINE=(filesrc "location=$VIDEO" "!" qtdemux "!" h264parse "!" vah264dec
                "!" mgmvideoconvert
                "!" mgmpreproc net-width=640 net-height=640
                "!" mgminfer
                     model-onnx-file="$ONNX_DIR/yolov8n.onnx"
                     model-mxr-file="$MXR_DIR/yolov8n.mxr"
                     parser-plugin="$ADDONS/libyolov8-parser.so"
                     confidence-threshold=0.51 nms-threshold=0.45 max-detections=1000
                "!" mgmosd "!" mgmvideoconvert
                "!" fakesink num-buffers="$FRAMES" sync=false)
      ;;
    tensordump)
      PIPELINE=(filesrc "location=$VIDEO" "!" qtdemux "!" h264parse "!" vah264dec
                "!" mgmvideoconvert
                "!" mgmpreproc net-width=640 net-height=640
                "!" mgmtensordump net-width=640 net-height=640
                "!" fakesink num-buffers="$FRAMES" sync=false)
      ;;
    *)
      echo "Unknown component: $comp" >&2
      echo "Available: decode, upload, preproc, preproc-roi, infer, download, tensordump" >&2
      exit 1
      ;;
  esac
}

build_pipeline "$component"

echo "=== Isolating: $component ==="
echo "Pipeline: gst-launch-1.0 ${PIPELINE[*]}"
echo "Output: $outdir"
echo ""

rocprofv3 --kernel-trace --hip-trace \
  --output-directory "$outdir" \
  --output-format csv \
  -- gst-launch-1.0 "${PIPELINE[@]}" "$@"

echo ""
echo "=== Results ==="
csv=$(find "$outdir" -name '*kernel_trace*' 2>/dev/null | head -1)
if [ -n "$csv" ]; then
  echo "Kernel timing CSV: $csv"
  echo ""
  echo "Top kernels by total time (us):"
  awk -F, 'NR>1 {
    name=$8; dur=($11+0)-($10+0)
    if (name != "") { k[name]+=dur; c[name]++ }
  } END {
    for (n in k) printf "%9.0f %5d  %s\n", k[n]/1000, c[n], n
  }' "$csv" | sort -rn | head -20
fi
echo ""
echo "HIP trace CSV:"
find "$outdir" -name '*hip_trace*' 2>/dev/null | head -1
