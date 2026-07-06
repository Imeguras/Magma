#!/usr/bin/env bash
set -euo pipefail

export LC_NUMERIC=C
export GST_DEBUG=2

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TEST_DATA="$ROOT/gst-plugin/tests/test_data"
ONNX_DIR="$ROOT/gst-plugin/tests/onnx-gen/onnx-models"
MXR_DIR="$ROOT/gst-plugin/tests/onnx-gen/migraph"
ADDONS="/usr/lib/magma/addons"
RESULTS="$ROOT/benchmarks/results.txt"

VIDEO="${1:-$TEST_DATA/ny-walking.mp4}"
FRAMES="${2:-300}"

if [ ! -f "$VIDEO" ]; then echo "ERROR: $VIDEO not found"; exit 1; fi
mkdir -p "$(dirname "$RESULTS")"

run() {
    local label="$1"
    shift
    local start end elapsed fps

    printf "  %-55s" "$label" >&2

    start=$(date +%s%N)
    timeout 10 gst-launch-1.0 -e "$@" ! fakesink num-buffers="$FRAMES" sync=false >/dev/null 2>&1 || true
    end=$(date +%s%N)

    elapsed=$(echo "scale=3; ($end - $start) / 1000000000" | bc)
    fps=$(echo "scale=1; $FRAMES / $elapsed" | bc)

    printf "%6.1f fps  (%5.2fs)\n" "$fps" "$elapsed"
    printf "  %-55s %6.1f fps  (%5.2fs)\n" "$label" "$fps" "$elapsed" >> "$RESULTS"
}

header() { printf "\n%s\n" "$1" | tee -a "$RESULTS"; }

rm -f "$RESULTS"
printf "%s\n" "==========================================" | tee -a "$RESULTS"
printf "  Magma Benchmark  —  %s\n" "$(date)" | tee -a "$RESULTS"
printf "  Commit: %s\n" "$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)" | tee -a "$RESULTS"
printf "  Video: %s\n" "$VIDEO" | tee -a "$RESULTS"
printf "  Frames per test: %s\n" "$FRAMES" | tee -a "$RESULTS"
printf "==========================================" | tee -a "$RESULTS"

DEC="filesrc location=$VIDEO ! qtdemux ! h264parse ! vah264dec"

header $'\n─── 1. Decode-only baseline ────────────────────'
run "decode + mgmvideoconvert" $DEC ! mgmvideoconvert

header $'\n─── 2. Preproc (no inference) ─────────────────'
run "preproc (224x224)" $DEC ! mgmvideoconvert ! mgmpreproc net-width=224 net-height=224
run "preproc (640x640)" $DEC ! mgmvideoconvert ! mgmpreproc net-width=640 net-height=640

header $'\n─── 3. YOLOv8n inference ──────────────────────'
YOLO="$DEC ! mgmvideoconvert ! mgmpreproc net-width=640 net-height=640 ! mgminfer model-onnx-file=$ONNX_DIR/yolov8n.onnx model-mxr-file=$MXR_DIR/yolov8n.mxr parser-plugin=$ADDONS/libyolov8-parser.so confidence-threshold=0.51 nms-threshold=0.45 max-detections=1000"
run "infer only" $YOLO
run "infer + osd" $YOLO ! mgmosd ! mgmvideoconvert

header $'\n─── 4. BiFormer (descriptor) ──────────────────'
BIF="$DEC ! mgmvideoconvert ! mgmpreproc net-width=224 net-height=224 ! mgminfer model-onnx-file=$ONNX_DIR/biformer_tiny.onnx model-mxr-file=$MXR_DIR/biformer_tiny.mxr parser-plugin=$ADDONS/libmagmabiformer-desc-parser.so confidence-threshold=0.50 max-detections=5"
run "infer only" $BIF
run "infer + osd" $BIF ! mgmosd ! mgmvideoconvert

#header $'\n─── 5. BiFormerDetect ─────────────────────────'
#BID="$DEC ! mgmvideoconvert ! mgmpreproc net-width=224 net-height=224 ! mgminfer model-onnx-file=$ONNX_DIR/biformervit-detect_tiny.onnx model-mxr-file=$MXR_DIR/biformervit-detect_tiny.mxr parser-plugin=$ADDONS/libmagmabiformer-detect-parser.so confidence-threshold=0.51 nms-threshold=0.45 max-detections=100"
#run "infer only" $BID
#run "infer + osd" $BID ! mgmosd ! mgmvideoconvert

header $'\n─── 6. Inference interval (YOLOv8n + OSD) ────'
for iv in 1 2 4 8; do
    CMD="$DEC ! mgmvideoconvert ! mgmpreproc net-width=640 net-height=640 ! mgminfer inference-interval=$iv model-onnx-file=$ONNX_DIR/yolov8n.onnx model-mxr-file=$MXR_DIR/yolov8n.mxr parser-plugin=$ADDONS/libyolov8-parser.so confidence-threshold=0.51 nms-threshold=0.45 max-detections=1000 ! mgmosd ! mgmvideoconvert"
    run "interval=$iv" $CMD
done

header $'\n─── 7. Sink comparison (YOLOv8n + OSD) ───────'
YOLO_OSD="$DEC ! mgmvideoconvert ! mgmpreproc net-width=640 net-height=640 ! mgminfer model-onnx-file=$ONNX_DIR/yolov8n.onnx model-mxr-file=$MXR_DIR/yolov8n.mxr parser-plugin=$ADDONS/libyolov8-parser.so confidence-threshold=0.51 nms-threshold=0.45 max-detections=1000 ! mgmosd ! mgmvideoconvert"
run "fakesink (sync=false)" $YOLO_OSD
run "fakesink (sync=true)"  $YOLO_OSD

MGDEC="filesrc location=$VIDEO ! qtdemux ! h264parse ! mgmh264dec"

header $'\n─── 8. GPU decode (mgmh264dec) baseline ─────────'
run "decode + mgmvideoconvert (mgpu)" $MGDEC ! mgmvideoconvert

header $'\n─── 9. GPU decode preproc (no inference) ───────'
run "preproc 224x224 (mgpu)" $MGDEC ! mgmvideoconvert ! mgmpreproc net-width=224 net-height=224
run "preproc 640x640 (mgpu)" $MGDEC ! mgmvideoconvert ! mgmpreproc net-width=640 net-height=640

header $'\n─── 10. GPU decode YOLOv8n inference ──────────'
MGYOLO="$MGDEC ! mgmvideoconvert ! mgmpreproc net-width=640 net-height=640 ! mgminfer model-onnx-file=$ONNX_DIR/yolov8n.onnx model-mxr-file=$MXR_DIR/yolov8n.mxr parser-plugin=$ADDONS/libyolov8-parser.so confidence-threshold=0.51 nms-threshold=0.45 max-detections=1000"
run "infer only (mgpu)" $MGYOLO
run "infer + osd (mgpu)" $MGYOLO ! mgmosd ! mgmvideoconvert

header $'\n────────────────────────────────────────────────'
cat "$RESULTS"
printf "\nResults: %s\n" "$RESULTS"
