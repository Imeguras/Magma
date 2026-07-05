#!/usr/bin/env bash
set -euo pipefail
# ==============================================================
#  Magma Profiling Helper
#  Usage:  ./benchmarks/profile.sh <mode> [extra gst args]
#
#  Modes:
#    stats          rocprof --stats  — kernel timing & occupancy
#    counters       rocprof -i <counters.txt>  — HW perf counters
#    trace          rocprofv3 --sys-trace  — full HIP/HSA/kernel trace
#    kernel         rocprofv3 --kernel-trace  — kernel dispatch trace only
#    timeline       rocprofv3 --hip-trace --kernel-trace  — HIP API + kernel
# ==============================================================

export LC_NUMERIC=C
export GST_DEBUG=2
export GST_PLUGIN_PATH="${GST_PLUGIN_PATH:-build/gst-plugin}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TEST_DATA="$ROOT/gst-plugin/tests/test_data"
ONNX_DIR="$ROOT/gst-plugin/tests/onnx-gen/onnx-models"
MXR_DIR="$ROOT/gst-plugin/tests/onnx-gen/migraph"
ADDONS="/usr/lib/magma/addons"

VIDEO="${TEST_DATA}/ny-walking.mp4"
FRAMES=100
PROFILE_DIR="$ROOT/profiles"
mkdir -p "$PROFILE_DIR"

DEC="filesrc location=$VIDEO ! qtdemux ! h264parse ! vah264dec"
PIPE="$DEC ! mgmvideoconvert ! mgmpreproc net-width=640 net-height=640"
YOLO="$PIPE ! mgminfer model-onnx-file=$ONNX_DIR/yolov8n.onnx model-mxr-file=$MXR_DIR/yolov8n.mxr parser-plugin=$ADDONS/libyolov8-parser.so confidence-threshold=0.51 nms-threshold=0.45 max-detections=1000 ! mgmosd ! mgmvideoconvert"
END="! fakesink num-buffers=$FRAMES sync=false"

mode="${1:-stats}"
shift || true

case "$mode" in
  stats)
    echo "=== rocprof --stats ==="
    echo "Pipeline: $YOLO $END"
    echo "Output: $PROFILE_DIR/stats"
    rm -f "$PROFILE_DIR"/stats.*.txt "$PROFILE_DIR"/stats.*.csv
    rocprof --stats -d "$PROFILE_DIR/stats" \
      gst-launch-1.0 $YOLO $END "$@"
    echo
    echo "Quick summary of top kernels by duration:"
    if [ -f "$PROFILE_DIR/stats/results.csv" ]; then
      column -t -s',' "$PROFILE_DIR/stats/results.csv" | sort -t' ' -k4 -rn | head -10
    elif [ -f "$PROFILE_DIR/stats.stats.csv" ]; then
      column -t -s',' "$PROFILE_DIR/stats.stats.csv" | sort -t' ' -k4 -rn | head -10
    fi
    ;;

  counters)
    echo "=== rocprof -i counters.txt ==="
    cat > /tmp/magma_counters.txt << 'EOF'
# Preproc kernel drill-down
pmc : SQ_WAVES SQ_LDS_INST_L1_SUM SQ_VMEM_INST_L1_SUM SQ_SMEM_INST_L1_SUM SQ_TOTAL_MEM_STALL_L1_SUM
range: 0 :
gpu: 0
kernel: nv12_to_rgb_normalized
EOF
    rocprof -i /tmp/magma_counters.txt -d "$PROFILE_DIR/counters" \
      gst-launch-1.0 $YOLO $END "$@"
    echo "Counters output: $PROFILE_DIR/counters"
    ;;

  trace)
    stamp=$(date +%Y%m%d_%H%M%S)
    outdir="$PROFILE_DIR/trace_$stamp"
    echo "=== rocprofv3 --sys-trace (output: $outdir) ==="
    echo "Launching with 100 frames..."
    rocprofv3 --sys-trace \
      --output-directory "$outdir" \
      --output-format pftrace \
      -- gst-launch-1.0 $YOLO $END "$@"
    echo
    # Find the .pftrace file
    pf=$(find "$outdir" -name '*.pftrace' 2>/dev/null | head -1)
    if [ -n "$pf" ]; then
      echo "Perfetto trace: $pf"
      echo "View with:   perfetto --ui  # then open $pf"
      echo "Or copy:     cp $pf /tmp/   # for loading in ui.perfetto.dev"
    fi
    ;;

  kernel)
    stamp=$(date +%Y%m%d_%H%M%S)
    outdir="$PROFILE_DIR/kernel_$stamp"
    echo "=== rocprofv3 --kernel-trace (output: $outdir) ==="
    rocprofv3 --kernel-trace \
      --output-directory "$outdir" \
      --output-format csv \
      -- gst-launch-1.0 $YOLO $END "$@"
    echo
    csv=$(find "$outdir" -name '*.csv' 2>/dev/null | head -1)
    if [ -n "$csv" ]; then
      echo "Kernel timing CSV: $csv"
      column -t -s',' "$csv" | head -20
    fi
    ;;

  timeline)
    stamp=$(date +%Y%m%d_%H%M%S)
    outdir="$PROFILE_DIR/timeline_$stamp"
    echo "=== rocprofv3 --hip-trace --kernel-trace (output: $outdir) ==="
    rocprofv3 --hip-trace --kernel-trace \
      --output-directory "$outdir" \
      --output-format csv,pftrace \
      -- gst-launch-1.0 $YOLO $END "$@"
    echo
    pf=$(find "$outdir" -name '*.pftrace' 2>/dev/null | head -1)
    [ -n "$pf" ] && echo "Perfetto trace: $pf"
    csv=$(find "$outdir" -name '*kernel*' -name '*.csv' 2>/dev/null | head -1)
    [ -n "$csv" ] && echo "Kernel CSV: $csv"
    ;;

  *)
    echo "Usage: $0 <mode> [extra gst args]"
    echo ""
    echo "Modes:"
    echo "  stats       rocprof --stats  — kernel timing & occupancy"
    echo "  counters    rocprof -i <counter file>  — HW perf counters"
    echo "  trace       rocprofv3 --sys-trace  — full system trace (.pftrace)"
    echo "  kernel      rocprofv3 --kernel-trace  — kernel dispatch only"
    echo "  timeline    rocprofv3 --hip-trace --kernel-trace  — HIP + kernel"
    ;;
esac
