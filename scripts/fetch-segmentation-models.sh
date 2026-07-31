#!/usr/bin/env bash
set -euo pipefail

# ─────────────────────────────────────────────────────────────────────
# fetch-segmentation-models.sh
#
# Downloads / exports YOLO11 segmentation models for Magma.
# Requires: ultralytics, torch, onnx (all available in the dev env)
#
# Outputs are placed in
#   gst-plugin/tests/onnx-gen/onnx-models/segmentation/
#
# Usage:
#   ./scripts/fetch-segmentation-models.sh [variant]
#
# Variants: nano (default), medium, all
# ─────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
OUT_DIR="$REPO_DIR/gst-plugin/tests/onnx-gen/onnx-models/segmentation"
mkdir -p "$OUT_DIR"

variant="${1:-nano}"

case "$variant" in
  nano)   MODELS=("yolo11n-seg") ;;
  medium) MODELS=("yolo11m-seg") ;;
  all)    MODELS=("yolo11n-seg" "yolo11m-seg") ;;
  *)
    echo "Usage: $0 [nano|medium|all]"
    exit 1
    ;;
esac

export_py=$(mktemp --suffix=.py)
trap 'rm -f "$export_py"' EXIT

cat > "$export_py" << 'PYEOF'
import sys, subprocess, torch
torch.manual_seed(0)
from ultralytics import YOLO

model_name = sys.argv[1]
out_dir = sys.argv[2]

print(f"Exporting {model_name}...")
model = YOLO(f"{model_name}.pt")
model.export(format="onnx", imgsz=640, opset=17, dynamic=False)

src = f"{model_name}.onnx"
dst = f"{out_dir}/{model_name}.onnx"
subprocess.run(["cp", src, dst], check=True)
print(f"Copied to {dst}")

# Compile with MIGraphX (if migraphx-driver is available)
if subprocess.run(["which", "migraphx-driver"], capture_output=True).returncode == 0:
    mxr = f"{out_dir}/{model_name}.mxr"
    print(f"Compiling {dst} -> {mxr}...")
    subprocess.run([
        "/opt/rocm/bin/migraphx-driver", "compile",
        dst, "--output", mxr, "--batch", "1"
    ], check=True, timeout=600)
    print(f"Compiled to {mxr}")
else:
    print("migraphx-driver not found, skipping MXR compilation")
PYEOF

for m in "${MODELS[@]}"; do
    python3 "$export_py" "$m" "$OUT_DIR"
done

echo "Done. Models in $OUT_DIR"
ls -lh "$OUT_DIR"
