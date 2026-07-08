#!/usr/bin/env bash
# Enable core dumps for debugging
ulimit -c unlimited

# Ensure the core pattern includes PID so we can identify dumps
echo "core.%p" | sudo tee /proc/sys/kernel/core_pattern 2>/dev/null || true

# Point GST to our debug-built plugins
export GST_PLUGIN_PATH="$(dirname "$0")/../build-debug/gst-plugin"
export LD_LIBRARY_PATH="$GST_PLUGIN_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export GST_DEBUG_FILE=gst.log

echo "Core dumps enabled (ulimit -c unlimited)"
echo "GST_PLUGIN_PATH=$GST_PLUGIN_PATH"
echo ""
echo "Running: gst-launch-1.0 $*"
echo ""

exec gst-launch-1.0 "$@"
