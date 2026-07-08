#!/usr/bin/env bash
# Find and load the latest core dump for a given binary
BINARY="${1:-gst-launch-1.0}"
CORE_DIR="${2:-.}"

# Find newest core file
CORE=$(ls -t "$CORE_DIR"/core.* 2>/dev/null | head -1)
if [ -z "$CORE" ]; then
    echo "No core dump found in $CORE_DIR"
    echo "To enable core dumps: ulimit -c unlimited"
    echo "Then run your pipeline and let it crash."
    echo ""
    echo "Usage: $0 [binary] [core-dir]"
    exit 1
fi

echo "Found core: $CORE"
echo "Binary: $BINARY"
echo "GDB command: gdb -c \"$CORE\" \"$BINARY\""
echo ""
echo "Useful GDB commands after loading:"
echo "  bt                  - backtrace (call stack)"
echo "  bt full             - backtrace with local vars"
echo "  frame N             - switch to stack frame N"
echo "  info locals         - show local variables"
echo "  info args           - show function arguments"
echo "  up/down             - navigate the stack"
echo "  p variable          - print a variable"
echo "  l                   - show source code"
echo "  thread apply all bt - backtrace for all threads"
echo "  quit                - exit GDB"
echo ""

exec gdb -c "$CORE" "$BINARY"
