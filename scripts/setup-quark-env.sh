#!/usr/bin/env bash
# setup-quark-env.sh — Create a Python environment with AMD Quark for Magma
#
# Usage:
#   ./setup-quark-env.sh                    # creates venv at ./quark-env
#   ./setup-quark-env.sh /path/to/venv      # custom location
#   ./setup-quark-env.sh --conda            # use conda (if available)
#
# The resulting environment is auto-detected by magma_compile (in
# mgminfer.cpp and internimage/parse.cpp) when running quark_quantize.py.
#
# Environment variables:
#   QUARK_ENV_DIR    — venv/conda location (default: ./quark-env or ./quark-conda)
#   PIP_EXTRA        — extra pip packages (e.g., "pillow onnxruntime")

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

log()  { echo -e "${CYAN}[setup-quark]${NC} $*"; }
ok()   { echo -e "${GREEN}[setup-quark]${NC} $*"; }
warn() { echo -e "${RED}[setup-quark]${NC} $*"; }

# ── Resolve location ───────────────────────────────────────────────

USE_CONDA=false
if [[ "${1:-}" == "--conda" ]]; then
    USE_CONDA=true
    shift
fi

if [[ -n "${QUARK_ENV_DIR:-}" ]]; then
    ENV_DIR="$QUARK_ENV_DIR"
elif [[ $# -ge 1 ]]; then
    ENV_DIR="$1"
else
    ENV_DIR="${USE_CONDA:+./quark-conda}"
    ENV_DIR="${ENV_DIR:-./quark-env}"
fi

ENV_DIR="$(realpath -m "$ENV_DIR")"

# ── Check Python ───────────────────────────────────────────────────

PYTHON=$(command -v python3 || command -v python)
if [[ -z "$PYTHON" ]]; then
    warn "Python 3 not found. Install python3 or python3-minimal."
    exit 1
fi

PY_VER=$("$PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
log "Using $PYTHON (Python $PY_VER)"

# Quark requires >=3.11, <3.14
if [[ "$(echo "$PY_VER >= 3.11" | bc -l 2>/dev/null)" != "1" && \
      "$(echo "$PY_VER >= 3.11" | awk '{exit($1<$2)}')" == "0" ]]; then
    # fallback to awk if bc not present
    MAJ=$(echo "$PY_VER" | cut -d. -f1)
    MIN=$(echo "$PY_VER" | cut -d. -f2)
    if [[ "$MAJ" -lt 3 || ( "$MAJ" -eq 3 && "$MIN" -lt 11 ) ]]; then
        warn "Python 3.11+ required (found $PY_VER). Install python3.11 or newer."
        exit 1
    fi
fi

# ── Create environment ─────────────────────────────────────────────

if $USE_CONDA; then
    CONDA=$(command -v conda || true)
    if [[ -z "$CONDA" ]]; then
        warn "conda not found. Install Miniconda first:"
        echo "  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
        echo "  bash Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda3"
        exit 1
    fi
    if [[ -d "$ENV_DIR" ]]; then
        log "Conda env already exists at $ENV_DIR — skipping creation"
    else
        log "Creating conda env at $ENV_DIR ..."
        "$CONDA" create -y -p "$ENV_DIR" python="$PY_VER" pip
    fi
    PIP="$ENV_DIR/bin/pip"
    PYTHON="$ENV_DIR/bin/python"
else
    VENV_MODULE=$("$PYTHON" -c "import sys; print('venv' if sys.version_info >= (3,3) else '')" 2>/dev/null)
    if [[ -z "$VENV_MODULE" ]]; then
        warn "python3-venv not available. Install it:"
        echo "  apt install python3-venv    # Debian/Ubuntu"
        echo "  dnf install python3-virtualenv  # Fedora"
        exit 1
    fi
    if [[ -d "$ENV_DIR" ]]; then
        log "venv already exists at $ENV_DIR — skipping creation"
    else
        log "Creating venv at $ENV_DIR ..."
        "$PYTHON" -m venv "$ENV_DIR"
    fi
    PIP="$ENV_DIR/bin/pip"
    PYTHON="$ENV_DIR/bin/python"
fi

# ── Install packages ───────────────────────────────────────────────

log "Upgrading pip ..."
"$PIP" install --quiet --upgrade pip

log "Installing AMD Quark ..."
"$PIP" install --quiet amd-quark

# Runtime extras: PIL (image loading for calibration) + ONNX Runtime
log "Installing runtime extras (Pillow, onnxruntime, numpy) ..."
"$PIP" install --quiet pillow onnxruntime numpy

# User-specified extras
if [[ -n "${PIP_EXTRA:-}" ]]; then
    log "Installing extra packages: $PIP_EXTRA"
    "$PIP" install --quiet $PIP_EXTRA
fi

# ── Verify ─────────────────────────────────────────────────────────

log "Verifying installation ..."
"$PYTHON" -c "
import quark.onnx
print(f'  AMD Quark version: {quark.onnx.__version__}')" 2>/dev/null || {
    warn "Quark import failed."
    # Try with pypi-style import for newer versions
    "$PYTHON" -c "
from quark.onnx import ModelQuantizer, QConfig
print('  quark.onnx import OK')
" 2>/dev/null || {
        warn "Import still failing — check error above"
        exit 1
    }
}

# ── Activation helpers ─────────────────────────────────────────────

ENV_NAME=$(basename "$ENV_DIR")
log ""
ok "AMD Quark environment ready at: ${ENV_DIR}"
ok ""
ok "Activate with:"
if $USE_CONDA; then
    ok "  conda activate ${ENV_DIR}"
else
    ok "  source ${ENV_DIR}/bin/activate"
fi
ok ""
ok "Then run magma_compile with INT8 precision and calib-dir set."
ok ""
ok "Or set the Python path in magma_compile's quark_quantize.py call:"
ok "  QUARK_PYTHON=${ENV_DIR}/bin/python"
