#!/usr/bin/env python3
"""AMD Quark INT8 PTQ wrapper for Magma model compilation.

Supports a calibration table cache: the quantized ONNX + per-tensor
calibration metadata are cached by model hash, so recalibration is
skipped when the model and calibration data haven't changed.

Usage:
    # Full calibration (first run, populates cache)
    quark_quantize.py --onnx model.onnx --output model_int8.onnx \\
        --calib-dir /path/to/images [--input-shape 1,3,640,640] \\
        [--input-name input] [--precision A8W8]

    # Cache hit (subsequent runs with same model+data are instant)
    quark_quantize.py --onnx model.onnx --output model_int8.onnx \\
        --calib-dir /path/to/images

    # Force recalibration even if cache hit
    quark_quantize.py --onnx model.onnx --output model_int8.onnx \\
        --calib-dir /path/to/images --no-cache

    # Calibrate only (no output): populate calibration table without
    # producing the quantized ONNX (useful in CI pipelines)
    quark_quantize.py --onnx model.onnx --calibrate-only \\
        --calib-dir /path/to/images

    # Use a specific calibration table file (shareable between envs)
    quark_quantize.py --onnx model.onnx --output model_int8.onnx \\
        --calib-dir /path/to/images --calib-table /path/to/calib-table.json

Output: ONNX model with QDQ nodes baked in, ready for MIGraphX compilation.

Precision options (Quark config names):
    A8W8    — symmetric INT8 activations + weights (Ryzen AI default)
    VINT8   — Vitis AI compatible INT8 (symmetric activation, power-of-2 scales)
    W8A8    — alias for A8W8
"""

import argparse
import hashlib
import json
import logging
import os
import shutil
import sys
import time

logging.basicConfig(level=logging.INFO, format="[quark] %(levelname)s: %(message)s")
log = logging.getLogger("quark_quantize")

# ── Calibration table ──────────────────────────────────────────────
#
# The calibration table is a JSON file that maps a composite key
#   (model_hash × calib_hash × config_hash) → cache entry.
#
# Schema:
#   {
#     "version": 1,
#     "entries": {
#       "<sha256 of model + calib + config>": {
#         "model_hash": "sha256:...",
#         "calib_hash": "sha256:...",
#         "config_key": "A8W8|VINT8",
#         "created_at": 1712345678.0,
#         "model_path": "/path/to/model.onnx",
#         "quantized_path": "model_int8.<cache_key>.onnx",
#         "tensor_table": {
#           "/conv1/weight": {"scale": 0.0123, "zero_point": 0, "min": -1.0, "max": 1.0},
#           "/fc/weight":    {"scale": 0.0456, "zero_point": 0, "min": -2.0, "max": 2.0},
#           ...
#         }
#       }
#     }
#   }
#
# The tensor_table maps ONNX tensor names to their calibration
# parameters.  This is populated by Quark during calibration and
# reused to skip image loading on cache hits.

CALIB_TABLE_VERSION = 1
DEFAULT_CACHE_DIR = os.path.expanduser("~/.cache/magma/quark-cache")
DEFAULT_TABLE_PATH = os.path.join(DEFAULT_CACHE_DIR, "calib-table.json")


def _hash_file(path):
    """SHA-256 of file contents."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def _hash_calib_dir(calib_dir):
    """Hash of sorted (relpath, mtime_ns, size) for all images in calib_dir.

    This is stable across repeated runs (mtime is included so adding
    or replacing images invalidates the cache).
    """
    h = hashlib.sha256()
    entries = []
    for root, _dirs, files in os.walk(calib_dir):
        for fn in sorted(files):
            if not fn.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                continue
            fp = os.path.join(root, fn)
            st = os.stat(fp)
            rel = os.path.relpath(fp, calib_dir)
            entries.append((rel, st.st_mtime_ns, st.st_size))
    for rel, mtime, size in entries:
        h.update(f"{rel}:{mtime}:{size}\n".encode())
    return "sha256:" + h.hexdigest()


def _load_table(path):
    if os.path.isfile(path):
        try:
            with open(path) as f:
                tbl = json.load(f)
            if tbl.get("version") == CALIB_TABLE_VERSION:
                return tbl.get("entries", {})
        except (json.JSONDecodeError, OSError):
            log.warning("Corrupt calibration table %s, rebuilding", path)
    return {}


def _save_table(entries, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp." + str(os.getpid())
    with open(tmp, "w") as f:
        json.dump({"version": CALIB_TABLE_VERSION, "entries": entries}, f, indent=2)
    os.rename(tmp, path)


# ── Image preprocessing helpers ─────────────────────────────────────


class ImageCalibrationDataReader:
    """Calibration data reader for AMD Quark onnx PTQ.

    Loads images from a directory, preprocesses them to match the
    model's expected input shape and normalization, then yields
    batches via get_next().
    """

    def __init__(self, calib_dir, input_name, input_shape, num_samples=200):
        import glob as _glob

        self.input_name = input_name
        self.input_shape = input_shape  # (N, C, H, W) or (N, H, W, C)
        self.batch_size = input_shape[0] if input_shape else 1
        self.enum_data = None

        # Collect image paths
        exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
        images = []
        for ext in exts:
            images.extend(_glob.glob(os.path.join(calib_dir, ext)))
            images.extend(_glob.glob(os.path.join(calib_dir, ext.upper())))
        if not images:
            log.warning("No images found in %s — will use random data", calib_dir)
            self.data_list = None
            return

        images = sorted(images)[:num_samples]
        log.info("Loading %d calibration images from %s", len(images), calib_dir)
        self.data_list = self._preprocess_many(images, input_shape)

    def _preprocess_many(self, image_paths, input_shape):
        import numpy as np
        from PIL import Image

        N, C, H, W = input_shape
        nhwc = input_shape[-1] == C and input_shape[-2] != C  # heuristic
        data = []
        for path in image_paths:
            img = Image.open(path).convert("RGB")
            img = img.resize((W, H), Image.BILINEAR)
            arr = np.asarray(img, dtype=np.float32)  # (H, W, C), [0,255]
            # Normalize to [0,1]
            arr = arr / 255.0
            if not nhwc:
                # Convert to NCHW
                arr = np.transpose(arr, (2, 0, 1))  # (C, H, W)
            data.append(arr)

        # Stack into batches
        batched = []
        for i in range(0, len(data), self.batch_size):
            batch = data[i : i + self.batch_size]
            if len(batch) < self.batch_size:
                break  # drop incomplete last batch
            batched.append(np.stack(batch, axis=0))
        return batched

    def get_next(self):
        if self.data_list is None or len(self.data_list) == 0:
            # Fall back to random data
            import numpy as np

            N, C, H, W = self.input_shape
            return {self.input_name: np.random.randn(N, C, H, W).astype(np.float32)}

        if self.enum_data is None:
            self.enum_data = iter(
                [{self.input_name: data} for data in self.data_list]
            )
        return next(self.enum_data, None)

    def rewind(self):
        self.enum_data = None


# ── Quark quantization ──────────────────────────────────────────────


def _quark_quantize(onnx_path, output_path, calib_reader, precision):
    """Run AMD Quark PTQ, return per-tensor calibration data."""
    from quark.onnx import ModelQuantizer, QConfig

    if precision == "VINT8":
        quant_config = QConfig.get_default_config("VINT8")
        quant_config.global_quant_config.extra_options["Int32Bias"] = False
        quant_config.global_quant_config.extra_options["DedicatedQDQPair"] = True
        quant_config.global_quant_config.extra_options["QuantizeAllOpTypes"] = True
    else:
        quant_config = QConfig.get_default_config("A8W8")

    # Enable calibration output capture if supported
    quant_config.global_quant_config.extra_options.get("CalibrationData", {})

    quantizer = ModelQuantizer(quant_config)
    log.info("Quantizing %s → %s (%s, calib=%s)", onnx_path, output_path, precision,
             "provided" if calib_reader and calib_reader.data_list else "random")
    quantizer.quantize_model(
        model_input=onnx_path,
        model_output=output_path,
        calibration_data_reader=calib_reader,
    )

    # Attempt to extract per-tensor calibration data from the
    # quantized model by reading QDQ node parameters.
    tensor_table = _extract_tensor_table(output_path)
    return tensor_table


def _extract_tensor_table(quantized_onnx_path):
    """Read QDQ scale/zero-point from the quantized ONNX graph.

    Returns a dict: {tensor_name: {"scale": float, "zero_point": int}}
    """
    import onnx

    model = onnx.load(quantized_onnx_path)
    tbl = {}

    # Collect initializer values keyed by name
    init_map = {init.name: init for init in model.graph.initializer}

    for node in model.graph.node:
        # QuantizeLinear: input → scale, zero_point
        if node.op_type in ("QuantizeLinear", "DequantizeLinear"):
            if len(node.input) >= 3:
                scale_name = node.input[1]
                zp_name = node.input[2]
                scale_init = init_map.get(scale_name)
                zp_init = init_map.get(zp_name)
                if scale_init and zp_init:
                    import numpy as np
                    scale = np.frombuffer(scale_init.raw_data, dtype=np.float32)
                    zp = np.frombuffer(zp_init.raw_data, dtype=(
                        np.int8 if zp_init.data_type == 3 else
                        np.uint8 if zp_init.data_type == 2 else
                        np.float32
                    ))
                    for j in range(len(scale)):
                        key = f"{scale_name}/{j}" if len(scale) > 1 else scale_name
                        tbl[key] = {
                            "scale": float(scale[j]),
                            "zero_point": int(zp[j % len(zp)]),
                        }
    return tbl


# ── Cache helpers ───────────────────────────────────────────────────


def _cache_key(model_hash, calib_hash, config_key):
    """Composite key for the calibration table."""
    h = hashlib.sha256()
    h.update(model_hash.encode())
    h.update(calib_hash.encode())
    h.update(config_key.encode())
    return h.hexdigest()[:32]


# ── Main ────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="AMD Quark INT8 PTQ for Magma")
    parser.add_argument("--onnx", required=True, help="Input FP32 ONNX model")
    parser.add_argument("--output", default=None, help="Output INT8 ONNX model")
    parser.add_argument("--calib-dir", default=None,
                        help="Directory of calibration images")
    parser.add_argument("--input-shape", default="1,3,640,640",
                        help="Model input shape (N,C,H,W) comma-separated")
    parser.add_argument("--input-name", default="input",
                        help="ONNX graph input name")
    parser.add_argument("--precision", default="A8W8",
                        choices=["A8W8", "VINT8", "W8A8"],
                        help="Quark quantization config name")
    parser.add_argument("--num-calib-samples", type=int, default=200,
                        help="Max calibration images to use")
    parser.add_argument("--calib-table", default=None,
                        help="Path to calibration table JSON (default: "
                             "~/.cache/magma/quark-cache/calib-table.json)")
    parser.add_argument("--cache-dir", default=None,
                        help="Directory for cached quantized ONNX files "
                             "(default: ~/.cache/magma/quark-cache)")
    parser.add_argument("--calibrate-only", action="store_true",
                        help="Only populate calibration table, do not write "
                             "quantized ONNX")
    parser.add_argument("--no-cache", action="store_true",
                        help="Force recalibration, ignore cache")
    args = parser.parse_args()

    # Validate
    if not os.path.isfile(args.onnx):
        log.error("Input ONNX not found: %s", args.onnx)
        sys.exit(1)

    if not args.calibrate_only and not args.output:
        log.error("Either --output or --calibrate-only is required")
        sys.exit(1)

    # Resolve cache paths
    cache_dir = args.cache_dir or DEFAULT_CACHE_DIR
    table_path = args.calib_table or os.path.join(cache_dir, "calib-table.json")
    os.makedirs(cache_dir, exist_ok=True)

    # Compute hashes
    model_hash = _hash_file(args.onnx)
    calib_hash = _hash_calib_dir(args.calib_dir) if args.calib_dir and os.path.isdir(args.calib_dir) else "none"
    config_key = args.precision
    ckey = _cache_key(model_hash, calib_hash, config_key)

    # Check calibration table
    entries = _load_table(table_path)
    cached = entries.get(ckey)
    quantized_cache_path = os.path.join(cache_dir, ckey + ".onnx")
    tensor_table = None

    if cached and os.path.isfile(quantized_cache_path) and not args.no_cache:
        log.info("Cache HIT for %s (%s)", args.onnx, ckey)
        tensor_table = cached.get("tensor_table", {})
        if not args.calibrate_only:
            shutil.copy2(quantized_cache_path, args.output)
            log.info("Copied cached quantized model to %s", args.output)
        # Print calibration table summary for diagnostics
        if tensor_table:
            log.info("Calibration table has %d tensor entries (reused from cache)",
                     len(tensor_table))
        return

    # Cache miss — run Quark
    log.info("Cache MISS for %s — running Quark PTQ", args.onnx)

    try:
        import quark.onnx  # noqa: F401 — verify importable
    except ImportError:
        log.error("AMD Quark not installed. Run: pip install amd-quark")
        sys.exit(1)

    input_shape = tuple(int(x) for x in args.input_shape.split(","))

    # Build calibration data reader
    calib_reader = None
    if args.calib_dir and os.path.isdir(args.calib_dir):
        calib_reader = ImageCalibrationDataReader(
            calib_dir=args.calib_dir,
            input_name=args.input_name,
            input_shape=input_shape,
            num_samples=args.num_calib_samples,
        )
    else:
        log.warning("No calibration directory — using random data")

    # Quantize to cache path first
    tensor_table = _quark_quantize(args.onnx, quantized_cache_path,
                                   calib_reader, args.precision)

    # Save calibration table entry
    entries[ckey] = {
        "model_hash": model_hash,
        "calib_hash": calib_hash,
        "config_key": config_key,
        "created_at": time.time(),
        "model_path": os.path.abspath(args.onnx),
        "quantized_path": ckey + ".onnx",
        "tensor_table": tensor_table,
    }
    _save_table(entries, table_path)
    log.info("Calibration table updated at %s", table_path)

    # Produce output
    if args.calibrate_only:
        log.info("Calibrate-only mode: cached at %s (table has %d tensor entries)",
                 quantized_cache_path, len(tensor_table or {}))
    else:
        shutil.copy2(quantized_cache_path, args.output)
        log.info("Quantized model saved to %s", args.output)

    # Print per-tensor calibration summary
    if tensor_table:
        scales = [v["scale"] for v in tensor_table.values() if v["scale"] > 0]
        if scales:
            log.info("Calibration ranges: scale min=%.6f  max=%.6f  median=%.6f  "
                     "count=%d", min(scales), max(scales),
                     sorted(scales)[len(scales) // 2], len(scales))


if __name__ == "__main__":
    main()
