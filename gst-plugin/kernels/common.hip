// Common device helpers shared across magma kernels.
// This file is prepended to kernel source at JIT compile time.

// --- Error codes (written by kernel, read by host) ---
#define MAGMA_ERR_NONE        0
#define MAGMA_ERR_ROI_BOUNDS  1  // crop rectangle exceeds source dimensions

__device__ inline int clampi(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

__device__ inline float clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

__device__ inline float lerp(float a, float b, float t) {
    return a + (b - a) * t;
}
