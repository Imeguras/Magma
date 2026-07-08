extern "C" __global__ void nv12_to_xrgb8888(
    const unsigned char* y, int y_stride,
    const unsigned char* uv, int uv_stride,
    unsigned int* dst, int dst_stride_pixels,
    int w, int h)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int yp = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || yp >= h) return;

    int Y = y[yp * y_stride + x];
    int U = uv[(yp / 2) * uv_stride + (x / 2) * 2];
    int V = uv[(yp / 2) * uv_stride + (x / 2) * 2 + 1];

    // BT.601 limited→full range
    int yv = Y  - 16;
    int u_val = U  - 128;
    int v_val = V  - 128;
    int R = (298 * yv + 409 * v_val + 128) >> 8;
    int G = (298 * yv - 100 * u_val - 208 * v_val + 128) >> 8;
    int B = (298 * yv + 516 * u_val + 128) >> 8;

    if (R < 0) R = 0; if (R > 255) R = 255;
    if (G < 0) G = 0; if (G > 255) G = 255;
    if (B < 0) B = 0; if (B > 255) B = 255;

    dst[yp * dst_stride_pixels + x] = 0xFF000000 | (R << 16) | (G << 8) | B;
}
