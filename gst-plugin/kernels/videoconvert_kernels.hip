/**
 * @brief Interleave separate I420 U/V planes into NV12 UV plane.
 *
 * Each thread handles one UV pixel (one U, one V at the same location).
 *
 * @param dst_uv   Output NV12 UV plane (pitched, row stride = dst_pitch)
 * @param dst_pitch Row stride of dst_uv in bytes
 * @param src_u    Input I420 U plane (contiguous, row stride = width/2)
 * @param src_v    Input I420 V plane (contiguous, row stride = width/2)
 * @param width    Width of the full Y frame in pixels
 * @param height   Height of the full Y frame in pixels
 */
extern "C" __global__ void i420_to_nv12_uv(
    unsigned char* dst_uv, int dst_pitch,
    const unsigned char* src_u, const unsigned char* src_v,
    int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width / 2 || y >= height / 2) return;
  int uv_width = width / 2;
  dst_uv[y * dst_pitch + x * 2]     = src_u[y * uv_width + x];
  dst_uv[y * dst_pitch + x * 2 + 1] = src_v[y * uv_width + x];
}
