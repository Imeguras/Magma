/**
 * @brief Convert NV12 frame to CHW planar RGB float32 tensor with nearest-neighbor resize + normalize.
 *
 * One thread per output pixel. Each thread maps its output coordinate (x, y) back
 * to a source pixel via nearest-neighbor, reads Y/U/V, converts to RGB, scales to [0,1],
 * and writes into a planar CHW tensor (RRR...GGG...BBB).
 *
 * ROI is validated on the host before launch — this kernel always receives valid bounds.
 *
 * @param frame       Input NV12 frame (packed: Y plane then interleaved UV at offset height*stride)
 * @param width       Width of the input frame in pixels
 * @param height      Height of the input frame in pixels
 * @param stride      Row stride of the input frame in bytes (may differ from width)
 * @param tensor      Output float32 tensor, size 3 * net_w * net_h, planar CHW layout
 * @param net_w       Target tensor width (network input width)
 * @param net_h       Target tensor height (network input height)
 * @param crop_x      Left edge of crop rectangle in source
 * @param crop_y      Top edge of crop rectangle in source
 * @param crop_w      Width of crop rectangle in source
 * @param crop_h      Height of crop rectangle in source
 * @param scale_factor  Multiplier applied after YUV→RGB (typically 1/255.0f to normalize to [0,1])
 */
extern "C" {
__global__ void
nv12_to_rgb_normalized(unsigned char* frame, int width, int height, int stride, float* tensor, int net_w, int net_h, int crop_x, int crop_y, int crop_w, int crop_h, float scale_factor, float mean_r, float mean_g, float mean_b, float std_r, float std_g, float std_b) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    float* r_plane = tensor;
    float* g_plane = tensor + net_w * net_h;
    float* b_plane = tensor + 2 * net_w * net_h;
    int point_tensor = y * net_w + x;

    if (x < net_w && y < net_h) {
        if (crop_x < 0 || crop_y < 0 || crop_w <= 0 || crop_h <= 0 || crop_x + crop_w > width || crop_y + crop_h > height) {
            // Invalid crop region, set output to black
            r_plane[point_tensor] = 0.0f;
            g_plane[point_tensor] = 0.0f;
            b_plane[point_tensor] = 0.0f;

            return;
        }

        float proportion_x = (float)crop_w / (float)net_w;
        float proportion_y = (float)crop_h / (float)net_h;

        int src_x = (x * proportion_x) + crop_x;
        int src_y = (y * proportion_y) + crop_y;

        unsigned char Y = frame[src_y * stride + src_x];
        int uv_index = height * stride + (src_y / 2) * stride + (src_x / 2) * 2;
        unsigned char U = frame[uv_index];
        unsigned char V = frame[uv_index + 1];

        float r = Y + 1.402f * (V - 128);
        float g = Y - 0.344136f * (U - 128) - 0.714136f * (V - 128);
        float b = Y + 1.772f * (U - 128);

        r = (r * scale_factor - mean_r) / std_r;
        g = (g * scale_factor - mean_g) / std_g;
        b = (b * scale_factor - mean_b) / std_b;

        r_plane[point_tensor] = r;
        g_plane[point_tensor] = g;
        b_plane[point_tensor] = b;
    }
}
}