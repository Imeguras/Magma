/**
 * @brief Convert NV12 frame to CHW planar RGB float32 tensor with nearest-neighbor resize + normalize.
 *
 * I know this is obvious, but im not very experienced with kernel coding so yeah...
 * If you're wondering where the fuck did the guy fetch this weird functions?  theres a common.hip since i dislike repeating small functions.
 * One thread per output pixel. Each thread maps its output coordinate (x, y) back
 * to a source pixel via nearest-neighbor, reads Y/U/V, converts to RGB, scales to [0,1],
 * and writes into a planar CHW tensor (RRR...GGG...BBB).
 *
 * @param frame       Input NV12 frame (packed: Y plane then interleaved UV at offset height*stride)
 * @param width       Width of the input frame in pixels
 * @param height      Height of the input frame in pixels
 * @param stride      Row stride of the input frame in bytes (may differ from width)
 * @param tensor      Output float32 tensor, size 3 * net_w * net_h, planar CHW layout
 * @param net_w       Target tensor width (network input width)
 * @param net_h       Target tensor height (network input height)
 * @param scale_factor  Multiplier applied after YUV→RGB (typically 1/255.0f to normalize to [0,1])
 */
extern "C" __global__ void nv12_to_rgb_normalized(unsigned char* frame, int width, int height, int stride,
    float* tensor, int net_w, int net_h, int crop_x, int crop_y, int crop_w, int crop_h, float scale_factor,
    int* error_code) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    //its just to clean up the code since again i dont like repeating stuff
    float* r_plane = tensor;
    float* g_plane = tensor + net_w * net_h;
    float* b_plane = tensor + 2 * net_w * net_h;
    int point_tensor= y * net_w + x;
    // damn the solution made me feel embarassed that i took about 2 hours to figure it out even with an llm telling me what was wrong with each iteration...
    if (x < net_w && y < net_h) {
        if(crop_x < 0 || crop_y < 0 || crop_w <= 0 || crop_h <= 0 || crop_x + crop_w > width || crop_y + crop_h > height) {
            if (x == 0 && y == 0) *error_code = MAGMA_ERR_ROI_BOUNDS;
            // Invalid crop region, set output to black
            r_plane[point_tensor] = 0.0f;
            g_plane[point_tensor] = 0.0f;
            b_plane[point_tensor] = 0.0f;

            return;
        }
        float proportion_x = (float)crop_w / (float)net_w;
        float proportion_y = (float)crop_h / (float)net_h;

        int src_x = (x * proportion_x)+crop_x; 
        int src_y = (y * proportion_y)+crop_y;

        //src_x = clampi(src_x, crop_x, crop_x+crop_w-1);
        //src_y = clampi(src_y, crop_y, crop_y+crop_h-1);

        // Extract stuff from NV12
        unsigned char Y = frame[src_y * stride + src_x];
        int uv_index = height * stride + (src_y / 2) * stride + (src_x / 2) * 2;
        unsigned char U = frame[uv_index];
        unsigned char V = frame[uv_index + 1];
        //BT 601 conversion into RGB
        float r = Y + 1.402f * (V - 128);
        float g = Y - 0.344136f * (U - 128) - 0.714136f * (V - 128);
        float b = Y + 1.772f * (U - 128);
        //Apply intensity scale_factor and clamp to [0,1] voilá we have a normalized RGB tensor
        r = clampf(r * scale_factor, 0.0f, 1.0f);
        g = clampf(g * scale_factor, 0.0f, 1.0f);
        b = clampf(b * scale_factor, 0.0f, 1.0f);
        r_plane[point_tensor] = r;
        g_plane[point_tensor] = g;
        b_plane[point_tensor] = b;
    }
}
