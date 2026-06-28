extern "C" __global__ void nv12_to_rgb_normalized(unsigned char* frame, int width, int height, int stride,
    float* tensor, int net_w, int net_h, float scale_factor) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < net_w && y < net_h) {
        int src_x = x * width / net_w;
        int src_y = y * height / net_h;

        unsigned char Y = frame[src_y * stride + src_x];

        int uv_index = (src_y / 2) * stride + (src_x / 2) * 2;
        unsigned char U = frame[uv_index];
        unsigned char V = frame[uv_index + 1];

        float r = Y + 1.402f * (V - 128);
        float g = Y - 0.344136f * (U - 128) - 0.714136f * (V - 128);
        float b = Y + 1.772f * (U - 128);

        r = fminf(fmaxf(r * scale_factor, 0.0f), 1.0f);
        g = fminf(fmaxf(g * scale_factor, 0.0f), 1.0f);
        b = fminf(fmaxf(b * scale_factor, 0.0f), 1.0f);

        tensor[y * net_w + x] = r;
        tensor[net_w * net_h + y * net_w + x] = g;
        tensor[2 * net_w * net_h + y * net_w + x] = b;
    }
}
