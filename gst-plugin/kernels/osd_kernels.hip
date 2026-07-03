/* OSD kernel: draw colored rectangle outlines on NV12.
 * Compile-time prepend: common.hip (provides clampi)
 */

struct BoxParam {
    int x, y, w, h;  /* source pixel coords */
    unsigned char y_val, u_val, v_val;
    unsigned char _pad[1];  /* struct is 20 bytes */
};

extern "C" {

__global__ void draw_boxes_kernel(
    unsigned char* __restrict__ y_plane,
    unsigned char* __restrict__ uv_plane,
    int y_stride, int uv_stride,
    int img_w, int img_h,
    const BoxParam* __restrict__ boxes,
    int num_boxes, int line_width)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_boxes) return;

    BoxParam b = boxes[i];
    int x1 = clampi(b.x, 0, img_w - 1);
    int y1 = clampi(b.y, 0, img_h - 1);
    int x2 = clampi(b.x + b.w - 1, 0, img_w - 1);
    int y2 = clampi(b.y + b.h - 1, 0, img_h - 1);

    /* top edge */
    for (int l = 0; l < line_width; l++) {
        int yy = y1 + l;
        if (yy > y2) break;
        for (int x = x1; x <= x2; x++) {
            y_plane[yy * y_stride + x] = b.y_val;
            int ux = x >> 1, uy = yy >> 1;
            uv_plane[uy * uv_stride + (ux << 1)    ] = b.u_val;
            uv_plane[uy * uv_stride + (ux << 1) + 1] = b.v_val;
        }
    }

    /* bottom edge */
    for (int l = 0; l < line_width; l++) {
        int yy = y2 - l;
        if (yy < y1) break;
        for (int x = x1; x <= x2; x++) {
            y_plane[yy * y_stride + x] = b.y_val;
            int ux = x >> 1, uy = yy >> 1;
            uv_plane[uy * uv_stride + (ux << 1)    ] = b.u_val;
            uv_plane[uy * uv_stride + (ux << 1) + 1] = b.v_val;
        }
    }

    /* left edge (skip corners) */
    for (int yy = y1 + line_width; yy <= y2 - line_width; yy++) {
        for (int l = 0; l < line_width; l++) {
            int xx = x1 + l;
            if (xx > x2) break;
            y_plane[yy * y_stride + xx] = b.y_val;
            int ux = xx >> 1, uy = yy >> 1;
            uv_plane[uy * uv_stride + (ux << 1)    ] = b.u_val;
            uv_plane[uy * uv_stride + (ux << 1) + 1] = b.v_val;
        }
    }

    /* right edge (skip corners) */
    for (int yy = y1 + line_width; yy <= y2 - line_width; yy++) {
        for (int l = 0; l < line_width; l++) {
            int xx = x2 - l;
            if (xx < x1) break;
            y_plane[yy * y_stride + xx] = b.y_val;
            int ux = xx >> 1, uy = yy >> 1;
            uv_plane[uy * uv_stride + (ux << 1)    ] = b.u_val;
            uv_plane[uy * uv_stride + (ux << 1) + 1] = b.v_val;
        }
    }
}

} /* extern "C" */
