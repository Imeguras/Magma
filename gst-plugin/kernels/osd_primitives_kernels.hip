/* OSD primitive kernels — one kernel per primitive type.
 * Each kernel renders one type of primitive (rect, polyline, point, text, etc.)
 * onto an NV12 frame given a flat array of per-type parameters.
 *
 * Compile-time prepend: common.hip (provides clampi, clampf, lerp)
 */

/* ─── GPU parameter structs (must match magma-primitives.h) ───────── */

struct RectParam {
	int x, y, w, h;
	unsigned int color; /* RGBA 8:8:8:8 */
	int filled;
};

struct PolylineParam {
	unsigned int color;
	int line_width;
	int num_verts;
	int verts_offset; /* byte offset into shared vertex buffer */
};

struct PolygonParam {
	unsigned int fill_color;
	unsigned int border_color;
	int num_verts;
	int verts_offset;
};

struct PointParam {
	int x, y;
	unsigned int color;
	int radius;
};

struct TextParam {
	int x, y;
	unsigned int color;
	char text[32];
};

struct ArrowParam {
	int x1, y1, x2, y2;
	unsigned int color;
	int line_width;
	int head_size;
};

/* ─── 5×7 bitmap font (ASCII 32..126) ─────────────────────────────── */

static __device__ const unsigned char font5x7[95][7] = {
    {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, {0x04, 0x04, 0x04, 0x04, 0x04, 0x00, 0x04}, {0x0a, 0x0a, 0x0a, 0x00, 0x00, 0x00, 0x00}, {0x0a, 0x0a, 0x1f, 0x0a, 0x1f, 0x0a, 0x0a},
    {0x0e, 0x15, 0x05, 0x0e, 0x14, 0x15, 0x0e}, {0x00, 0x12, 0x09, 0x04, 0x12, 0x09, 0x00}, {0x0c, 0x12, 0x12, 0x0c, 0x12, 0x12, 0x0c}, {0x04, 0x04, 0x04, 0x00, 0x00, 0x00, 0x00},
    {0x08, 0x04, 0x04, 0x04, 0x04, 0x04, 0x08}, {0x02, 0x04, 0x04, 0x04, 0x04, 0x04, 0x02}, {0x00, 0x04, 0x15, 0x0e, 0x15, 0x04, 0x00}, {0x00, 0x04, 0x04, 0x1f, 0x04, 0x04, 0x00},
    {0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x08}, {0x00, 0x00, 0x00, 0x1f, 0x00, 0x00, 0x00}, {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04}, {0x10, 0x08, 0x08, 0x04, 0x02, 0x02, 0x01},
    {0x0e, 0x11, 0x19, 0x15, 0x13, 0x11, 0x0e}, {0x04, 0x06, 0x04, 0x04, 0x04, 0x04, 0x0e}, {0x0e, 0x11, 0x10, 0x08, 0x04, 0x02, 0x1f}, {0x1f, 0x08, 0x04, 0x08, 0x10, 0x11, 0x0e},
    {0x08, 0x0c, 0x0a, 0x09, 0x1f, 0x08, 0x08}, {0x1f, 0x01, 0x0f, 0x10, 0x10, 0x11, 0x0e}, {0x0c, 0x02, 0x01, 0x0f, 0x11, 0x11, 0x0e}, {0x1f, 0x10, 0x08, 0x04, 0x02, 0x02, 0x02},
    {0x0e, 0x11, 0x11, 0x0e, 0x11, 0x11, 0x0e}, {0x0e, 0x11, 0x11, 0x1e, 0x10, 0x08, 0x06}, {0x00, 0x04, 0x00, 0x00, 0x00, 0x04, 0x00}, {0x00, 0x04, 0x00, 0x00, 0x00, 0x04, 0x08},
    {0x08, 0x04, 0x02, 0x01, 0x02, 0x04, 0x08}, {0x00, 0x00, 0x1f, 0x00, 0x1f, 0x00, 0x00}, {0x02, 0x04, 0x08, 0x10, 0x08, 0x04, 0x02}, {0x0e, 0x11, 0x10, 0x08, 0x04, 0x00, 0x04},
    {0x0e, 0x11, 0x10, 0x16, 0x15, 0x15, 0x0e}, {0x04, 0x0a, 0x11, 0x11, 0x1f, 0x11, 0x11}, {0x0f, 0x11, 0x11, 0x0f, 0x11, 0x11, 0x0f}, {0x0e, 0x11, 0x01, 0x01, 0x01, 0x11, 0x0e},
    {0x0f, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0f}, {0x1f, 0x01, 0x01, 0x0f, 0x01, 0x01, 0x1f}, {0x1f, 0x01, 0x01, 0x0f, 0x01, 0x01, 0x01}, {0x0e, 0x11, 0x01, 0x1d, 0x11, 0x11, 0x0e},
    {0x11, 0x11, 0x11, 0x1f, 0x11, 0x11, 0x11}, {0x0e, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0e}, {0x1c, 0x08, 0x08, 0x08, 0x08, 0x09, 0x06}, {0x11, 0x09, 0x05, 0x03, 0x05, 0x09, 0x11},
    {0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x1f}, {0x11, 0x1b, 0x15, 0x11, 0x11, 0x11, 0x11}, {0x11, 0x13, 0x15, 0x19, 0x11, 0x11, 0x11}, {0x0e, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0e},
    {0x0f, 0x11, 0x11, 0x0f, 0x01, 0x01, 0x01}, {0x0e, 0x11, 0x11, 0x11, 0x15, 0x09, 0x16}, {0x0f, 0x11, 0x11, 0x0f, 0x05, 0x09, 0x11}, {0x0e, 0x11, 0x01, 0x0e, 0x10, 0x11, 0x0e},
    {0x1f, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04}, {0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0e}, {0x11, 0x11, 0x11, 0x11, 0x0a, 0x0a, 0x04}, {0x11, 0x11, 0x11, 0x15, 0x15, 0x1b, 0x11},
    {0x11, 0x11, 0x0a, 0x04, 0x0a, 0x11, 0x11}, {0x11, 0x11, 0x0a, 0x04, 0x04, 0x04, 0x04}, {0x1f, 0x10, 0x08, 0x04, 0x02, 0x01, 0x1f}, {0x0e, 0x02, 0x02, 0x02, 0x02, 0x02, 0x0e},
    {0x01, 0x02, 0x02, 0x04, 0x08, 0x08, 0x10}, {0x0e, 0x08, 0x08, 0x08, 0x08, 0x08, 0x0e}, {0x04, 0x0a, 0x11, 0x00, 0x00, 0x00, 0x00}, {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1f},
    {0x02, 0x04, 0x08, 0x00, 0x00, 0x00, 0x00}, {0x00, 0x00, 0x0e, 0x10, 0x1e, 0x11, 0x1e}, {0x01, 0x01, 0x0f, 0x11, 0x11, 0x11, 0x0f}, {0x00, 0x00, 0x0e, 0x11, 0x01, 0x11, 0x0e},
    {0x10, 0x10, 0x1e, 0x11, 0x11, 0x11, 0x1e}, {0x00, 0x00, 0x0e, 0x11, 0x1f, 0x01, 0x0e}, {0x0c, 0x12, 0x02, 0x07, 0x02, 0x02, 0x02}, {0x00, 0x00, 0x1e, 0x11, 0x1e, 0x10, 0x0e},
    {0x01, 0x01, 0x0f, 0x11, 0x11, 0x11, 0x11}, {0x04, 0x00, 0x06, 0x04, 0x04, 0x04, 0x0e}, {0x08, 0x00, 0x0c, 0x08, 0x08, 0x09, 0x06}, {0x01, 0x01, 0x09, 0x05, 0x03, 0x05, 0x09},
    {0x06, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0e}, {0x00, 0x00, 0x0b, 0x15, 0x15, 0x11, 0x11}, {0x00, 0x00, 0x0f, 0x11, 0x11, 0x11, 0x11}, {0x00, 0x00, 0x0e, 0x11, 0x11, 0x11, 0x0e},
    {0x00, 0x00, 0x0f, 0x11, 0x0f, 0x01, 0x01}, {0x00, 0x00, 0x1e, 0x11, 0x1e, 0x10, 0x10}, {0x00, 0x00, 0x0f, 0x11, 0x01, 0x01, 0x01}, {0x00, 0x00, 0x0e, 0x01, 0x0e, 0x10, 0x0e},
    {0x02, 0x02, 0x0f, 0x02, 0x02, 0x12, 0x0c}, {0x00, 0x00, 0x11, 0x11, 0x11, 0x13, 0x0d}, {0x00, 0x00, 0x11, 0x11, 0x0a, 0x0a, 0x04}, {0x00, 0x00, 0x11, 0x11, 0x15, 0x15, 0x0a},
    {0x00, 0x00, 0x11, 0x0a, 0x04, 0x0a, 0x11}, {0x00, 0x00, 0x11, 0x11, 0x1e, 0x10, 0x0e}, {0x00, 0x00, 0x1f, 0x08, 0x04, 0x02, 0x1f}, {0x08, 0x04, 0x04, 0x02, 0x04, 0x04, 0x08},
    {0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04}, {0x02, 0x04, 0x04, 0x08, 0x04, 0x04, 0x02}, {0x00, 0x00, 0x00, 0x00, 0x12, 0x09, 0x00},
};

/* ─── Colour unpack: RGBA → YUV (NV12) ────────────────────────────── *
 * Simple approximate conversion for overlay colours.
 * Full BT.601 matrix would be more accurate but this is fast and
 * produces visually acceptable results for overlay boxes/lines.
 */

static __device__ void rgba_to_yuv(unsigned int rgba, unsigned char* y, unsigned char* u, unsigned char* v) {
	int r = (rgba >> 24) & 0xFF;
	int g = (rgba >> 16) & 0xFF;
	int b = (rgba >> 8) & 0xFF;
	/* BT.601 approximate */
	*y = (unsigned char)clampi(((66 * r + 129 * g + 25 * b + 128) >> 8) + 16, 16, 235);
	*u = (unsigned char)clampi(((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128, 16, 240);
	*v = (unsigned char)clampi(((112 * r - 94 * g - 18 * b + 128) >> 8) + 128, 16, 240);
}

static __device__ void write_nv12_pixel(unsigned char* y_plane, unsigned char* uv_plane, int y_stride, int uv_stride, int x, int y, unsigned char y_val, unsigned char u_val, unsigned char v_val) {
	y_plane[y * y_stride + x] = y_val;
	int ux = x >> 1;
	int uy = y >> 1;
	uv_plane[uy * uv_stride + (ux << 1)] = u_val;
	uv_plane[uy * uv_stride + (ux << 1) + 1] = v_val;
}

/* ─── Draw a single character from the 5×7 bitmap font ─────────────── */

static __device__ void
draw_char(unsigned char* y_plane, unsigned char* uv_plane, int y_stride, int uv_stride, int x, int y, unsigned char c, unsigned char bg_y, unsigned char bg_u, unsigned char bg_v) {
	if (c < 32 || c > 126)
		return;
	int ci = c - 32;
	/* Contrast-aware glyph colour: dark text on light label backgrounds,
	 * light text on dark ones. 125 is the midpoint of the studio-swing luma
	 * range (16..235); chroma stays neutral so glyphs are pure black/white. */
	const unsigned char fg_y = (bg_y > 125) ? 16 : 235;
	for (int row = 0; row < 7; row++) {
		int yy = y + row;
		if (yy < 0)
			continue;
		unsigned char mask = font5x7[ci][row];
		for (int col = 0; col < 5; col++) {
			int xx = x + col;
			if (xx < 0)
				continue;
			if (mask & (1 << col)) {
				y_plane[yy * y_stride + xx] = fg_y;
				int ux = xx >> 1, uy = yy >> 1;
				uv_plane[uy * uv_stride + (ux << 1)] = 128;
				uv_plane[uy * uv_stride + (ux << 1) + 1] = 128;
			}
		}
	}
	(void)bg_u;
	(void)bg_v;
}

/* ═══════════════════════════════════════════════════════════════════ *
 *  Per-primitive-type kernels                                       *
 * ═══════════════════════════════════════════════════════════════════ */

/* ─── draw_rects_kernel ────────────────────────────────────────────── *
 * One thread per rect. Draws filled or outlined rectangle.             */

extern "C" __global__ void draw_rects_kernel(
    unsigned char* __restrict__ y_plane, unsigned char* __restrict__ uv_plane, int y_stride, int uv_stride, int img_w, int img_h, const RectParam* __restrict__ rects, int num_rects, int line_width) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= num_rects)
		return;

	RectParam r = rects[i];
	int x1 = clampi(r.x, 0, img_w - 1);
	int y1 = clampi(r.y, 0, img_h - 1);
	int x2 = clampi(r.x + r.w - 1, 0, img_w - 1);
	int y2 = clampi(r.y + r.h - 1, 0, img_h - 1);
	if (x2 < x1 || y2 < y1)
		return;

	unsigned char yv, uv, vv;
	rgba_to_yuv(r.color, &yv, &uv, &vv);

	if (r.filled) {
		for (int yy = y1; yy <= y2; yy++)
			for (int xx = x1; xx <= x2; xx++)
				write_nv12_pixel(y_plane, uv_plane, y_stride, uv_stride, xx, yy, yv, uv, vv);
	} else {
		int lw = line_width > 1 ? line_width : 1;
		/* top & bottom edges */
		for (int l = 0; l < lw; l++) {
			int yt = y1 + l;
			int yb = y2 - l;
			if (yt > y2)
				break;
			for (int xx = x1; xx <= x2; xx++) {
				if (yt <= y2)
					write_nv12_pixel(y_plane, uv_plane, y_stride, uv_stride, xx, yt, yv, uv, vv);
				if (yb >= y1 && yb != yt)
					write_nv12_pixel(y_plane, uv_plane, y_stride, uv_stride, xx, yb, yv, uv, vv);
			}
		}
		/* left & right edges (skip corners already drawn) */
		for (int yy = y1 + lw; yy <= y2 - lw; yy++) {
			for (int l = 0; l < lw; l++) {
				int xl = x1 + l;
				int xr = x2 - l;
				if (xl <= x2)
					write_nv12_pixel(y_plane, uv_plane, y_stride, uv_stride, xl, yy, yv, uv, vv);
				if (xr >= x1 && xr != xl)
					write_nv12_pixel(y_plane, uv_plane, y_stride, uv_stride, xr, yy, yv, uv, vv);
			}
		}
	}
}

/* ─── draw_polylines_kernel ─────────────────────────────────────────── *
 * One thread per polyline. Walks line segments between consecutive       *
 * vertices using Bresenham-style iteration (one pixel per step).         */

static __device__ void
draw_line(unsigned char* y_plane, unsigned char* uv_plane, int y_stride, int uv_stride, int img_w, int img_h, int x0, int y0, int x1, int y1, unsigned char yv, unsigned char uv, unsigned char vv) {
	int dx = abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
	int dy = -abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
	int err = dx + dy;
	while (1) {
		if (x0 >= 0 && x0 < img_w && y0 >= 0 && y0 < img_h)
			write_nv12_pixel(y_plane, uv_plane, y_stride, uv_stride, x0, y0, yv, uv, vv);
		if (x0 == x1 && y0 == y1)
			break;
		int e2 = 2 * err;
		if (e2 >= dy) {
			err += dy;
			x0 += sx;
		}
		if (e2 <= dx) {
			err += dx;
			y0 += sy;
		}
	}
}

extern "C" __global__ void draw_polylines_kernel(unsigned char* __restrict__ y_plane,
                                                 unsigned char* __restrict__ uv_plane,
                                                 int y_stride,
                                                 int uv_stride,
                                                 int img_w,
                                                 int img_h,
                                                 const PolylineParam* __restrict__ polylines,
                                                 int num_polylines,
                                                 const int* __restrict__ vertex_arena,
                                                 int arena_bytes) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= num_polylines)
		return;

	PolylineParam pl = polylines[i];
	unsigned char yv, uv, vv;
	rgba_to_yuv(pl.color, &yv, &uv, &vv);

	int base = pl.verts_offset / sizeof(int);
	int nv = pl.num_verts;
	for (int j = 0; j < nv - 1; j++) {
		int idx0 = base + j * 2;
		int idx1 = base + (j + 1) * 2;
		if (idx0 + 1 >= arena_bytes / (int)sizeof(int) || idx1 + 1 >= arena_bytes / (int)sizeof(int))
			break;
		draw_line(y_plane, uv_plane, y_stride, uv_stride, img_w, img_h, vertex_arena[idx0], vertex_arena[idx0 + 1], vertex_arena[idx1], vertex_arena[idx1 + 1], yv, uv, vv);
	}
}

/* ─── Forward declarations ──────────────────────────────────────────── */
static __device__ void write_nv12_pixel_blend(
    unsigned char* y_plane, unsigned char* uv_plane, int y_stride, int uv_stride, int x, int y, unsigned char y_overlay, unsigned char u_overlay, unsigned char v_overlay, unsigned int alpha);

/* ─── draw_polygons_kernel ──────────────────────────────────────────── *
 * One thread per polygon. Fills using a simple scanline approach.       */

extern "C" __global__ void draw_polygons_kernel(unsigned char* __restrict__ y_plane,
                                                unsigned char* __restrict__ uv_plane,
                                                int y_stride,
                                                int uv_stride,
                                                int img_w,
                                                int img_h,
                                                const PolygonParam* __restrict__ polygons,
                                                int num_polygons,
                                                const int* __restrict__ vertex_arena,
                                                int arena_bytes) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= num_polygons)
		return;

	PolygonParam pg = polygons[i];
	unsigned char fy, fu, fv, by, bu, bv;
	unsigned int fill_alpha = (pg.fill_color >> 0) & 0xFF;
	rgba_to_yuv(pg.fill_color, &fy, &fu, &fv);
	rgba_to_yuv(pg.border_color, &by, &bu, &bv);

	int base = pg.verts_offset / sizeof(int);
	int nv = pg.num_verts;
	if (nv < 3)
		return;

	/* Find bounding box */
	int min_y = img_h, max_y = 0;
	for (int j = 0; j < nv; j++) {
		int v_idx = base + j * 2;
		if (v_idx + 1 >= arena_bytes / (int)sizeof(int))
			break;
		int vy = vertex_arena[v_idx + 1];
		if (vy < min_y)
			min_y = vy;
		if (vy > max_y)
			max_y = vy;
	}
	min_y = clampi(min_y, 0, img_h - 1);
	max_y = clampi(max_y, 0, img_h - 1);

	/* Scanline fill with alpha blending + solid border */
	for (int yy = min_y; yy <= max_y; yy++) {
		int hits[128];
		int nh = 0;
		for (int j = 0; j < nv; j++) {
			int j_n = (j + 1) % nv;
			int idx0 = base + j * 2;
			int idx1 = base + j_n * 2;
			if (idx0 + 1 >= arena_bytes / (int)sizeof(int) || idx1 + 1 >= arena_bytes / (int)sizeof(int))
				break;
			int x0 = vertex_arena[idx0], y0 = vertex_arena[idx0 + 1];
			int x1 = vertex_arena[idx1], y1 = vertex_arena[idx1 + 1];
			if ((y0 > yy) == (y1 > yy))
				continue;
			int x_hit = x0 + (int)((float)(yy - y0) / (float)(y1 - y0) * (float)(x1 - x0));
			if (nh < 128)
				hits[nh++] = x_hit;
		}
		/* Sort hits */
		for (int j = 0; j < nh - 1; j++)
			for (int k = j + 1; k < nh; k++)
				if (hits[j] > hits[k]) {
					int t = hits[j];
					hits[j] = hits[k];
					hits[k] = t;
				}
		/* Fill spans with alpha blend */
		for (int j = 0; j + 1 < nh; j += 2) {
			int xs = clampi(hits[j], 0, img_w - 1);
			int xe = clampi(hits[j + 1], 0, img_w - 1);
			/* First/last border pixels — solid border_color */
			if (fill_alpha > 0 && fill_alpha < 255) {
				/* Interior: alpha blend */
				for (int xx = xs + 1; xx < xe; xx++)
					write_nv12_pixel_blend(y_plane, uv_plane, y_stride, uv_stride, xx, yy, fy, fu, fv, fill_alpha);
				/* Border: solid */
				write_nv12_pixel(y_plane, uv_plane, y_stride, uv_stride, xs, yy, by, bu, bv);
				if (xe > xs)
					write_nv12_pixel(y_plane, uv_plane, y_stride, uv_stride, xe, yy, by, bu, bv);
			} else {
				for (int xx = xs; xx <= xe; xx++)
					write_nv12_pixel(y_plane, uv_plane, y_stride, uv_stride, xx, yy, fy, fu, fv);
			}
		}
	}
}

/* ─── draw_points_kernel ────────────────────────────────────────────── *
 * One thread per point. Draws a filled circle using midpoint algorithm.  */

extern "C" __global__ void draw_points_kernel(
    unsigned char* __restrict__ y_plane, unsigned char* __restrict__ uv_plane, int y_stride, int uv_stride, int img_w, int img_h, const PointParam* __restrict__ points, int num_points) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= num_points)
		return;

	PointParam pt = points[i];
	unsigned char yv, uv, vv;
	rgba_to_yuv(pt.color, &yv, &uv, &vv);

	int cx = pt.x, cy = pt.y, r = pt.radius;
	if (r < 1)
		r = 1;
	for (int yy = cy - r; yy <= cy + r; yy++) {
		if (yy < 0 || yy >= img_h)
			continue;
		for (int xx = cx - r; xx <= cx + r; xx++) {
			if (xx < 0 || xx >= img_w)
				continue;
			int dx = xx - cx, dy = yy - cy;
			if (dx * dx + dy * dy <= r * r)
				write_nv12_pixel(y_plane, uv_plane, y_stride, uv_stride, xx, yy, yv, uv, vv);
		}
	}
}

/* ─── draw_text_kernel ──────────────────────────────────────────────── *
 * One thread per text label. Draws a coloured background strip and       *
 * renders the 5×7 bitmap font string.                                   */

extern "C" __global__ void
draw_text_kernel(unsigned char* __restrict__ y_plane, unsigned char* __restrict__ uv_plane, int y_stride, int uv_stride, int img_w, int img_h, const TextParam* __restrict__ texts, int num_texts) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= num_texts)
		return;

	TextParam t = texts[i];
	unsigned char yv, uv, vv;
	rgba_to_yuv(t.color, &yv, &uv, &vv);

	int label_len = 0;
	while (label_len < 31 && t.text[label_len])
		label_len++;
	if (label_len == 0)
		return;

	int char_w = 6;
	int label_h = 8;
	int label_w = label_len * char_w;

	int lx = t.x;
	int ly = t.y;
	if (lx + label_w > img_w)
		lx = img_w - label_w;
	if (lx < 0)
		lx = 0;
	if (ly + label_h > img_h)
		ly = img_h - label_h;
	if (ly < 0)
		ly = 0;

	/* Background strip */
	for (int yy = ly; yy < ly + label_h; yy++) {
		if (yy >= img_h)
			break;
		for (int xx = lx; xx < lx + label_w; xx++) {
			if (xx >= img_w)
				break;
			write_nv12_pixel(y_plane, uv_plane, y_stride, uv_stride, xx, yy, yv, uv, vv);
		}
	}

	/* Characters */
	int cx = lx;
	for (int ci = 0; ci < label_len; ci++) {
		draw_char(y_plane, uv_plane, y_stride, uv_stride, cx, ly, t.text[ci], yv, uv, vv);
		cx += char_w;
	}
}

/* ─── Alpha-blended NV12 pixel write ───────────────────────────────────*
 * Blends (overlay_yuv * alpha + bg_yuv * (255-alpha)) / 255            */

static __device__ void write_nv12_pixel_blend(
    unsigned char* y_plane, unsigned char* uv_plane, int y_stride, int uv_stride, int x, int y, unsigned char y_overlay, unsigned char u_overlay, unsigned char v_overlay, unsigned int alpha) {
	if (alpha >= 254) {
		write_nv12_pixel(y_plane, uv_plane, y_stride, uv_stride, x, y, y_overlay, u_overlay, v_overlay);
		return;
	}
	if (alpha == 0)
		return;

	unsigned char y_bg = y_plane[y * y_stride + x];
	int y_new = ((int)alpha * y_overlay + (int)(255 - alpha) * y_bg) / 255;
	y_plane[y * y_stride + x] = (unsigned char)clampi(y_new, 16, 235);

	int ux = x >> 1;
	int uy = y >> 1;
	unsigned char u_bg = uv_plane[uy * uv_stride + (ux << 1)];
	unsigned char v_bg = uv_plane[uy * uv_stride + (ux << 1) + 1];
	int u_new = ((int)alpha * u_overlay + (int)(255 - alpha) * u_bg) / 255;
	int v_new = ((int)alpha * v_overlay + (int)(255 - alpha) * v_bg) / 255;
	uv_plane[uy * uv_stride + (ux << 1)] = (unsigned char)clampi(u_new, 16, 240);
	uv_plane[uy * uv_stride + (ux << 1) + 1] = (unsigned char)clampi(v_new, 16, 240);
}

/* ─── draw_arrows_kernel ────────────────────────────────────────────── *
 * One thread per arrow. Draws a line with a triangular arrowhead.       */

extern "C" __global__ void draw_arrows_kernel(
    unsigned char* __restrict__ y_plane, unsigned char* __restrict__ uv_plane, int y_stride, int uv_stride, int img_w, int img_h, const ArrowParam* __restrict__ arrows, int num_arrows) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= num_arrows)
		return;

	ArrowParam a = arrows[i];
	unsigned char yv, uv, vv;
	rgba_to_yuv(a.color, &yv, &uv, &vv);

	/* Draw shaft */
	draw_line(y_plane, uv_plane, y_stride, uv_stride, img_w, img_h, a.x1, a.y1, a.x2, a.y2, yv, uv, vv);

	/* Draw arrowhead if requested */
	if (a.head_size > 0) {
		float dx = (float)(a.x2 - a.x1);
		float dy = (float)(a.y2 - a.y1);
		float len = sqrtf(dx * dx + dy * dy);
		if (len > 1.0f) {
			float ux = dx / len;
			float uy = dy / len;
			float hs = (float)a.head_size;
			int p1x = a.x2 - (int)(ux * hs - uy * hs * 0.5f);
			int p1y = a.y2 - (int)(uy * hs + ux * hs * 0.5f);
			int p2x = a.x2 - (int)(ux * hs + uy * hs * 0.5f);
			int p2y = a.y2 - (int)(uy * hs - ux * hs * 0.5f);
			draw_line(y_plane, uv_plane, y_stride, uv_stride, img_w, img_h, a.x2, a.y2, p1x, p1y, yv, uv, vv);
			draw_line(y_plane, uv_plane, y_stride, uv_stride, img_w, img_h, a.x2, a.y2, p2x, p2y, yv, uv, vv);
		}
	}
}
