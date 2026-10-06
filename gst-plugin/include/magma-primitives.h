#pragma once

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ─── Primitive type enum ──────────────────────────────────────────── */
typedef enum {
	MAGMA_PRIMITIVE_RECT = 0,
	MAGMA_PRIMITIVE_POLYLINE = 1,
	MAGMA_PRIMITIVE_POLYGON = 2,
	MAGMA_PRIMITIVE_POINT = 3,
	MAGMA_PRIMITIVE_TEXT = 4,
	MAGMA_PRIMITIVE_ARROW = 5,
	MAGMA_PRIMITIVE_TICKER = 6,
} MagmaPrimitiveType;

/* ─── GPU-side primitive structs (one per type) ───────────────────── */
#define MAGMA_LABEL_MAX 32

typedef struct {
	int x, y, w, h;
	uint32_t color; /* RGBA 8:8:8:8 */
	int filled;
} MagmaRectGpu;

typedef struct {
	uint32_t color;
	int line_width;
	int num_verts;
	int verts_offset; /* byte offset into shared vertex arena */
} MagmaPolylineGpu;

typedef struct {
	uint32_t fill_color;
	uint32_t border_color;
	int num_verts;
	int verts_offset;
} MagmaPolygonGpu;

typedef struct {
	int x, y;
	uint32_t color;
	int radius;
} MagmaPointGpu;

typedef struct {
	int x, y;
	uint32_t color;
	char text[MAGMA_LABEL_MAX];
} MagmaTextGpu;

typedef struct {
	int x1, y1, x2, y2;
	uint32_t color;
	int line_width;
	int head_size;
} MagmaArrowGpu;

/* ─── Host-side primitive list (zero-copy arena for variable data) ── */
typedef struct _MagmaPrimitiveList {
	int num_rects;
	int rects_cap;
	MagmaRectGpu* rects;

	int num_polylines;
	int polylines_cap;
	MagmaPolylineGpu* polylines;

	int num_polygons;
	int polygons_cap;
	MagmaPolygonGpu* polygons;

	int num_points;
	int points_cap;
	MagmaPointGpu* points;

	int num_texts;
	int texts_cap;
	MagmaTextGpu* texts;

	int num_arrows;
	int arrows_cap;
	MagmaArrowGpu* arrows;

	int* vertex_arena;
	int vertex_arena_cap;
	int vertex_arena_used;

	int total_primitives;
} MagmaPrimitiveList;

/* ─── Converter parameters ────────────────────────────────────────── */
typedef struct _MagmaToPrimitivesParams {
	const char* type_id; /* semantic type string */
	const void* d_data;  /* GPU device pointer, or NULL */
	int data_size;       /* bytes in GPU/CPU data */
	int source_width;    /* source frame dimensions */
	int source_height;
	int model_width; /* model input dimensions */
	int model_height;
	int roi_x, roi_y, roi_w, roi_h; /* ROI within source */
	void* stream;                   /* HIP stream */
	/* CPU fallback: pre-mapped data for converters that can't use GPU */
	const void* cpu_data;
	int cpu_data_size;
	/* Class name labels: path to a COCO-format labels file (one name per line,
	   line number = class_id).  If NULL, class_id is printed numerically. */
	const char* labels_path;

	/* Optional instance segmentation masks (CPU data, one per detection in same order) */
	const float* mask_data;
	int mask_bytes;
	int mask_count;
	int mask_h;
	int mask_w;

	/* GPU-resident mask data for GPU decode path (0 = not set) */
	const void* d_masks_gpu;
	const void* d_objects_gpu;
	int d_masks_bytes;
} MagmaToPrimitivesParams;

/* ─── Converter function type ─────────────────────────────────────── */
typedef int (*MagmaToPrimitivesFunc)(const MagmaToPrimitivesParams* params, MagmaPrimitiveList* out);

/* ─── Registry API ────────────────────────────────────────────────── */
void magma_register_to_primitives(const char* type_id_str, MagmaToPrimitivesFunc func);
MagmaToPrimitivesFunc magma_lookup_to_primitives(const char* type_id_str);

/* ─── Primitive list lifecycle ────────────────────────────────────── */
void magma_primitive_list_init(MagmaPrimitiveList* pl);
void magma_primitive_list_reset(MagmaPrimitiveList* pl);
void magma_primitive_list_destroy(MagmaPrimitiveList* pl);

/* ─── Class name lookup ─────────────────────────────────────────────
 * Loads a COCO-format labels file (one name per line, line=class_id)
 * and caches it.  Returns NULL if no label file is set or class_id
 * is out of range.  Thread-safe.
 */
const char* magma_lookup_class_name(unsigned int class_id, const char* labels_path);

/* ─── Inline append helpers ─────────────────────────────────────────
 *
 * These are static inline so parser .so files (which do not link
 * against libmagma-meta) can use them directly. Memory growth uses
 * realloc(NULL, ...) which is equivalent to malloc() on first call,
 * so the list does not require init before first append (though
 * magma_primitive_list_init() should be called for clarity).
 */

static inline int magma_primitive_list_add_rect(MagmaPrimitiveList* pl, int x, int y, int w, int h, uint32_t color, int filled) {
	if (pl->num_rects >= pl->rects_cap) {
		int new_cap = pl->rects_cap == 0 ? 64 : pl->rects_cap * 2;
		void* np = realloc(pl->rects, (size_t)new_cap * sizeof(MagmaRectGpu));
		if (!np)
			return -1;
		pl->rects = (MagmaRectGpu*)np;
		pl->rects_cap = new_cap;
	}
	MagmaRectGpu* r = &pl->rects[pl->num_rects++];
	r->x = x;
	r->y = y;
	r->w = w;
	r->h = h;
	r->color = color;
	r->filled = filled;
	pl->total_primitives++;
	return 0;
}

static inline int magma_primitive_list_add_polyline(MagmaPrimitiveList* pl, const int* verts, int num_verts, uint32_t color, int line_width) {
	if (num_verts < 2)
		return 0;
	if (pl->num_polylines >= pl->polylines_cap) {
		int new_cap = pl->polylines_cap == 0 ? 64 : pl->polylines_cap * 2;
		void* np = realloc(pl->polylines, (size_t)new_cap * sizeof(MagmaPolylineGpu));
		if (!np)
			return -1;
		pl->polylines = (MagmaPolylineGpu*)np;
		pl->polylines_cap = new_cap;
	}
	int verts_needed = num_verts * 2;
	while (pl->vertex_arena_used + verts_needed > pl->vertex_arena_cap) {
		int new_cap = pl->vertex_arena_cap == 0 ? 512 : pl->vertex_arena_cap * 2;
		void* np = realloc(pl->vertex_arena, (size_t)new_cap * sizeof(int));
		if (!np)
			return -1;
		pl->vertex_arena = (int*)np;
		pl->vertex_arena_cap = new_cap;
	}
	MagmaPolylineGpu* p = &pl->polylines[pl->num_polylines++];
	p->color = color;
	p->line_width = line_width;
	p->num_verts = num_verts;
	p->verts_offset = pl->vertex_arena_used * (int)sizeof(int);
	memcpy(&pl->vertex_arena[pl->vertex_arena_used], verts, (size_t)verts_needed * sizeof(int));
	pl->vertex_arena_used += verts_needed;
	pl->total_primitives++;
	return 0;
}

static inline int magma_primitive_list_add_polygon(MagmaPrimitiveList* pl, const int* verts, int num_verts, uint32_t fill_color, uint32_t border_color) {
	if (num_verts < 3)
		return 0;
	if (pl->num_polygons >= pl->polygons_cap) {
		int new_cap = pl->polygons_cap == 0 ? 64 : pl->polygons_cap * 2;
		void* np = realloc(pl->polygons, (size_t)new_cap * sizeof(MagmaPolygonGpu));
		if (!np)
			return -1;
		pl->polygons = (MagmaPolygonGpu*)np;
		pl->polygons_cap = new_cap;
	}
	int verts_needed = num_verts * 2;
	while (pl->vertex_arena_used + verts_needed > pl->vertex_arena_cap) {
		int new_cap = pl->vertex_arena_cap == 0 ? 512 : pl->vertex_arena_cap * 2;
		void* np = realloc(pl->vertex_arena, (size_t)new_cap * sizeof(int));
		if (!np)
			return -1;
		pl->vertex_arena = (int*)np;
		pl->vertex_arena_cap = new_cap;
	}
	MagmaPolygonGpu* p = &pl->polygons[pl->num_polygons++];
	p->fill_color = fill_color;
	p->border_color = border_color;
	p->num_verts = num_verts;
	p->verts_offset = pl->vertex_arena_used * (int)sizeof(int);
	memcpy(&pl->vertex_arena[pl->vertex_arena_used], verts, (size_t)verts_needed * sizeof(int));
	pl->vertex_arena_used += verts_needed;
	pl->total_primitives++;
	return 0;
}

static inline int magma_primitive_list_add_point(MagmaPrimitiveList* pl, int x, int y, uint32_t color, int radius) {
	if (pl->num_points >= pl->points_cap) {
		int new_cap = pl->points_cap == 0 ? 64 : pl->points_cap * 2;
		void* np = realloc(pl->points, (size_t)new_cap * sizeof(MagmaPointGpu));
		if (!np)
			return -1;
		pl->points = (MagmaPointGpu*)np;
		pl->points_cap = new_cap;
	}
	MagmaPointGpu* p = &pl->points[pl->num_points++];
	p->x = x;
	p->y = y;
	p->color = color;
	p->radius = radius;
	pl->total_primitives++;
	return 0;
}

static inline int magma_primitive_list_add_text(MagmaPrimitiveList* pl, int x, int y, uint32_t color, const char* text, int len) {
	if (len <= 0 || !text)
		len = 0;
	if (len > MAGMA_LABEL_MAX - 1)
		len = MAGMA_LABEL_MAX - 1;
	if (pl->num_texts >= pl->texts_cap) {
		int new_cap = pl->texts_cap == 0 ? 64 : pl->texts_cap * 2;
		void* np = realloc(pl->texts, (size_t)new_cap * sizeof(MagmaTextGpu));
		if (!np)
			return -1;
		pl->texts = (MagmaTextGpu*)np;
		pl->texts_cap = new_cap;
	}
	MagmaTextGpu* t = &pl->texts[pl->num_texts++];
	t->x = x;
	t->y = y;
	t->color = color;
	memset(t->text, 0, MAGMA_LABEL_MAX);
	if (len > 0)
		memcpy(t->text, text, (size_t)len);
	pl->total_primitives++;
	return 0;
}

static inline int magma_primitive_list_add_arrow(MagmaPrimitiveList* pl, int x1, int y1, int x2, int y2, uint32_t color, int line_width, int head_size) {
	if (pl->num_arrows >= pl->arrows_cap) {
		int new_cap = pl->arrows_cap == 0 ? 64 : pl->arrows_cap * 2;
		void* np = realloc(pl->arrows, (size_t)new_cap * sizeof(MagmaArrowGpu));
		if (!np)
			return -1;
		pl->arrows = (MagmaArrowGpu*)np;
		pl->arrows_cap = new_cap;
	}
	MagmaArrowGpu* a = &pl->arrows[pl->num_arrows++];
	a->x1 = x1;
	a->y1 = y1;
	a->x2 = x2;
	a->y2 = y2;
	a->color = color;
	a->line_width = line_width;
	a->head_size = head_size;
	pl->total_primitives++;
	return 0;
}

/* ─── Default converter ───────────────────────────────────────────── */
int magma_default_detection_to_primitives(const MagmaToPrimitivesParams* params, MagmaPrimitiveList* out);

/* ─── Auto-register (called from magma-meta.cpp constructor) ──────── */
void magma_primitives_auto_register(void);

#ifdef __cplusplus
}
#endif
