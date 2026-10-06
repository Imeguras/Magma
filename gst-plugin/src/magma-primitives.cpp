// SPDX-FileCopyrightText: 2026 João Vieira <joaodavid2001@gmail.com>
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "magma-primitives.h"
#include "magma-meta.h"
#include "magma-infer-meta.h"

#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <sys/stat.h>
#include <unordered_map>
#include <string>
#include <mutex>
#include <gst/gst.h>

/* ─── Registry (lazy-initialized to avoid static init order fiasco) ── */

static std::unordered_map<std::string, MagmaToPrimitivesFunc>& get_registry() {
	static std::unordered_map<std::string, MagmaToPrimitivesFunc>* reg = new std::unordered_map<std::string, MagmaToPrimitivesFunc>();
	return *reg;
}

static std::mutex& get_registry_mutex() {
	static std::mutex* mtx = new std::mutex();
	return *mtx;
}

void magma_register_to_primitives(const char* type_id_str, MagmaToPrimitivesFunc func) {
	if (!type_id_str || !func)
		return;
	std::lock_guard<std::mutex> lock(get_registry_mutex());
	get_registry()[std::string(type_id_str)] = func;
}

MagmaToPrimitivesFunc magma_lookup_to_primitives(const char* type_id_str) {
	if (!type_id_str)
		return nullptr;
	std::lock_guard<std::mutex> lock(get_registry_mutex());
	auto& reg = get_registry();
	auto it = reg.find(std::string(type_id_str));
	if (it != reg.end())
		return it->second;
	/* Prefix match: "magma.detection.*" → default converter */
	if (strncmp(type_id_str, "magma.detection.", 16) == 0) {
		return magma_default_detection_to_primitives;
	}
	return nullptr;
}

/* ─── Primitive list lifecycle ────────────────────────────────────── */

void magma_primitive_list_init(MagmaPrimitiveList* pl) {
	memset(pl, 0, sizeof(*pl));
}

void magma_primitive_list_reset(MagmaPrimitiveList* pl) {
	pl->num_rects = 0;
	pl->num_polylines = 0;
	pl->num_polygons = 0;
	pl->num_points = 0;
	pl->num_texts = 0;
	pl->num_arrows = 0;
	pl->vertex_arena_used = 0;
	pl->total_primitives = 0;
}

void magma_primitive_list_destroy(MagmaPrimitiveList* pl) {
	free(pl->rects);
	pl->rects = nullptr;
	free(pl->polylines);
	pl->polylines = nullptr;
	free(pl->polygons);
	pl->polygons = nullptr;
	free(pl->points);
	pl->points = nullptr;
	free(pl->texts);
	pl->texts = nullptr;
	free(pl->arrows);
	pl->arrows = nullptr;
	free(pl->vertex_arena);
	pl->vertex_arena = nullptr;
	pl->rects_cap = pl->polylines_cap = pl->polygons_cap = 0;
	pl->points_cap = pl->texts_cap = pl->arrows_cap = 0;
	pl->polygons_cap = pl->polylines_cap = 0;
	pl->vertex_arena_cap = 0;
	magma_primitive_list_reset(pl);
}

/* ─── Labels file cache ─────────────────────────────────────────────
 *
 * Loads COCO-format label files on demand and caches them by
 * (realpath + mtime) so repeated lookups don't re-read the file.
 */

struct LabelCacheEntry {
	std::string path;
	time_t mtime;
	char** names; /* malloc'd array of malloc'd strings */
	int count;
};

static std::unordered_map<std::string, LabelCacheEntry>& get_label_cache() {
	static auto* cache = new std::unordered_map<std::string, LabelCacheEntry>();
	return *cache;
}

static std::mutex& get_label_cache_mutex() {
	static std::mutex* mtx = new std::mutex();
	return *mtx;
}

const char* magma_lookup_class_name(unsigned int class_id, const char* labels_path) {
	if (!labels_path || !labels_path[0])
		return nullptr;

	/* Resolve real path for stable cache key */
	char real[4096];
	if (!realpath(labels_path, real))
		return nullptr;

	struct stat st;
	if (stat(real, &st) != 0)
		return nullptr;

	std::lock_guard<std::mutex> lock(get_label_cache_mutex());
	auto& cache = get_label_cache();
	auto it = cache.find(real);

	if (it == cache.end() || it->second.mtime != st.st_mtime) {
		/* Load or reload */
		LabelCacheEntry entry;
		entry.path = real;
		entry.mtime = st.st_mtime;

		FILE* f = fopen(real, "r");
		if (!f)
			return nullptr;

		/* First pass: count lines */
		int count = 0;
		int ch;
		while ((ch = fgetc(f)) != EOF) {
			if (ch == '\n')
				count++;
		}
		rewind(f);

		entry.count = count;
		entry.names = (char**)malloc(sizeof(char*) * (size_t)count);
		char buf[1024];
		int idx = 0;
		while (idx < count && fgets(buf, sizeof(buf), f)) {
			size_t len = strlen(buf);
			while (len > 0 && (buf[len - 1] == '\n' || buf[len - 1] == '\r'))
				buf[--len] = '\0';
			if (len > 0) {
				entry.names[idx] = strdup(buf);
			} else {
				entry.names[idx] = strdup("");
			}
			idx++;
		}
		fclose(f);

		if (it != cache.end())
			cache.erase(it);

		it = cache.emplace(std::string(real), entry).first;
	}

	if ((int)class_id < it->second.count && it->second.names[class_id][0])
		return it->second.names[class_id];
	return nullptr;
}

/* ─── Default detection converter (CPU data path) ──────────────────
 *
 * Reads MagmaInferObjectGPU[] from cpu_data and produces rect + text
 * primitives. Coordinates should already be in source-pixel space
 * (the caller remaps before attaching the semantic meta).
 *
 * If params->labels_path is set, class names are looked up from the
 * file instead of printing the numeric class_id.
 */

int magma_default_detection_to_primitives(const MagmaToPrimitivesParams* params, MagmaPrimitiveList* out) {
	if (!params || !out)
		return -1;

	/* Use CPU data path (pre-mapped by caller) */
	const void* data = params->cpu_data ? params->cpu_data : params->d_data;
	int data_size = params->cpu_data ? params->cpu_data_size : params->data_size;
	if (!data || data_size < (int)sizeof(MagmaInferObjectGPU))
		return 0;

	int num = data_size / (int)sizeof(MagmaInferObjectGPU);
	if (num > 500)
		num = 500; /* safety cap */

	const MagmaInferObjectGPU* objs = (const MagmaInferObjectGPU*)data;

	static const uint32_t palette_rgba[] = {
	    0xFF0000FF,
	    0xFF00FF00,
	    0xFFFF0000,
	    0xFFFFFF00,
	    0xFFFF00FF,
	    0xFF00FFFF,
	    0xFFFFA500,
	    0xFF800080,
	    0xFF87CEEB,
	    0xFFFFC0CB,
	    0xFF00FF80,
	    0xFF7FFF00,
	    0xFFFF1493,
	    0xFF00BFFF,
	    0xFF98FB98,
	    0xFFDDA0DD,
	};

	for (int i = 0; i < num; i++) {
		uint32_t c = palette_rgba[objs[i].class_id % 16];
		int sx = params->roi_x + (int)(objs[i].x * (params->roi_w > 0 ? params->roi_w : params->source_width));
		int sy = params->roi_y + (int)(objs[i].y * (params->roi_h > 0 ? params->roi_h : params->source_height));
		int sw = (int)(objs[i].width * (params->roi_w > 0 ? params->roi_w : params->source_width));
		int sh = (int)(objs[i].height * (params->roi_h > 0 ? params->roi_h : params->source_height));
		if (sw < 2 || sh < 2)
			continue;

		magma_primitive_list_add_rect(out, sx, sy, sw, sh, c, 0);

		char label[MAGMA_LABEL_MAX];
		const char* name = magma_lookup_class_name(objs[i].class_id, params->labels_path);
		if (name) {
			int n = snprintf(label, sizeof(label), "%s %.0f%%", name, objs[i].confidence * 100.0f);
			if (n > 0)
				magma_primitive_list_add_text(out, sx, sy - 12, c, label, n);
		} else {
			int n = snprintf(label, sizeof(label), "%u %.0f%%", objs[i].class_id, objs[i].confidence * 100.0f);
			if (n > 0)
				magma_primitive_list_add_text(out, sx, sy - 12, c, label, n);
		}
	}
	return 0;
}

/* ─── Auto-register at library load ───────────────────────────────── */

void magma_primitives_auto_register(void) {
	magma_register_to_primitives("magma.detection.v1", magma_default_detection_to_primitives);
}
