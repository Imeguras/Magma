#include "magma-meta.h"

static gsize _magma_tensor_meta_quark = 0;

GType magma_tensor_meta_api_get_type(void) {
	if (g_once_init_enter(&_magma_tensor_meta_quark)) {
		const gchar* tags[] = {NULL};
		GType t = gst_meta_api_type_register("MagmaTensorMetaAPI", tags);
		g_once_init_leave(&_magma_tensor_meta_quark, (gsize)t);
	}
	return (GType)_magma_tensor_meta_quark;
}

static gboolean magma_tensor_meta_init(GstMeta* meta, gpointer params, GstBuffer* buffer) {
	MagmaTensorMeta* m = (MagmaTensorMeta*)meta;
	m->tensor_mem = NULL;
	m->width = 0;
	m->height = 0;
	m->channels = 0;
	m->roi_x = m->roi_y = m->roi_w = m->roi_h = 0;
	return TRUE;
}

static void magma_tensor_meta_free(GstMeta* meta, GstBuffer* buffer) {
	MagmaTensorMeta* m = (MagmaTensorMeta*)meta;
	if (m->tensor_mem) {
		gst_memory_unref(m->tensor_mem);
		m->tensor_mem = NULL;
	}
}

static gboolean magma_tensor_meta_transform(GstBuffer* transbuf, GstMeta* meta, GstBuffer* buffer, GQuark type, gpointer data) {
	if (type != 0)
		return FALSE;
	MagmaTensorMeta* src = (MagmaTensorMeta*)meta;
	MagmaTensorMeta* dest;
	dest = magma_buffer_add_tensor_meta(transbuf, src->tensor_mem, src->width, src->height, src->channels);
	if (dest) {
		dest->roi_x = src->roi_x;
		dest->roi_y = src->roi_y;
		dest->roi_w = src->roi_w;
		dest->roi_h = src->roi_h;
	}
	return dest != NULL;
}

static gsize _magma_tensor_meta_info = 0;

const GstMetaInfo* magma_tensor_meta_get_info(void) {
	if (g_once_init_enter(&_magma_tensor_meta_info)) {
		const GstMetaInfo* info =
		    gst_meta_register(MAGMA_TENSOR_META_API_TYPE, "MagmaTensorMeta", sizeof(MagmaTensorMeta), magma_tensor_meta_init, magma_tensor_meta_free, magma_tensor_meta_transform);
		if (!info)
			info = gst_meta_get_info("MagmaTensorMeta");
		g_once_init_leave(&_magma_tensor_meta_info, (gsize)info);
	}
	return (const GstMetaInfo*)_magma_tensor_meta_info;
}

/**
 * @brief Attach a MagmaTensorMeta with preprocessed tensor data to a buffer.
 *
 * Stores the DMABuf-backed tensor memory and its dimensions in the
 * meta for downstream elements (mgminfer, mgmtensordump) to consume.
 *
 * @param buffer     Target GstBuffer
 * @param tensor_mem DMABuf-backed GstMemory with float32 RGB tensor
 * @param width      Tensor width
 * @param height     Tensor height
 * @param channels   Number of channels (typically 3)
 * @return Pointer to the attached meta, or NULL on failure
 */
MagmaTensorMeta* magma_buffer_add_tensor_meta(GstBuffer* buffer, GstMemory* tensor_mem, gint width, gint height, gint channels) {
	MagmaTensorMeta* m = (MagmaTensorMeta*)gst_buffer_add_meta(buffer, magma_tensor_meta_get_info(), NULL);
	if (m) {
		m->tensor_mem = tensor_mem ? gst_memory_ref(tensor_mem) : NULL;
		m->width = width;
		m->height = height;
		m->channels = channels;
	}
	return m;
}

// ─── MagmaHipMeta ────────────────────────────────────────────────────

static gsize _magma_hip_meta_quark = 0;

GType magma_hip_meta_api_get_type(void) {
	if (g_once_init_enter(&_magma_hip_meta_quark)) {
		const gchar* tags[] = {NULL};
		GType t = gst_meta_api_type_register("MagmaHipMetaAPI", tags);
		g_once_init_leave(&_magma_hip_meta_quark, (gsize)t);
	}
	return (GType)_magma_hip_meta_quark;
}

static gboolean magma_hip_meta_init(GstMeta* meta, gpointer params, GstBuffer* buffer) {
	MagmaHipMeta* m = (MagmaHipMeta*)meta;
	m->d_ptr = 0;
	m->release = NULL;
	m->user_data = NULL;
	return TRUE;
}

static void magma_hip_meta_free(GstMeta* meta, GstBuffer* buffer) {
	MagmaHipMeta* m = (MagmaHipMeta*)meta;
	if (m->release)
		m->release(m->user_data);
}

static gboolean magma_hip_meta_transform(GstBuffer* transbuf, GstMeta* meta, GstBuffer* buffer, GQuark type, gpointer data) {
	if (type != 0)
		return FALSE;
	MagmaHipMeta* src = (MagmaHipMeta*)meta;
	MagmaHipMeta* dest = magma_buffer_add_hip_meta(transbuf, src->d_ptr, src->release, src->user_data);
	if (dest && src->release)
		src->release = NULL; // ownership transferred
	return dest != NULL;
}

static gsize _magma_hip_meta_info = 0;

const GstMetaInfo* magma_hip_meta_get_info(void) {
	if (g_once_init_enter(&_magma_hip_meta_info)) {
		const GstMetaInfo* info = gst_meta_register(MAGMA_HIP_META_API_TYPE, "MagmaHipMeta", sizeof(MagmaHipMeta), magma_hip_meta_init, magma_hip_meta_free, magma_hip_meta_transform);
		if (!info)
			info = gst_meta_get_info("MagmaHipMeta");
		g_once_init_leave(&_magma_hip_meta_info, (gsize)info);
	}
	return (const GstMetaInfo*)_magma_hip_meta_info;
}

/**
 * @brief Attach a MagmaHipMeta with a HIP device pointer to a buffer.
 *
 * Used to pass GPU-resident NV12 frames between mgmh264dec and
 * mgmpreproc without DMABuf import/export.
 *
 * @param buffer    Target GstBuffer
 * @param d_ptr     HIP device pointer to the frame
 * @param release   Optional callback invoked when the meta is freed
 * @param user_data User data passed to the release callback
 * @return Pointer to the attached meta, or NULL on failure
 */
MagmaHipMeta* magma_buffer_add_hip_meta(GstBuffer* buffer, hipDeviceptr_t d_ptr, void (*release)(void*), void* user_data) {
	MagmaHipMeta* m = (MagmaHipMeta*)gst_buffer_add_meta(buffer, magma_hip_meta_get_info(), NULL);
	if (m) {
		m->d_ptr = d_ptr;
		m->release = release;
		m->user_data = user_data;
	}
	return m;
}

/* ─── MagmaSemanticMeta ─────────────────────────────────────────────── */

#include "magma-primitives.h"

static gsize _magma_semantic_meta_quark = 0;

GType magma_semantic_meta_api_get_type(void) {
	if (g_once_init_enter(&_magma_semantic_meta_quark)) {
		const gchar* tags[] = {NULL};
		GType t = gst_meta_api_type_register("MagmaSemanticMetaAPI", tags);
		g_once_init_leave(&_magma_semantic_meta_quark, (gsize)t);
	}
	return (GType)_magma_semantic_meta_quark;
}

static gboolean magma_semantic_meta_init(GstMeta* meta, gpointer params, GstBuffer* buffer) {
	MagmaSemanticMeta* m = (MagmaSemanticMeta*)meta;
	m->type_id = 0;
	m->data_gpu = NULL;
	m->source_width = 0;
	m->source_height = 0;
	m->roi_x = m->roi_y = m->roi_w = m->roi_h = 0;
	m->model_width = m->model_height = 0;
	m->masks_gpu = NULL;
	m->mask_count = 0;
	m->mask_width = 0;
	m->mask_height = 0;
	m->d_masks_gpu = 0;
	m->d_objects_gpu = 0;
	m->masks_gpu_bytes = 0;
	return TRUE;
}

static void magma_semantic_meta_free(GstMeta* meta, GstBuffer* buffer) {
	MagmaSemanticMeta* m = (MagmaSemanticMeta*)meta;
	if (m->data_gpu) {
		gst_memory_unref(m->data_gpu);
		m->data_gpu = NULL;
	}
	if (m->masks_gpu) {
		gst_memory_unref(m->masks_gpu);
		m->masks_gpu = NULL;
	}
}

static gboolean magma_semantic_meta_transform(GstBuffer* transbuf, GstMeta* meta, GstBuffer* buffer, GQuark type, gpointer data) {
	if (type != 0)
		return FALSE;
	MagmaSemanticMeta* src = (MagmaSemanticMeta*)meta;
	MagmaSemanticMeta* dest = magma_buffer_add_semantic_meta(transbuf, src->type_id, src->data_gpu ? gst_memory_ref(src->data_gpu) : NULL, src->source_width, src->source_height);
	if (!dest)
		return FALSE;
	dest->roi_x = src->roi_x;
	dest->roi_y = src->roi_y;
	dest->roi_w = src->roi_w;
	dest->roi_h = src->roi_h;
	dest->model_width = src->model_width;
	dest->model_height = src->model_height;
	dest->mask_count = src->mask_count;
	dest->mask_width = src->mask_width;
	dest->mask_height = src->mask_height;
	if (src->masks_gpu)
		dest->masks_gpu = gst_memory_ref(src->masks_gpu);
	dest->d_masks_gpu = src->d_masks_gpu;
	dest->d_objects_gpu = src->d_objects_gpu;
	dest->masks_gpu_bytes = src->masks_gpu_bytes;
	return TRUE;
}

static gsize _magma_semantic_meta_info = 0;

const GstMetaInfo* magma_semantic_meta_get_info(void) {
	if (g_once_init_enter(&_magma_semantic_meta_info)) {
		const GstMetaInfo* info =
		    gst_meta_register(MAGMA_SEMANTIC_META_API_TYPE, "MagmaSemanticMeta", sizeof(MagmaSemanticMeta), magma_semantic_meta_init, magma_semantic_meta_free, magma_semantic_meta_transform);
		if (!info)
			info = gst_meta_get_info("MagmaSemanticMeta");
		g_once_init_leave(&_magma_semantic_meta_info, (gsize)info);
	}
	return (const GstMetaInfo*)_magma_semantic_meta_info;
}

MagmaSemanticMeta* magma_buffer_add_semantic_meta(GstBuffer* buffer, GQuark type_id, GstMemory* data_gpu, gint source_width, gint source_height) {
	MagmaSemanticMeta* m = (MagmaSemanticMeta*)gst_buffer_add_meta(buffer, magma_semantic_meta_get_info(), NULL);
	if (m) {
		m->type_id = type_id;
		m->data_gpu = data_gpu ? gst_memory_ref(data_gpu) : NULL;
		m->source_width = source_width;
		m->source_height = source_height;
	}
	return m;
}

MagmaSemanticMeta* magma_buffer_add_semantic_meta_full(
    GstBuffer* buffer, GQuark type_id, GstMemory* data_gpu, gint source_width, gint source_height, GstMemory* masks_gpu, gint mask_count, gint mask_width, gint mask_height) {
	MagmaSemanticMeta* m = magma_buffer_add_semantic_meta(buffer, type_id, data_gpu, source_width, source_height);
	if (m) {
		m->masks_gpu = masks_gpu ? gst_memory_ref(masks_gpu) : NULL;
		m->mask_count = mask_count;
		m->mask_width = mask_width;
		m->mask_height = mask_height;
	}
	return m;
}

/* ─── Auto-register default converters ─────────────────────────────── */

__attribute__((constructor)) static void magma_meta_lib_init(void) {
	magma_primitives_auto_register();
}
