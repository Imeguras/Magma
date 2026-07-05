#pragma once

#include <gst/gst.h>
#include <hip/hip_runtime.h>

G_BEGIN_DECLS

typedef struct _MagmaTensorMeta MagmaTensorMeta;
typedef struct _MagmaHipMeta MagmaHipMeta;

struct _MagmaTensorMeta {
    GstMeta meta;

    GstMemory *tensor_mem;
    gint width;
    gint height;
    gint channels;

    /* ROI that was cropped from source before resize (0,0,src_w,src_h if none) */
    gint roi_x, roi_y, roi_w, roi_h;
};

GType magma_tensor_meta_api_get_type(void);
const GstMetaInfo *magma_tensor_meta_get_info(void);

MagmaTensorMeta *magma_buffer_add_tensor_meta(GstBuffer *buffer, GstMemory *tensor_mem,
                                               gint width, gint height, gint channels);

#define MAGMA_TENSOR_META_API_TYPE (magma_tensor_meta_api_get_type())
#define magma_buffer_get_tensor_meta(b) ((MagmaTensorMeta *)gst_buffer_get_meta((b), MAGMA_TENSOR_META_API_TYPE))

// ─── MagmaHipMeta — carries a HIP device pointer through GStreamer ───
// Used by mgmh264dec → mgmpreproc to avoid DMABuf import.
struct _MagmaHipMeta {
    GstMeta meta;
    hipDeviceptr_t d_ptr;
    void (*release)(void*);
    void* user_data;
};

GType magma_hip_meta_api_get_type(void);
const GstMetaInfo *magma_hip_meta_get_info(void);

MagmaHipMeta *magma_buffer_add_hip_meta(GstBuffer *buffer, hipDeviceptr_t d_ptr,
                                        void (*release)(void*), void* user_data);

#define MAGMA_HIP_META_API_TYPE (magma_hip_meta_api_get_type())
#define magma_buffer_get_hip_meta(b) ((MagmaHipMeta *)gst_buffer_get_meta((b), MAGMA_HIP_META_API_TYPE))

G_END_DECLS
