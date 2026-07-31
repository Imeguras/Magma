#pragma once

#include <gst/gst.h>
#include <gst/video/video.h>
#include <gst/base/gstbasetransform.h>
#include <gst/allocators/gstdmabuf.h>
#include <hip/hip_runtime.h>

#include "magma-meta.h"
#include "magma-primitives.h"
#include "kernel_utils.hpp"

G_BEGIN_DECLS

#define GST_TYPE_MAGMA_OSD (gst_magma_osd_get_type())
G_DECLARE_FINAL_TYPE(GstMagmaOsd, gst_magma_osd, GST, MAGMA_OSD, GstBaseTransform)

/**
 * @brief Magma on-screen display element (model-agnostic).
 *
 * Reads MagmaSemanticMeta (and legacy MagmaInferenceMeta) from
 * buffers, converts to render primitives via the type_id-based
 * converter registry, and dispatches per-primitive-type HIP kernels.
 *
 * No model-specific header is included — all model knowledge comes
 * through the converter API.
 */

struct _GstMagmaOsd {
    GstBaseTransform parent;

    /* Video dimensions (set from caps) */
    gint in_width;
    gint in_height;

    /* Properties */
    guint line_width;
    guint max_primitives;       /* configurable ceiling (default 500) */
    gboolean show_labels;
    gboolean palette_by_track;  /* FALSE=class_id, TRUE=track_id */
    gchar* labels_file;         /* path to COCO-format labels file (or NULL) */

    /* ROI (model-space → source-space coordinate offset) */
    guint roi_x, roi_y, roi_w, roi_h;

    /* HIP stream (shared across pipeline) */
    hipStream_t hip_stream;

    /* ─── Kernel modules & function handles (one per primitive type) ─── */
    hipModule_t kernel_module;
    hipFunction_t kernel_funcs[7]; /* indexed by MagmaPrimitiveType */

    /* Device buffers for per-type primitive arrays */
    hipDeviceptr_t d_rects;
    hipDeviceptr_t d_polylines;
    hipDeviceptr_t d_polygons;
    hipDeviceptr_t d_points;
    hipDeviceptr_t d_texts;
    hipDeviceptr_t d_arrows;
    hipDeviceptr_t d_vertex_arena;  /* shared vertex data */
    int d_vertex_arena_bytes;

    /* Host-side primitive list (reused frame-to-frame) */
    MagmaPrimitiveList primitives;

    gboolean kernel_ready;

    /* ─── Frame pointer acquisition (reusable) ───────────────────────── */
    hipExternalMemory_t external_memory;  /* cached DMABuf import */
    hipDeviceptr_t d_image;
    hipDeviceptr_t d_input_upload;        /* system-memory upload fallback */
};

G_END_DECLS
