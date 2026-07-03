#pragma once

#include <gst/gst.h>
#include <gst/video/video.h>
#include <gst/base/gstbasetransform.h>
#include <gst/allocators/gstdmabuf.h>
#include <hip/hip_runtime.h>

#include "magma-infer-meta.h"
#include "kernel_utils.hpp"

G_BEGIN_DECLS

#define OSD_SINK_CAPS "video/x-raw(memory:DMABuf),format=(string)NV12"
#define OSD_SRC_CAPS "video/x-raw(memory:DMABuf),format=(string)NV12"

#define GST_TYPE_MAGMA_OSD (gst_magma_osd_get_type())

G_DECLARE_FINAL_TYPE(GstMagmaOsd, gst_magma_osd, GST, MAGMA_OSD, GstBaseTransform)

struct _GstMagmaOsd {
    GstBaseTransform parent;

    gint in_width;
    gint in_height;

    guint line_width;

    /* ROI parameters to map normalized bbox back to source coords */
    guint roi_x, roi_y, roi_w, roi_h;

    /* HIP kernel */
    hipStream_t hip_stream;
    hipModule_t kernel_module;
    hipFunction_t kernel_func;
    gboolean kernel_ready;

    /* Per-frame DMABuf import */
    hipExternalMemory_t external_memory;
    hipDeviceptr_t d_image;

    /* GPU objects buffer (uploaded per frame) */
    hipDeviceptr_t d_boxes;
};

G_END_DECLS
