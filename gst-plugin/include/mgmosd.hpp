#pragma once

/**
 * @file mgmosd.hpp
 * @brief On-screen display element — draws detection boxes on video frames.
 */

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

/**
 * @brief Magma on-screen display element.
 *
 * Reads MagmaInferenceMeta from input buffers and draws bounding
 * boxes, class labels, and confidence scores directly onto the
 * NV12 frame using a GPU HIP kernel.
 *
 * @property line-width  Width of bounding box lines in pixels
 */
struct _GstMagmaOsd {
    GstBaseTransform parent;

    gint in_width;
    gint in_height;

    guint line_width;

    guint roi_x, roi_y, roi_w, roi_h;

    hipStream_t hip_stream;
    hipModule_t kernel_module;
    hipFunction_t kernel_func;
    gboolean kernel_ready;

    hipExternalMemory_t external_memory;
    hipDeviceptr_t d_image;

    hipDeviceptr_t d_boxes;

    gboolean show_labels;

    hipDeviceptr_t d_input_upload;
};

G_END_DECLS
