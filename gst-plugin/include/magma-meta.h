#pragma once

/**
 * @file magma-meta.h
 * @brief GStreamer meta types — MagmaTensorMeta, MagmaHipMeta, MagmaSemanticMeta.
 *
 * These custom GstMeta types carry tensors, HIP device pointers, and
 * typed semantic inference output through GStreamer pipelines without
 * copying between GPU and CPU.
 */

#include <gst/gst.h>
#include <hip/hip_runtime.h>

G_BEGIN_DECLS

typedef struct _MagmaTensorMeta MagmaTensorMeta;
typedef struct _MagmaHipMeta MagmaHipMeta;
typedef struct _MagmaSemanticMeta MagmaSemanticMeta;

/**
 * @brief GStreamer metadata carrying a preprocessed tensor.
 *
 * Attached to output buffers of mgmpreproc. Carries the normalized
 * RGB tensor as a DMABuf-backed GstMemory, plus the ROI that was
 * cropped from the source frame.
 */
struct _MagmaTensorMeta {
	GstMeta meta;

	GstMemory* tensor_mem;           /** DMABuf-backed memory containing the float32 RGB tensor */
	gint width;                      /** Tensor width in pixels (e.g. 640) */
	gint height;                     /** Tensor height in pixels (e.g. 640) */
	gint channels;                   /** Number of channels (typically 3 for RGB) */
	gint roi_x, roi_y, roi_w, roi_h; /** ROI crop rectangle within the source frame */
};

/** @brief Query the GType for MagmaTensorMeta */
GType magma_tensor_meta_api_get_type(void);

/** @brief Get the GstMetaInfo for MagmaTensorMeta */
const GstMetaInfo* magma_tensor_meta_get_info(void);

/**
 * @brief Attach a tensor to a GstBuffer as MagmaTensorMeta.
 * @param buffer    Target GstBuffer
 * @param tensor_mem DMABuf-backed GstMemory with float32 RGB data
 * @param width     Tensor width
 * @param height    Tensor height
 * @param channels  Number of channels
 * @return Pointer to the newly-attached MagmaTensorMeta, or NULL on failure
 */
MagmaTensorMeta* magma_buffer_add_tensor_meta(GstBuffer* buffer, GstMemory* tensor_mem, gint width, gint height, gint channels);

#define MAGMA_TENSOR_META_API_TYPE (magma_tensor_meta_api_get_type())
#define magma_buffer_get_tensor_meta(b) ((MagmaTensorMeta*)gst_buffer_get_meta((b), MAGMA_TENSOR_META_API_TYPE))

/**
 * @brief GStreamer metadata carrying a HIP device pointer.
 *
 * Used by mgmh264dec -> mgmpreproc to pass GPU-resident NV12 frames
 * without importing/exporting a DMABuf. The pointer is valid on the
 * current HIP device.
 */
struct _MagmaHipMeta {
	GstMeta meta;

	/** HIP device pointer to the NV12 frame data */
	hipDeviceptr_t d_ptr;

	/** Optional release callback (called when the meta is freed) */
	void (*release)(void*);

	/** User data passed to release callback */
	void* user_data;
};

/** @brief Query the GType for MagmaHipMeta */
GType magma_hip_meta_api_get_type(void);

/** @brief Get the GstMetaInfo for MagmaHipMeta */
const GstMetaInfo* magma_hip_meta_get_info(void);

/**
 * @brief Attach a HIP device pointer to a GstBuffer as MagmaHipMeta.
 * @param buffer    Target GstBuffer
 * @param d_ptr     HIP device pointer
 * @param release   Optional cleanup callback (called on meta free)
 * @param user_data User data passed to release callback
 * @return Pointer to the newly-attached MagmaHipMeta, or NULL on failure
 */
MagmaHipMeta* magma_buffer_add_hip_meta(GstBuffer* buffer, hipDeviceptr_t d_ptr, void (*release)(void*), void* user_data);

#define MAGMA_HIP_META_API_TYPE (magma_hip_meta_api_get_type())
#define magma_buffer_get_hip_meta(b) ((MagmaHipMeta*)gst_buffer_get_meta((b), MAGMA_HIP_META_API_TYPE))

/**
 * @brief Generic semantic meta — carries typed model output.
 *
 * Attached to buffers by mgminfer after magma_parse completes.
 * The type_id (GQuark) identifies the model kind (e.g. "magma.detection.v1",
 * "magma.pose.v1") and selects the converter that turns this data into
 * render primitives.
 *
 * @note The type_id string must be kept in sync with the convention
 *       used by magma_register_to_primitives().
 */
struct _MagmaSemanticMeta {
	GstMeta meta;

	/** Type identifier (GQuark of "magma.<kind>.v<version>") */
	GQuark type_id;

	/** GPU memory blob — interpreted by the converter registered for type_id.
	 *  For detection: MagmaInferObjectGPU[] (same layout as
	 *  MagmaInferenceMeta.objects_gpu). For other types: parser-defined. */
	GstMemory* data_gpu;

	/** Source-frame dimensions at inference time (for coord remapping debug) */
	gint source_width;
	gint source_height;

	/** Optional: ROI within source frame (0,0,0,0 = full frame) */
	gint roi_x, roi_y, roi_w, roi_h;

	/** Optional: model input dimensions (for denormalisation if needed) */
	gint model_width, model_height;

	/** Optional: instance segmentation masks (CPU-copy buffer, matched 1:1 with data_gpu objects) */
	GstMemory* masks_gpu;
	gint mask_count;
	gint mask_width;
	gint mask_height;

	/** GPU-resident mask + object data for direct GPU decode (0 = not set) */
	hipDeviceptr_t d_masks_gpu;   /** compacted mask tensor on GPU */
	hipDeviceptr_t d_objects_gpu; /** compacted detection data on GPU (6 floats per object) */
	gint masks_gpu_bytes;
};

/** @brief Query the GType for MagmaSemanticMeta */
GType magma_semantic_meta_api_get_type(void);

/** @brief Get the GstMetaInfo for MagmaSemanticMeta */
const GstMetaInfo* magma_semantic_meta_get_info(void);

/**
 * @brief Attach a MagmaSemanticMeta to a GstBuffer.
 * @param buffer        Target GstBuffer
 * @param type_id       GQuark for the semantic type (e.g. "magma.detection.v1")
 * @param data_gpu      GPU memory with model output data
 * @param source_width  Source frame width
 * @param source_height Source frame height
 * @return Pointer to the newly-attached meta, or NULL on failure
 */
MagmaSemanticMeta* magma_buffer_add_semantic_meta(GstBuffer* buffer, GQuark type_id, GstMemory* data_gpu, gint source_width, gint source_height);

MagmaSemanticMeta* magma_buffer_add_semantic_meta_full(
    GstBuffer* buffer, GQuark type_id, GstMemory* data_gpu, gint source_width, gint source_height, GstMemory* masks_gpu, gint mask_count, gint mask_width, gint mask_height);

#define MAGMA_SEMANTIC_META_API_TYPE (magma_semantic_meta_api_get_type())
#define magma_buffer_get_semantic_meta(b) ((MagmaSemanticMeta*)gst_buffer_get_meta((b), MAGMA_SEMANTIC_META_API_TYPE))

G_END_DECLS
