// SPDX-FileCopyrightText: 2026 João Vieira <joaodavid2001@gmail.com>
// SPDX-License-Identifier: LGPL-3.0-or-later

#pragma once

/**
 * @file mgmserialize.hpp
 * @brief Serialization element — converts inference results to JSON or Protobuf.
 */

#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>

G_BEGIN_DECLS

#define GST_TYPE_MAGMA_SERIALIZE (gst_magma_serialize_get_type())
G_DECLARE_FINAL_TYPE(GstMagmaSerialize, gst_magma_serialize, GST, MAGMA_SERIALIZE, GstBaseTransform)

/**
 * @brief Magma serialization element.
 *
 * Reads MagmaInferenceMeta and produces a text buffer containing
 * JSON or Protobuf-serialized detection results.
 *
 * @property format            Output format: "json" (default) or "protobuf"
 * @property max-mask-pixels   Cap on how many segmentation-mask floats are
 *                             emitted per buffer (default 64). Raise it to dump
 *                             whole masks when debugging a parser; a full set of
 *                             N 160x160 instance masks needs N*25600.
 */
struct _GstMagmaSerialize {
	GstBaseTransform parent;
	gchar* format;
	guint max_mask_pixels;
};

G_END_DECLS
