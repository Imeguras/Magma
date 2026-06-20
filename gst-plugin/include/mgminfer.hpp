#pragma once

#include <gst/gst.h>

namespace magma::infer {

bool register_plugin(GstPlugin *plugin);

} // namespace magma::infer
