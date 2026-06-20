#include "mgminfer.hpp"

#include <gst/gst.h>

namespace magma::infer {

bool register_plugin (GstPlugin *plugin){
  GST_INFO ("Magma Inference (mgminfer) Plugin Loaded");
  return true;
}

} // namespace magma::infer

static gboolean plugin_init (GstPlugin *plugin){
  return magma::infer::register_plugin (plugin);
}

GST_PLUGIN_DEFINE (
    GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    mgminfer,
    "Magma Inference Plugin",
    plugin_init,
    "v0.1.0",
    "LGPLv3",
    "magma",
    "https://imeguras.eu.org"
)
