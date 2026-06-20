#include "mgmpreproc.hpp"

G_DEFINE_TYPE(
    GstMagmaPreproc,
    gst_magma_preproc,
    GST_TYPE_BASE_TRANSFORM
)


static gboolean plugin_init(GstPlugin *plugin){
    return gst_element_register(
        plugin,
        "mgmpreproc",
        GST_RANK_NONE,
        GST_TYPE_MAGMA_PREPROC
    );
}
static void gst_magma_preproc_class_init(GstMagmaPreprocClass *klass){
    GstElementClass *element_class = GST_ELEMENT_CLASS(klass);

    gst_element_class_set_static_metadata(
        element_class,
        "Magma Preprocessor",
        "Filter/Video",
        "ROCm/HIP tensor preprocessing",
        "Magma"
    );
}


static void gst_magma_preproc_init(GstMagmaPreproc *self){
    self->net_width = 224;
    self->net_height = 224;
    self->scale_factor = 1.0f / 255.0f;

    self->hip_stream = nullptr;
    self->d_tensor_output = nullptr;
}

GST_PLUGIN_DEFINE(
    GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    mgmpreproc,
    "Magma Preprocessing Plugin",
    plugin_init,
    "0.1.0",
    "LGPL",
    "magma",
    "https://imeguras.eu.org"
)