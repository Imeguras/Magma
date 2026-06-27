#include "mgminfer.hpp"

#include <cstring>

/** --- GOBJECT / GSTREAMER STUFF --- */
GST_DEBUG_CATEGORY_STATIC(magma_infer_debug);
#define GST_CAT_DEFAULT magma_infer_debug

enum
{
    PROP_0,
    PROP_MODEL_PATH,
    PROP_INFERENCE_INTERVAL,
};

G_DEFINE_TYPE(
    GstMagmaInfer,
    gst_magma_infer,
    GST_TYPE_BASE_TRANSFORM
)

/** --- INFERENCE-OBJECT LIFECYCLE (CPU convenience) --- */
MagmaInferObject *
magma_infer_object_new (guint class_id, const gchar *label,
                         gfloat confidence,
                         gfloat x, gfloat y, gfloat w, gfloat h)
{
    MagmaInferObject *obj = g_slice_new0 (MagmaInferObject);
    obj->class_id   = class_id;
    obj->label      = g_strdup (label);
    obj->confidence = confidence;
    obj->x          = x;
    obj->y          = y;
    obj->width      = w;
    obj->height     = h;
    return obj;
}

void
magma_infer_object_free (MagmaInferObject *obj)
{
    if (!obj) return;
    g_free (obj->label);
    g_slice_free (MagmaInferObject, obj);
}

/** --- CUSTOM METADATA IMPLEMENTATION --- */

/* metadata API type  ------------------------------------------------------ */
static gsize _magma_inference_meta_quark = 0;

GType
magma_inference_meta_api_get_type (void)
{
    if (g_once_init_enter (&_magma_inference_meta_quark)) {
        GQuark q = g_quark_from_static_string ("MagmaInferenceMetaAPI");
        g_once_init_leave (&_magma_inference_meta_quark, q);
    }
    return (GType)_magma_inference_meta_quark;
}

/* metadata callbacks  ----------------------------------------------------- */

static gboolean
magma_inference_meta_init (GstMeta *meta, gpointer params, GstBuffer *buffer)
{
    MagmaInferenceMeta *m = (MagmaInferenceMeta *)meta;
    m->source_width     = 0;
    m->source_height    = 0;
    m->num_objects      = 0;
    m->objects_gpu      = NULL;
    m->output_tensors   = NULL;
    return TRUE;
}

static void
magma_inference_meta_free (GstMeta *meta, GstBuffer *buffer)
{
    MagmaInferenceMeta *m = (MagmaInferenceMeta *)meta;

    if (m->objects_gpu) {
        gst_memory_unref (m->objects_gpu);
        m->objects_gpu = NULL;
    }
    if (m->output_tensors) {
        for (guint i = 0; i < m->output_tensors->len; i++) {
            GstMemory *mem = (GstMemory *)g_ptr_array_index (m->output_tensors, i);
            gst_memory_unref (mem);
        }
        g_ptr_array_unref (m->output_tensors);
        m->output_tensors = NULL;
    }
}

static gboolean
magma_inference_meta_transform (GstBuffer *transbuf, GstMeta *meta,
                                 GstBuffer *buffer, GQuark type, gpointer data)
{
    MagmaInferenceMeta *src  = (MagmaInferenceMeta *)meta;
    MagmaInferenceMeta *dest;

    /* type == 0  means GST_META_TRANSFORM_IS_COPY */
    if (type != 0)
        return FALSE;

    dest = magma_buffer_add_inference_meta (transbuf,
                                             src->source_width,
                                             src->source_height);
    if (!dest)
        return FALSE;

    dest->num_objects = src->num_objects;

    /* share the GPU objects memory (ref it, don't deep-copy) */
    if (src->objects_gpu) {
        dest->objects_gpu = gst_memory_ref (src->objects_gpu);
    }

    /* share output tensors */
    if (src->output_tensors) {
        dest->output_tensors =
            g_ptr_array_new_with_free_func ((GDestroyNotify) gst_memory_unref);
        for (guint i = 0; i < src->output_tensors->len; i++) {
            GstMemory *mem = (GstMemory *)g_ptr_array_index (src->output_tensors, i);
            g_ptr_array_add (dest->output_tensors, gst_memory_ref (mem));
        }
    }

    return TRUE;
}

/* meta info singleton  ---------------------------------------------------- */
static gsize _magma_inference_meta_info = 0;

const GstMetaInfo *
magma_inference_meta_get_info (void)
{
    if (g_once_init_enter (&_magma_inference_meta_info)) {
        const GstMetaInfo *info = gst_meta_register (
            MAGMA_INFERENCE_META_API_TYPE,
            "MagmaInferenceMeta",
            sizeof (MagmaInferenceMeta),
            magma_inference_meta_init,
            magma_inference_meta_free,
            magma_inference_meta_transform);
        g_once_init_leave (&_magma_inference_meta_info, (gsize)info);
    }
    return (const GstMetaInfo *)_magma_inference_meta_info;
}

/* public metadata helpers  ------------------------------------------------ */

MagmaInferenceMeta *
magma_buffer_add_inference_meta (GstBuffer *buffer,
                                  guint source_width, guint source_height)
{
    MagmaInferenceMeta *m = (MagmaInferenceMeta *)gst_buffer_add_meta (
        buffer, magma_inference_meta_get_info (), NULL);
    if (m) {
        m->source_width  = source_width;
        m->source_height = source_height;
    }
    return m;
}

/** --- PAD TEMPLATES --- */
static GstStaticPadTemplate sink_template = GST_STATIC_PAD_TEMPLATE(
    "sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS("video/x-raw(memory:DMABuf),format=(string)NV12")
);

static GstStaticPadTemplate src_template = GST_STATIC_PAD_TEMPLATE(
    "src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS("video/x-raw(memory:DMABuf),format=(string)NV12")
);

/** --- PROPERTIES --- */
static void
gst_magma_infer_set_property (GObject *object, guint prop_id,
                               const GValue *value, GParamSpec *pspec)
{
    GstMagmaInfer *self = GST_MAGMA_INFER (object);

    switch (prop_id) {
        case PROP_MODEL_PATH:
            g_free (self->model_path);
            self->model_path = g_value_dup_string (value);
            GST_INFO_OBJECT (self, "model path set to %s", self->model_path);
            break;
        case PROP_INFERENCE_INTERVAL:
            self->inference_interval = g_value_get_uint (value);
            GST_INFO_OBJECT (self, "inference interval set to %u",
                             self->inference_interval);
            break;
        default:
            G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
            break;
    }
}

static void
gst_magma_infer_get_property (GObject *object, guint prop_id,
                               GValue *value, GParamSpec *pspec)
{
    GstMagmaInfer *self = GST_MAGMA_INFER (object);

    switch (prop_id) {
        case PROP_MODEL_PATH:
            g_value_set_string (value, self->model_path);
            break;
        case PROP_INFERENCE_INTERVAL:
            g_value_set_uint (value, self->inference_interval);
            break;
        default:
            G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
            break;
    }
}

/** --- FINALIZE --- */
static void
gst_magma_infer_finalize (GObject *object)
{
    GstMagmaInfer *self = GST_MAGMA_INFER (object);

    g_free (self->model_path);
    self->model_path = NULL;

    G_OBJECT_CLASS (gst_magma_infer_parent_class)->finalize (object);
}

/** --- INIT --- */
static void
gst_magma_infer_init (GstMagmaInfer *self)
{
    self->model_path          = NULL;
    self->inference_interval  = 1;
    self->frame_counter       = 0;
    self->in_width            = 0;
    self->in_height           = 0;

    gst_base_transform_set_in_place (GST_BASE_TRANSFORM (self), TRUE);
}

/** --- CAPS NEGOTIATION --- */
static gboolean
gst_magma_infer_set_caps (GstBaseTransform *trans,
                           GstCaps *incaps,
                           GstCaps *outcaps)
{
    GstMagmaInfer *self = GST_MAGMA_INFER (trans);
    GstVideoInfo info;

    if (!gst_video_info_from_caps (&info, incaps)) {
        GST_ERROR_OBJECT (self, "failed to parse incaps");
        return FALSE;
    }

    self->in_width  = GST_VIDEO_INFO_WIDTH (&info);
    self->in_height = GST_VIDEO_INFO_HEIGHT (&info);

    GST_INFO_OBJECT (self, "configured %dx%d %s",
                     self->in_width, self->in_height,
                     GST_VIDEO_INFO_NAME (&info));
    return TRUE;
}

/** --- TRANSFORM (per-frame): attach GPU inference metadata --- */
static GstFlowReturn
gst_magma_infer_transform_ip (GstBaseTransform *trans, GstBuffer *buf)
{
    GstMagmaInfer *self = GST_MAGMA_INFER (trans);

    GstMemory *mem = gst_buffer_peek_memory (buf, 0);
    if (!gst_is_dmabuf_memory (mem)) {
        GST_ERROR_OBJECT (self, "mgminfer requires DMABuf memory");
        return GST_FLOW_ERROR;
    }

    /* ----------------------------------------------------------------
     *  S K E L E T O N
     *
     *  Real inference will:
     *    1. import the DMABuf into HIP (same pattern as mgmpreproc)
     *    2. run the model
     *    3. populate objects_gpu / output_tensors with DMABuf memories
     *       containing the GPU-resident detection arrays
     *
     *  For now we attach an empty meta so downstream can already
     *  exercise the metadata API.
     * ---------------------------------------------------------------- */

    if (!magma_buffer_add_inference_meta (buf,
                                           self->in_width,
                                           self->in_height))
    {
        GST_WARNING_OBJECT (self, "failed to attach inference metadata");
    }

    self->frame_counter++;
    GST_LOG_OBJECT (self, "frame %u — metadata attached", self->frame_counter);

    return GST_FLOW_OK;
}

/** --- CLASS INIT --- */
static void
gst_magma_infer_class_init (GstMagmaInferClass *klass)
{
    GObjectClass *gobject_class    = G_OBJECT_CLASS (klass);
    GstElementClass *element_class = GST_ELEMENT_CLASS (klass);
    GstBaseTransformClass *trans   = GST_BASE_TRANSFORM_CLASS (klass);

    gobject_class->set_property = gst_magma_infer_set_property;
    gobject_class->get_property = gst_magma_infer_get_property;
    gobject_class->finalize     = gst_magma_infer_finalize;

    /* --- properties --- */
    g_object_class_install_property (
        gobject_class, PROP_MODEL_PATH,
        g_param_spec_string (
            "model-path", "Model path",
            "Path to the inference model file",
            NULL, G_PARAM_READWRITE));

    g_object_class_install_property (
        gobject_class, PROP_INFERENCE_INTERVAL,
        g_param_spec_uint (
            "inference-interval", "Inference interval",
            "Run inference every N frames (1 = every frame)",
            1, G_MAXUINT32, 1, G_PARAM_READWRITE));

    /* --- pad templates --- */
    gst_element_class_add_static_pad_template (element_class, &sink_template);
    gst_element_class_add_static_pad_template (element_class, &src_template);

    gst_element_class_set_static_metadata (
        element_class,
        "Magma Inference",
        "Meta/Inference/Video",
        "Attaches GPU inference results as buffer metadata (skeleton)",
        "Magma");

    /* --- transform vfuncs --- */
    trans->set_caps   = gst_magma_infer_set_caps;
    trans->transform_ip = gst_magma_infer_transform_ip;

    /* ensure custom meta is registered exactly once */
    magma_inference_meta_get_info ();

    GST_DEBUG_CATEGORY_INIT (magma_infer_debug, "magma_infer", 0,
                              "Magma Inference Plugin");
}

/** --- PLUGIN REGISTRATION --- */
static gboolean
plugin_init (GstPlugin *plugin)
{
    return gst_element_register (
        plugin, "mgminfer", GST_RANK_NONE, GST_TYPE_MAGMA_INFER);
}

GST_PLUGIN_DEFINE (
    GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    mgminfer,
    "Magma Inference Plugin",
    plugin_init,
    "0.1.0",
    "LGPL",
    "magma",
    "https://imeguras.eu.org"
)
