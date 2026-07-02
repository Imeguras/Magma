#include "mgmtensordump.hpp"

#include <cstdio>
#include <cstring>
#include <cstdint>

/** --- GOBJECT / GSTREAMER STUFF --- */
GST_DEBUG_CATEGORY_STATIC(magma_tensor_dump_debug);
#define GST_CAT_DEFAULT magma_tensor_dump_debug

enum {
    PROP_0,
    PROP_NET_WIDTH,
    PROP_NET_HEIGHT,
    PROP_DUMP_LOCATION,
    PROP_FRAME_SKIP,
};

G_DEFINE_TYPE(GstMagmaTensorDump, gst_magma_tensor_dump, GST_TYPE_BASE_TRANSFORM)

/** --- DUMP HEADER --- */
struct __attribute__((packed)) DumpHeader {
    char magic[4];
    uint64_t frame_num;
    uint64_t timestamp_ns;
    uint32_t width;
    uint32_t height;
    uint32_t channels;
    uint64_t data_size;
};
static_assert(sizeof(DumpHeader) == 40, "DumpHeader size mismatch");

static void compute_panel_size(int net_w, int net_h, int* pw, int* ph) {
    *pw = net_w;
    *ph = net_h;
    int max_dim = 640;
    if (*pw > max_dim || *ph > max_dim) {
        float scale = (float)max_dim / (*pw > *ph ? *pw : *ph);
        *pw = (int)(*pw * scale + 0.5f);
        *ph = (int)(*ph * scale + 0.5f);
        if (*pw < 1)
            *pw = 1;
        if (*ph < 1)
            *ph = 1;
    }
}

/** --- PROPERTIES --- */
static void gst_magma_tensor_dump_set_property(GObject* object, guint prop_id, const GValue* value, GParamSpec* pspec) {
    GstMagmaTensorDump* self = GST_MAGMA_TENSOR_DUMP(object);
    switch (prop_id) {
    case PROP_NET_WIDTH:
        self->net_width = g_value_get_int(value);
        compute_panel_size(self->net_width, self->net_height, &self->panel_w, &self->panel_h);
        break;
    case PROP_NET_HEIGHT:
        self->net_height = g_value_get_int(value);
        compute_panel_size(self->net_width, self->net_height, &self->panel_w, &self->panel_h);
        break;
    case PROP_DUMP_LOCATION:
        g_free(self->dump_location);
        self->dump_location = g_value_dup_string(value);
        break;
    case PROP_FRAME_SKIP:
        self->frame_skip = g_value_get_int(value);
        break;
    default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
        break;
    }
}

static void gst_magma_tensor_dump_get_property(GObject* object, guint prop_id, GValue* value, GParamSpec* pspec) {
    GstMagmaTensorDump* self = GST_MAGMA_TENSOR_DUMP(object);
    switch (prop_id) {
    case PROP_NET_WIDTH:
        g_value_set_int(value, self->net_width);
        break;
    case PROP_NET_HEIGHT:
        g_value_set_int(value, self->net_height);
        break;
    case PROP_DUMP_LOCATION:
        g_value_set_string(value, self->dump_location);
        break;
    case PROP_FRAME_SKIP:
        g_value_set_int(value, self->frame_skip);
        break;
    default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
        break;
    }
}

/** --- FINALIZE --- */
static void gst_magma_tensor_dump_finalize(GObject* object) {
    GstMagmaTensorDump* self = GST_MAGMA_TENSOR_DUMP(object);

    if (self->dump_file) {
        fclose(self->dump_file);
        self->dump_file = nullptr;
    }

    if (self->tensor_ext_mem) {
        (void)hipDestroyExternalMemory(self->tensor_ext_mem);
        self->tensor_ext_mem = nullptr;
        self->d_tensor = 0;
    }

    if (self->host_tensor) {
        free(self->host_tensor);
        self->host_tensor = nullptr;
    }

    if (self->hip_stream) {
        (void)hipStreamDestroy(self->hip_stream);
        self->hip_stream = nullptr;
    }

    g_free(self->dump_location);
    self->dump_location = nullptr;

    G_OBJECT_CLASS(gst_magma_tensor_dump_parent_class)->finalize(object);
}

/** --- PAD TEMPLATES --- */
static GstStaticPadTemplate sink_template = GST_STATIC_PAD_TEMPLATE("sink", GST_PAD_SINK, GST_PAD_ALWAYS, GST_STATIC_CAPS("video/x-raw(memory:DMABuf),format=(string)NV12"));

static GstStaticPadTemplate src_template = GST_STATIC_PAD_TEMPLATE("src", GST_PAD_SRC, GST_PAD_ALWAYS, GST_STATIC_CAPS("video/x-raw,format=(string)RGB"));

/** --- INIT --- */
static void gst_magma_tensor_dump_init(GstMagmaTensorDump* self) {
    self->net_width = 224;
    self->net_height = 224;
    self->dump_location = nullptr;
    self->frame_skip = 0;

    self->in_width = 0;
    self->in_height = 0;

    self->hip_stream = nullptr;
    self->tensor_ext_mem = nullptr;
    self->d_tensor = 0;
    self->tensor_imported = FALSE;

    self->host_tensor = nullptr;
    self->tensor_bytes = 0;

    compute_panel_size(self->net_width, self->net_height, &self->panel_w, &self->panel_h);

    self->dump_file = nullptr;
    self->frame_num = 0;
}

/** --- CAPS NEGOTIATION --- */

/**
 * Returns the fixed src pad width (net_w * 3) for caps negotiation.
 * Required when downstream queries src pad caps.
 */
static GstCaps* gst_magma_tensor_dump_transform_caps(GstBaseTransform* trans, GstPadDirection direction, GstCaps* caps, GstCaps* filter) {
    GstMagmaTensorDump* self = GST_MAGMA_TENSOR_DUMP(trans);

    if (direction == GST_PAD_SINK) {
        int pw, ph;
        compute_panel_size(self->net_width, self->net_height, &pw, &ph);
        GstCaps* src = gst_caps_new_simple("video/x-raw", "format", G_TYPE_STRING, "RGB", "width", G_TYPE_INT, pw * 4, "height", G_TYPE_INT, ph, NULL);
        if (filter) {
            GstCaps* tmp = gst_caps_intersect_full(src, filter, GST_CAPS_INTERSECT_FIRST);
            gst_caps_unref(src);
            src = tmp;
        }
        return src;
    }

    // direction == GST_PAD_SRC: given src caps, return allowed sink caps
    (void)caps;
    GstCaps* sink = gst_static_pad_template_get_caps(&sink_template);
    if (filter) {
        GstCaps* tmp = gst_caps_intersect_full(sink, filter, GST_CAPS_INTERSECT_FIRST);
        gst_caps_unref(sink);
        sink = tmp;
    }
    return sink;
}

static gboolean gst_magma_tensor_dump_set_caps(GstBaseTransform* trans, GstCaps* incaps, GstCaps* outcaps) {
    GstMagmaTensorDump* self = GST_MAGMA_TENSOR_DUMP(trans);
    GstStructure* s = gst_caps_get_structure(incaps, 0);

    gst_structure_get_int(s, "width", &self->in_width);
    gst_structure_get_int(s, "height", &self->in_height);

    compute_panel_size(self->net_width, self->net_height, &self->panel_w, &self->panel_h);

    GST_INFO_OBJECT(self,
                    "Input %dx%d NV12 → RGB preview %dx%d (tensor %dx%d panels %dpx each)",
                    self->in_width,
                    self->in_height,
                    self->panel_w * 4,
                    self->panel_h,
                    self->net_width,
                    self->net_height,
                    self->panel_w);

    // Allocate host tensor buffer
    gsize n = (gsize)self->net_width * self->net_height * 3;
    self->tensor_bytes = n * sizeof(float);
    self->host_tensor = (float*)realloc(self->host_tensor, self->tensor_bytes);
    if (!self->host_tensor) {
        GST_ERROR_OBJECT(self, "Failed to allocate host tensor buffer (%zu bytes)", self->tensor_bytes);
        return FALSE;
    }

    // Open dump file if requested
    if (self->dump_location && self->dump_location[0]) {
        self->dump_file = fopen(self->dump_location, "wb");
        if (!self->dump_file)
            GST_WARNING_OBJECT(self, "Failed to open dump file %s", self->dump_location);
        else
            GST_INFO_OBJECT(self, "Dumping tensor to %s", self->dump_location);
    }

    return TRUE;
}

static gboolean gst_magma_tensor_dump_transform_size(GstBaseTransform* trans, GstPadDirection direction, GstCaps* caps, gsize size, GstCaps* othercaps, gsize* othersize) {
    GstMagmaTensorDump* self = GST_MAGMA_TENSOR_DUMP(trans);
    if (direction == GST_PAD_SRC) {
        // othercaps is the sink pad caps — return input size unchanged
        *othersize = size;
        return TRUE;
    }
    // direction == GST_PAD_SINK: othercaps is src pad caps, compute RGB size
    *othersize = (gsize)self->panel_w * 4 * self->panel_h * 3;
    return TRUE;
}

/** --- TRANSFORM (per-frame): read tensor meta → dump + RGB preview --- */
static GstFlowReturn gst_magma_tensor_dump_transform(GstBaseTransform* trans, GstBuffer* inbuf, GstBuffer* outbuf) {
    GstMagmaTensorDump* self = GST_MAGMA_TENSOR_DUMP(trans);

    magma_tensor_meta_get_info();

    // 1. Extract tensor meta from input buffer
    MagmaTensorMeta* meta = magma_buffer_get_tensor_meta(inbuf);
    if (!meta || !meta->tensor_mem) {
        GST_WARNING_OBJECT(self, "No tensor meta on incoming buffer — forwarding empty preview");
        GstMapInfo emap;
        gsize preview_size = (gsize)self->panel_w * 4 * self->panel_h * 3;
        GstMemory* empty = gst_allocator_alloc(nullptr, preview_size, nullptr);
        if (gst_memory_map(empty, &emap, GST_MAP_WRITE)) {
            memset(emap.data, 0, preview_size);
            gst_memory_unmap(empty, &emap);
        }
        gst_buffer_remove_all_memory(outbuf);
        gst_buffer_append_memory(outbuf, empty);
        return GST_FLOW_OK;
    }

    // Ensure HIP stream
    if (!self->hip_stream) {
        hipError_t herr = hipStreamCreate(&self->hip_stream);
        if (herr != hipSuccess) {
            GST_ERROR_OBJECT(self, "hipStreamCreate failed: %s", hipGetErrorString(herr));
            return GST_FLOW_ERROR;
        }
    }

    // 2. Import tensor DMABuf on first frame only (same buffer reused by mgmpreproc)
    if (!self->tensor_imported) {
        gint tfd = gst_dmabuf_memory_get_fd(meta->tensor_mem);
        gsize tsize = (gsize)meta->width * meta->height * meta->channels * sizeof(float);

        hipExternalMemoryHandleDesc desc{};
        desc.type = hipExternalMemoryHandleTypeOpaqueFd;
        desc.handle.fd = tfd;
        desc.size = tsize;

        hipError_t herr = hipImportExternalMemory(&self->tensor_ext_mem, &desc);
        if (herr != hipSuccess) {
            GST_ERROR_OBJECT(self, "hipImportExternalMemory(tensor) failed: %s", hipGetErrorString(herr));
            return GST_FLOW_ERROR;
        }

        hipExternalMemoryBufferDesc bdesc{};
        bdesc.offset = 0;
        bdesc.size = tsize;

        herr = hipExternalMemoryGetMappedBuffer(&self->d_tensor, self->tensor_ext_mem, &bdesc);
        if (herr != hipSuccess) {
            GST_ERROR_OBJECT(self, "hipExternalMemoryGetMappedBuffer(tensor) failed: %s", hipGetErrorString(herr));
            (void)hipDestroyExternalMemory(self->tensor_ext_mem);
            self->tensor_ext_mem = nullptr;
            return GST_FLOW_ERROR;
        }

        self->tensor_imported = TRUE;
        GST_INFO_OBJECT(self, "Tensor DMABuf imported (fd=%d, %zux%zux%d float)", tfd, (gsize)meta->width, (gsize)meta->height, meta->channels);
    }

    // 3. Copy tensor from GPU to host
    hipError_t herr = hipMemcpyDtoH(self->host_tensor, self->d_tensor, self->tensor_bytes);
    if (herr != hipSuccess) {
        GST_ERROR_OBJECT(self, "hipMemcpyDtoH failed: %s", hipGetErrorString(herr));
        return GST_FLOW_ERROR;
    }

    herr = hipStreamSynchronize(self->hip_stream);
    if (herr != hipSuccess) {
        GST_ERROR_OBJECT(self, "hipStreamSynchronize failed: %s", hipGetErrorString(herr));
        return GST_FLOW_ERROR;
    }

    gint w = meta->width;
    gint h = meta->height;
    gint c = meta->channels;
    float* data = self->host_tensor;

    // 4. Write binary dump (one header + raw floats per frame)
    if (self->dump_file && (self->frame_num % (self->frame_skip + 1) == 0)) {
        GstClockTime pts = GST_BUFFER_PTS(inbuf);

        DumpHeader dh;
        memcpy(dh.magic, "MTD1", 4);
        dh.frame_num = self->frame_num;
        dh.timestamp_ns = GST_CLOCK_TIME_IS_VALID(pts) ? pts : 0;
        dh.width = (uint32_t)w;
        dh.height = (uint32_t)h;
        dh.channels = (uint32_t)c;
        dh.data_size = self->tensor_bytes;

        fwrite(&dh, sizeof(dh), 1, self->dump_file);
        fwrite(data, 1, self->tensor_bytes, self->dump_file);
        fflush(self->dump_file);
    }

    // 5. Build RGB24 preview: 4 panels side-by-side (RGB | R | G | B)
    // Panels are downsampled to self->panel_w x self->panel_h via nearest neighbor
    int pw = self->panel_w;
    int ph = self->panel_h;
    gsize preview_size = (gsize)pw * 4 * ph * 3;
    GstMemory* outmem = gst_allocator_alloc(nullptr, preview_size, nullptr);
    GstMapInfo out_map;
    gst_memory_map(outmem, &out_map, GST_MAP_WRITE);

    guint8* preview = out_map.data;
    guint row_stride = (guint)(pw * 4 * 3);

    for (gint py = 0; py < ph; py++) {
        int src_y = py * h / ph;
        for (gint px = 0; px < pw; px++) {
            int src_x = px * w / pw;

            int r = (int)(data[src_y * w + src_x] * 255.0f + 0.5f);
            int g = (int)(data[w * h + src_y * w + src_x] * 255.0f + 0.5f);
            int b = (int)(data[2 * w * h + src_y * w + src_x] * 255.0f + 0.5f);
            if (r < 0)
                r = 0;
            if (r > 255)
                r = 255;
            if (g < 0)
                g = 0;
            if (g > 255)
                g = 255;
            if (b < 0)
                b = 0;
            if (b > 255)
                b = 255;

            guint8* prgb = preview + py * row_stride + px * 3;
            prgb[0] = (guint8)r;
            prgb[1] = (guint8)g;
            prgb[2] = (guint8)b;

            guint8* pr = prgb + pw * 3;
            pr[0] = (guint8)r;
            pr[1] = 0;
            pr[2] = 0;

            guint8* pg = pr + pw * 3;
            pg[0] = 0;
            pg[1] = (guint8)g;
            pg[2] = 0;

            guint8* pb = pg + pw * 3;
            pb[0] = 0;
            pb[1] = 0;
            pb[2] = (guint8)b;
        }
    }

    gst_memory_unmap(outmem, &out_map);
    gst_buffer_remove_all_memory(outbuf);
    gst_buffer_append_memory(outbuf, outmem);

    self->frame_num++;
    return GST_FLOW_OK;
}

/** --- PLUGIN REGISTRATION --- */
static gboolean plugin_init(GstPlugin* plugin) {
    GST_DEBUG_CATEGORY_INIT(magma_tensor_dump_debug, "magma_tensor_dump", 0, "Magma Tensor Dump");
    return gst_element_register(plugin, "mgmtensordump", GST_RANK_NONE, GST_TYPE_MAGMA_TENSOR_DUMP);
}

static void gst_magma_tensor_dump_class_init(GstMagmaTensorDumpClass* klass) {
    GstElementClass* element_class = GST_ELEMENT_CLASS(klass);
    GObjectClass* object_class = G_OBJECT_CLASS(klass);
    GstBaseTransformClass* trans_class = GST_BASE_TRANSFORM_CLASS(klass);

    object_class->set_property = gst_magma_tensor_dump_set_property;
    object_class->get_property = gst_magma_tensor_dump_get_property;
    object_class->finalize = gst_magma_tensor_dump_finalize;

    g_object_class_install_property(
        object_class, PROP_NET_WIDTH, g_param_spec_int("net-width", "Network Input Width", "Width of the tensor (must match mgmpreproc)", 1, G_MAXINT, 224, G_PARAM_READWRITE));
    g_object_class_install_property(
        object_class, PROP_NET_HEIGHT, g_param_spec_int("net-height", "Network Input Height", "Height of the tensor (must match mgmpreproc)", 1, G_MAXINT, 224, G_PARAM_READWRITE));
    g_object_class_install_property(
        object_class, PROP_DUMP_LOCATION, g_param_spec_string("dump-location", "Dump Location", "File path for binary tensor dump (empty = disabled)", "", G_PARAM_READWRITE));
    g_object_class_install_property(object_class, PROP_FRAME_SKIP, g_param_spec_int("frame-skip", "Frame Skip", "Dump every Nth frame (0 = dump all)", 0, G_MAXINT, 0, G_PARAM_READWRITE));

    gst_element_class_add_static_pad_template(element_class, &sink_template);
    gst_element_class_add_static_pad_template(element_class, &src_template);

    gst_element_class_set_static_metadata(element_class, "Magma Tensor Dump", "Filter/Converter/Video", "Dump tensor DMABuf to binary file and create RGB channel preview", "Magma");

    trans_class->transform_caps = gst_magma_tensor_dump_transform_caps;
    trans_class->set_caps = gst_magma_tensor_dump_set_caps;
    trans_class->transform_size = gst_magma_tensor_dump_transform_size;
    trans_class->transform = gst_magma_tensor_dump_transform;
}

GST_PLUGIN_DEFINE(GST_VERSION_MAJOR, GST_VERSION_MINOR, mgmtensordump, "Magma Tensor Dump Plugin", plugin_init, "0.1.0", "LGPL", "magma", "https://imeguras.eu.org")
