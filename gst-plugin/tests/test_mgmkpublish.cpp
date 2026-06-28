#include <gst/gst.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define TEST_ASSERT(cond, fmt, ...) \
    do { \
        if (!(cond)) { \
            g_printerr("FAIL (%s:%d): " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__); \
            failures++; \
        } else { \
            g_print("  PASS\n"); \
        } \
    } while (0)

static int failures = 0;
static GstPlugin *plugin = NULL;

static void
load_plugin(void)
{
    if (plugin) return;
    GError *err = NULL;
    plugin = gst_plugin_load_file(TEST_PLUGIN_DIR "/libmgmkpublish.so", &err);
    if (err) {
        g_printerr("FAIL: load libmgmkpublish.so: %s\n", err->message);
        g_error_free(err);
        failures++;
    } else {
        g_print("  loaded mgmkpublish\n");
    }
}

static void
unload_plugin(void)
{
    if (plugin) {
        gst_object_unref(plugin);
        plugin = NULL;
    }
}

static void
test_create(void)
{
    g_print("  test_mgmkpublish_create ...");
    GstElement *e = gst_element_factory_make("mgmkpublish", NULL);
    TEST_ASSERT(e != NULL, "failed to create mgmkpublish");
    if (e) gst_object_unref(e);
}

static void
test_defaults(void)
{
    g_print("  test_mgmkpublish_defaults ...");
    GstElement *e = gst_element_factory_make("mgmkpublish", NULL);
    TEST_ASSERT(e != NULL, "failed to create mgmkpublish");
    if (!e) return;

    gchar *str;
    g_object_get(e, "broker", &str, NULL);
    TEST_ASSERT(g_strcmp0(str, "localhost:9092") == 0, "broker default: %s", str);
    g_free(str);

    g_object_get(e, "topic", &str, NULL);
    TEST_ASSERT(g_strcmp0(str, "magma") == 0, "topic default: %s", str);
    g_free(str);

    g_object_get(e, "client-id", &str, NULL);
    TEST_ASSERT(g_strcmp0(str, "magma-publisher") == 0, "client-id default: %s", str);
    g_free(str);

    g_object_get(e, "compression", &str, NULL);
    TEST_ASSERT(g_strcmp0(str, "none") == 0, "compression default: %s", str);
    g_free(str);

    gst_object_unref(e);
}

static void
test_set_properties(void)
{
    g_print("  test_mgmkpublish_set_properties ...");
    GstElement *e = gst_element_factory_make("mgmkpublish", NULL);
    TEST_ASSERT(e != NULL, "failed to create mgmkpublish");
    if (!e) return;

    g_object_set(e, "broker", "192.168.1.100:9092", NULL);
    g_object_set(e, "topic", "detections", NULL);
    g_object_set(e, "client-id", "test-pub", NULL);
    g_object_set(e, "compression", "gzip", NULL);
    g_object_set(e, "extra-flags", "message.max.bytes=1048576", NULL);

    gchar *str;
    g_object_get(e, "broker", &str, NULL);
    TEST_ASSERT(g_strcmp0(str, "192.168.1.100:9092") == 0, "broker: %s", str);
    g_free(str);

    g_object_get(e, "topic", &str, NULL);
    TEST_ASSERT(g_strcmp0(str, "detections") == 0, "topic: %s", str);
    g_free(str);

    g_object_get(e, "compression", &str, NULL);
    TEST_ASSERT(g_strcmp0(str, "gzip") == 0, "compression: %s", str);
    g_free(str);

    gst_object_unref(e);
}

static void
test_create_with_name(void)
{
    g_print("  test_mgmkpublish_create_with_name ...");
    GstElement *e = gst_element_factory_make("mgmkpublish", "pub1");
    TEST_ASSERT(e != NULL, "failed to create mgmkpublish with name");
    if (!e) return;
    const gchar *name = gst_element_get_name(e);
    TEST_ASSERT(g_strcmp0(name, "pub1") == 0, "name: %s", name);
    gst_object_unref(e);
}

int
main(int argc, char *argv[])
{
    gst_init(&argc, &argv);
    load_plugin();

    if (plugin) {
        g_print("--- mgmkpublish ---\n");
        test_create();
        test_defaults();
        test_set_properties();
        test_create_with_name();
    }

    unload_plugin();
    g_print("\n%d failure(s)\n", failures);
    return failures ? EXIT_FAILURE : EXIT_SUCCESS;
}
