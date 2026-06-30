#include <gst/check/gstcheck.h>
#include "magma-infer-meta.h"

GST_START_TEST(test_infer_object_new_defaults) {
    MagmaInferObject* obj = magma_infer_object_new(0, NULL, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
    ck_assert(obj != NULL);
    ck_assert_int_eq(obj->class_id, 0);
    ck_assert(obj->label == NULL);
    ck_assert(fabsf(obj->confidence - 0.0f) < 0.0001f);
    ck_assert(fabsf(obj->x - 0.0f) < 0.0001f);
    ck_assert(fabsf(obj->y - 0.0f) < 0.0001f);
    ck_assert(fabsf(obj->width - 0.0f) < 0.0001f);
    ck_assert(fabsf(obj->height - 0.0f) < 0.0001f);
    magma_infer_object_free(obj);
}
GST_END_TEST

GST_START_TEST(test_infer_object_new_with_label) {
    MagmaInferObject* obj = magma_infer_object_new(1, "person", 0.95f, 0.1f, 0.2f, 0.3f, 0.4f);
    ck_assert(obj != NULL);
    ck_assert_int_eq(obj->class_id, 1);
    ck_assert_str_eq(obj->label, "person");
    ck_assert(fabsf(obj->confidence - 0.95f) < 0.0001f);
    ck_assert(fabsf(obj->x - 0.1f) < 0.0001f);
    ck_assert(fabsf(obj->y - 0.2f) < 0.0001f);
    ck_assert(fabsf(obj->width - 0.3f) < 0.0001f);
    ck_assert(fabsf(obj->height - 0.4f) < 0.0001f);
    magma_infer_object_free(obj);
}
GST_END_TEST

GST_START_TEST(test_infer_object_new_null_label) {
    MagmaInferObject* obj = magma_infer_object_new(42, NULL, 0.5f, 0.0f, 0.0f, 0.0f, 0.0f);
    ck_assert(obj != NULL);
    ck_assert_int_eq(obj->class_id, 42);
    ck_assert(obj->label == NULL);
    magma_infer_object_free(obj);
}
GST_END_TEST

GST_START_TEST(test_infer_object_free_null) {
    magma_infer_object_free(NULL);
}
GST_END_TEST

GST_START_TEST(test_infer_object_stress_alloc_free) {
    for (int i = 0; i < 500; i++) {
        MagmaInferObject* obj = magma_infer_object_new((guint)i, "stress", (gfloat)i / 500.0f, (gfloat)i, (gfloat)i, (gfloat)i, (gfloat)i);
        ck_assert_int_eq(obj->class_id, (guint)i);
        magma_infer_object_free(obj);
    }
}
GST_END_TEST

GST_START_TEST(test_infer_object_label_is_independent) {
    const gchar* original = "cat";
    MagmaInferObject* obj = magma_infer_object_new(0, original, 1.0f, 0, 0, 0, 0);
    ck_assert(obj->label != original);
    ck_assert_str_eq(obj->label, original);
    magma_infer_object_free(obj);
}
GST_END_TEST

static Suite* infer_object_suite(void) {
    Suite* s = suite_create("magma_infer_object");
    TCase* tc = tcase_create("general");
    tcase_add_test(tc, test_infer_object_new_defaults);
    tcase_add_test(tc, test_infer_object_new_with_label);
    tcase_add_test(tc, test_infer_object_new_null_label);
    tcase_add_test(tc, test_infer_object_free_null);
    tcase_add_test(tc, test_infer_object_stress_alloc_free);
    tcase_add_test(tc, test_infer_object_label_is_independent);
    suite_add_tcase(s, tc);
    return s;
}

int main(int argc, char* argv[]) {
    gst_init(&argc, &argv);
    Suite* s = infer_object_suite();
    SRunner* sr = srunner_create(s);
    srunner_run_all(sr, CK_NORMAL);
    int nf = srunner_ntests_failed(sr);
    srunner_free(sr);
    return nf ? EXIT_FAILURE : EXIT_SUCCESS;
}
