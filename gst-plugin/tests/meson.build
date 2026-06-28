gst_check_dep = dependency('gstreamer-check-1.0', required: true)

# -----------------------------------------------------------
# test-infer-object — MagmaInferObject CPU lifecycle
# -----------------------------------------------------------
test_infer_object = executable('test-infer-object',
  ['test_infer_object.cpp', '../src/magma-infer-meta.cpp'],
  cpp_args: ['-std=c++17'],
  include_directories: [inc, hip_inc],
  dependencies: [gst_dep, gstbase_dep, gstvideo_dep, gstallocators_dep,
                 gst_check_dep],
)
test('test-infer-object', test_infer_object)

# -----------------------------------------------------------
# test-inference-meta — MagmaInferenceMeta GstMeta API
# -----------------------------------------------------------
test_inference_meta = executable('test-inference-meta',
  ['test_inference_meta.cpp', '../src/magma-infer-meta.cpp'],
  cpp_args: ['-std=c++17'],
  include_directories: [inc, hip_inc],
  dependencies: [gst_dep, gstbase_dep, gstvideo_dep, gstallocators_dep,
                 gst_check_dep],
)
test('test-inference-meta', test_inference_meta)

# -----------------------------------------------------------
# test-properties — element creation & GObject property I/O
# Loads the built .so files at runtime to register the element types.
# -----------------------------------------------------------
fs = import('fs')
plugin_build_dir = fs.parent(meson.current_build_dir())

test_properties = executable('test-properties',
  'test_properties.cpp',
  cpp_args: [
    '-std=c++17',
    '-DTEST_PLUGIN_DIR="' + plugin_build_dir + '"',
  ],
  include_directories: [inc, hip_inc],
  dependencies: [gst_dep],
)
test('test-properties', test_properties,
  depends: [mgminfer_shlib, mgmpreproc_shlib, mgmvideoconvert_shlib])

# -----------------------------------------------------------
# test-mgmpreproc-tensor — NV12→tensor conversion (requires GPU)
# Loads libmgmpreproc.so at runtime.
# -----------------------------------------------------------
test_mgmpreproc_tensor = executable('test-mgmpreproc-tensor',
  ['test_mgmpreproc_tensor.cpp', '../src/magma-meta.cpp'],
  cpp_args: [
    '-std=c++17',
    '-DTEST_PLUGIN_DIR="' + plugin_build_dir + '"',
  ],
  include_directories: [inc, hip_inc],
  dependencies: [gst_dep, gstbase_dep, gstvideo_dep, gstallocators_dep,
                 gst_check_dep, hip_host_dep, gbm_dep, drm_dep],
)
test('test-mgmpreproc-tensor', test_mgmpreproc_tensor,
  depends: [mgmpreproc_shlib])

# -----------------------------------------------------------
# test-mgminfer-tensor-pipe — mgminfer reads MagmaTensorMeta,
# runs dummy kernel, attaches MagmaInferenceMeta (requires GPU)
# Loads libmgminfer.so at runtime.
# -----------------------------------------------------------
test_mgminfer_tensor_pipe = executable('test-mgminfer-tensor-pipe',
  'test_mgminfer_tensor_pipe.cpp',
  cpp_args: [
    '-std=c++17',
    '-DTEST_PLUGIN_DIR="' + plugin_build_dir + '"',
  ],
  include_directories: [inc, hip_inc],
  dependencies: [gst_dep, gstbase_dep, gstvideo_dep, gstallocators_dep,
                 gst_check_dep, hip_host_dep, gbm_dep, drm_dep],
)
test('test-mgminfer-tensor-pipe', test_mgminfer_tensor_pipe,
  depends: [mgminfer_shlib])

# -----------------------------------------------------------
# test-mgmserialize — serializes MagmaInferenceMeta → JSON/protobuf
# Loads libmgmserialize.so at runtime.
# -----------------------------------------------------------
test_mgmserialize = executable('test-mgmserialize',
  ['test_mgmserialize.cpp', '../src/magma-infer-meta.cpp'],
  cpp_args: [
    '-std=c++17',
    '-DTEST_PLUGIN_DIR="' + plugin_build_dir + '"',
  ],
  include_directories: [inc, hip_inc],
  dependencies: [gst_dep, gstbase_dep, gstvideo_dep, gstallocators_dep,
                 gst_check_dep],
)
test('test-mgmserialize', test_mgmserialize,
  depends: [mgmserialize_shlib])

# -----------------------------------------------------------
# test-mgmkpublish — Kafka sink element property I/O
# Loads libmgmkpublish.so at runtime (conditional on rdkafka).
# -----------------------------------------------------------
if rdkafka_dep.found()
    test_mgmkpublish = executable('test-mgmkpublish',
      'test_mgmkpublish.cpp',
      cpp_args: [
        '-std=c++17',
        '-DTEST_PLUGIN_DIR="' + plugin_build_dir + '"',
      ],
      include_directories: [inc],
      dependencies: [gst_dep],
    )
    test('test-mgmkpublish', test_mgmkpublish,
      depends: [mgmkpublish_shlib])
endif
