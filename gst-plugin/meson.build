plugin_c_args = ['-DHAVE_CONFIG_H']

cdata = configuration_data()
cdata.set_quoted('PACKAGE_VERSION', gst_version)
cdata.set_quoted('PACKAGE', 'magma-gst-plugins')
cdata.set_quoted('GST_LICENSE', 'LGPL')
cdata.set_quoted('GST_API_VERSION', api_version)
cdata.set_quoted('GST_PACKAGE_NAME', 'Magma GStreamer Plugins')
cdata.set_quoted('GST_PACKAGE_ORIGIN', 'https://magma.local')
configure_file(output : 'config.h', configuration : cdata)

hip_host_dep = cpp.find_library('amdhip64', dirs : ['/opt/rocm/lib'], required : true)
hiprtc_dep = cpp.find_library('hiprtc', dirs : ['/opt/rocm/lib'], required : true)
gstallocators_dep = dependency('gstreamer-allocators-1.0', required : true)
gbm_dep = dependency('gbm', required : true)
drm_dep = dependency('libdrm', required : true)

migraphx_dep = declare_dependency(
  include_directories: include_directories('/opt/rocm/include'),
  dependencies: cpp.find_library('migraphx_c', dirs: '/opt/rocm/lib'),
)

inc = include_directories('include')

kernel_src_dir = meson.current_source_dir() / 'kernels'

# Shared meta library — single compilation of meta types to avoid double-registration
magma_meta_sources = [
  'src/magma-meta.cpp',
  'src/magma-infer-meta.cpp',
]

magma_meta_shlib = shared_library('magma-meta',
  magma_meta_sources,
  c_args: plugin_c_args,
  cpp_args: ['-std=c++17'],
  include_directories: [inc, hip_inc],
  dependencies: [gst_dep],
  install: true,
  install_dir: get_option('libdir'),
)

magma_meta_dep = declare_dependency(
  link_with: magma_meta_shlib,
  include_directories: inc,
)

# Shared kernel compilation utility (static lib linked by all GPU plugins)
kernel_utils_src = ['kernels/kernel_utils.cpp']

kernel_utils_lib = static_library('kernel_utils', kernel_utils_src,
  cpp_args: ['-std=c++17'],
  include_directories: [inc, hip_inc],
  dependencies: [hip_host_dep, hiprtc_dep],
)

kernel_utils_dep = declare_dependency(
  link_with: kernel_utils_lib,
  include_directories: include_directories('kernels'),
)

# Plugin: mgminfer
mgminfer_sources = [
  'src/mgminfer.cpp',
]

mgminfer_shlib = shared_library('mgminfer',
  mgminfer_sources,
  c_args: plugin_c_args,
  cpp_args: [
    '-std=c++17',
    '-DMAGMA_KERNEL_SRC_DIR="' + kernel_src_dir + '"',
  ],
  include_directories: [inc, hip_inc],
  dependencies: [gst_dep, gstbase_dep, gstvideo_dep, gstallocators_dep,
                 kernel_utils_dep, gbm_dep, drm_dep, hip_host_dep,
                 migraphx_dep, magma_meta_dep],
  install: true,
  install_dir: plugins_install_dir,
)

# Plugin: mgmpreproc
mgmpreproc_sources = [
  'src/mgmpreproc.cpp',
]

mgmpreproc_shlib = shared_library('mgmpreproc',
  mgmpreproc_sources,
  c_args: plugin_c_args,
  cpp_args: [
    '-std=c++17',
    '-DMAGMA_KERNEL_SRC_DIR="' + kernel_src_dir + '"',
  ],
  include_directories: [inc, hip_inc],
  dependencies: [gst_dep, gstbase_dep, gstvideo_dep, gstallocators_dep,
                 kernel_utils_dep, gbm_dep, drm_dep, magma_meta_dep],
  install: true,
  install_dir: plugins_install_dir,
)

# Plugin: mgmvideoconvert
mgmvideoconvert_sources = [
  'src/mgmvideoconvert.cpp',
]

mgmvideoconvert_shlib = shared_library('mgmvideoconvert',
  mgmvideoconvert_sources,
  c_args: plugin_c_args,
  cpp_args: ['-std=c++17'],
  include_directories: [inc, hip_inc],
  dependencies: [gst_dep, gstbase_dep, gstvideo_dep, gstallocators_dep,
                 hip_host_dep, gbm_dep, drm_dep],
  install: true,
  install_dir: plugins_install_dir,
)

# Install kernel sources for runtime JIT loading on deployed systems
install_data('kernels/preproc_kernels.hip', 'kernels/dummy_infer.hip',
  install_dir: get_option('datadir') / 'magma' / 'kernels',
)

# ========================================================
# Optional: Protobuf message definition
# ========================================================
protoc = find_program('protoc', required: false)
have_protobuf = protoc.found()
protobuf_dep = dependency('protobuf', required: false)

if have_protobuf and protobuf_dep.found()
    magma_msg_pb = custom_target('magma_msg_pb',
        input: 'magma_msg.proto',
        output: ['magma_msg.pb.cc', 'magma_msg.pb.h'],
        command: [protoc, '--cpp_out=@OUTDIR@', '--proto_path=@CURRENT_SOURCE_DIR@', '@INPUT@'],
    )
    magma_msg_lib = static_library('magma_msg_pb', magma_msg_pb,
        cpp_args: ['-std=c++17'],
        dependencies: protobuf_dep,
    )
    magma_msg_dep = declare_dependency(
        link_with: magma_msg_lib,
        sources: magma_msg_pb,
    )
    magma_msg_cpp_args = ['-DHAVE_PROTOBUF']
else
    magma_msg_dep = disabler()
    magma_msg_cpp_args = []
endif

# ========================================================
# Plugin: mgmserialize — serializes MagmaInferenceMeta → msg
# ========================================================
mgmserialize_sources = [
  'src/mgmserialize.cpp',
]

mgmserialize_shlib = shared_library('mgmserialize',
  mgmserialize_sources,
  c_args: plugin_c_args,
  cpp_args: ['-std=c++17'] + magma_msg_cpp_args,
  include_directories: [inc],
  dependencies: [gst_dep, gstbase_dep, gstvideo_dep, gstallocators_dep,
                 magma_meta_dep, magma_msg_dep],
  install: true,
  install_dir: plugins_install_dir,
)

# ========================================================
# Plugin: mgmkpublish — Kafka sink (conditional on librdkafka)
# ========================================================
rdkafka_dep = dependency('rdkafka', required: false)

if rdkafka_dep.found()
    mgmkpublish_sources = ['src/mgmkpublish.cpp']
    mgmkpublish_shlib = shared_library('mgmkpublish',
      mgmkpublish_sources,
      c_args: plugin_c_args,
      cpp_args: ['-std=c++17'],
      include_directories: [inc],
      dependencies: [gst_dep, gstbase_dep, gstallocators_dep, rdkafka_dep],
      install: true,
      install_dir: plugins_install_dir,
    )
else
    mgmkpublish_shlib = disabler()
endif

subdir('tests')
