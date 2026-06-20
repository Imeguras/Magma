plugin_c_args = ['-DHAVE_CONFIG_H']

cdata = configuration_data()
cdata.set_quoted('PACKAGE_VERSION', gst_version)
cdata.set_quoted('PACKAGE', 'magma-gst-plugins')
cdata.set_quoted('GST_LICENSE', 'LGPL')
cdata.set_quoted('GST_API_VERSION', api_version)
cdata.set_quoted('GST_PACKAGE_NAME', 'Magma GStreamer Plugins')
cdata.set_quoted('GST_PACKAGE_ORIGIN', 'https://magma.local')
configure_file(output : 'config.h', configuration : cdata)

inc = include_directories('include')

# Plugin: mgminfer
mgminfer_sources = [
  'src/mgminfer.cpp',
]

shared_library('mgminfer',
  mgminfer_sources,
  c_args: plugin_c_args,
  cpp_args: ['-std=c++17'], 
  include_directories: [inc, hip_inc],
  dependencies: [gst_dep],
  install: true,
  install_dir: plugins_install_dir,
)
# Plugin: mgmpreproc
mgmpreproc_sources = [
  'src/mgmpreproc.cpp',
]

shared_library('mgmpreproc',
  mgmpreproc_sources,
  c_args: plugin_c_args,
  cpp_args: ['-std=c++17'],
  include_directories: [inc, hip_inc],
  dependencies: [gst_dep, gstbase_dep],
  install: true,
  install_dir: plugins_install_dir,
)
