project('Magma', 'c','cpp', version : '0.1.0', license : 'LGPLv3')
add_project_arguments('-DPACKAGE="magma"','-D__HIP_PLATFORM_AMD__', language: 'cpp')

plugins_install_dir = join_paths(get_option('libdir'), 'gstreamer-1.0')

cc = meson.get_compiler('c')
cpp = meson.get_compiler('cpp')
gst_version = meson.project_version()

api_version = '1.0'

gst_dep = dependency('gstreamer-1.0', version : '>=1.19', required : true, fallback : ['gstreamer', 'gst_dep'])
gstbase_dep = dependency('gstreamer-base-1.0', version : '>=1.19', fallback : ['gstreamer', 'gst_base_dep'])
gstvideo_dep = dependency('gstreamer-video-1.0', version : '>=1.19', fallback : ['gstreamer', 'gst_video_dep'])

hip_dep = dependency('hip', required: true)
hipconfig = find_program('hipconfig', required : true)

rocm_path = run_command(
  hipconfig,
  '--path',
  check : true
).stdout().strip()

hip_inc = include_directories(join_paths(rocm_path, 'include'))

subdir('gst-plugin')
