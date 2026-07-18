# Build YOLO parser shared library (loaded at runtime by mgminfer via dlopen)

yolo_parser_shlib = custom_target('libyolov8-parser',
  input: [
    'magma-yolo/src/parse.cpp',
    'magma-yolo/src/decode_nms_kernels.hip',
  ],
  output: 'libyolov8-parser.so',
  command: [
    'hipcc',
    '-std=c++17', '-fPIC', '-O3',
    '-I@CURRENT_SOURCE_DIR@/magma-yolo/include',
    '-I/opt/rocm/include',
    '-shared',
    '-L/opt/rocm/lib', '-lamdhip64',
    '@INPUT@',
    '-o', '@OUTPUT@',
  ],
  build_by_default: true,
  install: true,
  install_dir: get_option('libdir') / 'magma' / 'addons',
)

# add for magma-biformervit-desc parser shared library (loaded at runtime by mgminfer via dlopen)
biformervit_desc_parser_shlib = custom_target('libmagmabiformer-desc-parser',
  input: [
    'magma-biformervit-desc/src/parse.cpp',
    'magma-biformervit-desc/src/shapes.hip',
  ],
  output: 'libmagmabiformer-desc-parser.so',
  command: [
    'hipcc',
    '-std=c++17', '-fPIC', '-O3',
    '-I@CURRENT_SOURCE_DIR@/magma-biformervit-desc/include',
    '-I/opt/rocm/include',
    '-shared',
    '-L/opt/rocm/lib', '-lamdhip64',
    '@INPUT@',
    '-o', '@OUTPUT@',
  ],
  build_by_default: true,
  install: true,
  install_dir: get_option('libdir') / 'magma' / 'addons',
)

# add for magma-biformervit-detect parser shared library (loaded at runtime by mgminfer via dlopen)
biformervit_detect_parser_shlib = custom_target('libmagmabiformer-detect-parser',
  input: [
    'magma-biformervit-detect/src/parse.cpp',
    'magma-biformervit-detect/src/detect_kernels.hip',
  ],
  output: 'libmagmabiformer-detect-parser.so',
  command: [
    'hipcc',
    '-std=c++17', '-fPIC', '-O3',
    '-I@CURRENT_SOURCE_DIR@/magma-biformervit-detect/include',
    '-I/opt/rocm/include',
    '-shared',
    '-L/opt/rocm/lib', '-lamdhip64',
    '@INPUT@',
    '-o', '@OUTPUT@',
  ],
  build_by_default: true,
  install: true,
  install_dir: get_option('libdir') / 'magma' / 'addons',
)

# add for magma-internimage parser shared library (loaded at runtime by mgminfer via dlopen)
internimage_parser_shlib = custom_target('libmagmainternimage-parser',
  input: [
    'magma-internimage/src/parse.cpp',
    'magma-internimage/src/decode_nms_kernels.hip',
  ],
  output: 'libmagmainternimage-parser.so',
  command: [
    'hipcc',
    '-std=c++17', '-fPIC', '-O3',
    '-I@CURRENT_SOURCE_DIR@/magma-internimage/include',
    '-I/opt/rocm/include',
    '-shared',
    '-L/opt/rocm/lib', '-lamdhip64',
    '@INPUT@',
    '-o', '@OUTPUT@',
  ],
  build_by_default: true,
  install: true,
  install_dir: get_option('libdir') / 'magma' / 'addons',
)