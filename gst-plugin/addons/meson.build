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
