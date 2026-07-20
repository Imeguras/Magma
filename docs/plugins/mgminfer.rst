mgminfer — Model Inference
===========================

Runs a MIGraphX-compiled model (ONNX or MXR) on preprocessed tensors.
Supports parser plugin addons loaded via ``dlopen`` for custom output
decoding (YOLO, BiFormer, InternImage, …).

Pipeline position: ``… → mgmpreproc → mgminfer → mgmosd → …``

Properties
----------

.. doxygenfile:: mgminfer.hpp
   :project: Magma

Parser Plugins
--------------

Parser addons are built with ``hipcc`` and installed to
``<libdir>/magma/addons/``.  Available parsers:

* ``libyolov8-parser.so`` — YOLOv8 bounding-box decoder
* ``libmagmabiformer-desc-parser.so`` — BiFormer descriptor
* ``libmagmabiformer-detect-parser.so`` — BiFormer detection
* ``libmagmainternimage-parser.so`` — InternImage

Each is loaded at runtime — see ``addons/meson.build`` for the build pattern.
