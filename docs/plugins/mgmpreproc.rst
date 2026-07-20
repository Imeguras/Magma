mgmpreproc — Preprocessing & Tensor Normalization
==================================================

Crops, resizes, and normalizes video frames into float32 RGB tensors for
model inference.  Supports ROI cropping and configurable input dimensions.

Pipeline position: ``… → mgmvideoconvert → mgmpreproc → mgminfer → …``

Properties
----------

.. doxygenfile:: mgmpreproc.hpp
   :project: Magma
   :sections: func
