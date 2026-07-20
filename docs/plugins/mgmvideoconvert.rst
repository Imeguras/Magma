mgmvideoconvert — Video Format Conversion
==========================================

Converts between system memory, DMABuf, and ``MagmaHipMeta``-backed video
frames.  Handles NV12 ↔ I420 colour-space conversion via a dispatch table.

Pipeline position: used between any two elements with incompatible memory
types or formats.

Input / Output
--------------

Accepts any of the three memory paths:

* **DMABuf** — imported, optionally converted, and re-exported
* **MagmaHipMeta** — device pointer copied/converted directly
* **System memory** — uploaded to device, converted, output as ``MagmaHipMeta``

Properties
----------

.. doxygengroup:: mgmvideoconvert-props
   :project: Magma
   :content-only:

API
---

.. doxygenfile:: mgmvideoconvert.hpp
   :project: Magma
