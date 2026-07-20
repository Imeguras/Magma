mgmosd — On-Screen Display
===========================

Renders bounding boxes, labels, and confidence scores onto video frames
using GPU HIP kernels.  Outputs ``MagmaHipMeta``-backed frames.

Pipeline position: ``… → mgminfer → mgmosd → mgmvideoconvert → … | sink``

Properties
----------

No user-configurable properties currently exposed.

API
---

.. doxygenfile:: mgmosd.hpp
   :project: Magma
