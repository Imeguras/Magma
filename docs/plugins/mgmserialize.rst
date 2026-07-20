mgmserialize — Inference Result Serialization
==============================================

Serializes ``MagmaInferenceMeta`` detection results to JSON or Protobuf,
output as a GStreamer buffer.

Pipeline position: ``… → mgminfer → mgmserialize → …`` (typically at end
of pipeline before a filesink/appsink)

Properties
----------

.. doxygenfile:: mgmserialize.hpp
   :project: Magma
