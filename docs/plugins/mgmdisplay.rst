mgmdisplay — DRM/KMS Display Sink
==================================

Direct-display sink using DRM/KMS.  Takes ``MagmaHipMeta``-backed NV12
frames, converts to RGB via a HIP kernel on the GPU, and presents on a
display connector.

Pipeline position: terminal sink (``… → mgmdisplay``)

.. warning::
   Takes DRM master — only **one instance** per display.

Properties
----------

.. doxygenfile:: mgmdisplay.hpp
   :project: Magma
