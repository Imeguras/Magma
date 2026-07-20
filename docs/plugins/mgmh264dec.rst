mgmh264dec — H.264 GPU Decoder
===============================

Hardware-accelerated H.264 decoding via rocDecode.  Produces NV12 or I420
frames backed by ``MagmaHipMeta`` on the device.

Pipeline position: ``source → mgmh264dec → mgmvideoconvert → …``

Input caps: ``video/x-h264`` (AVC/byte-stream)

Output caps: ``video/x-raw`` (NV12/I420)

Properties
----------

No user-configurable properties currently exposed.

Notes
-----

- Uses ``OUT_SURFACE_MEM_DEV_INTERNAL`` decoder surfaces; the INTERNAL
  buffer is released immediately after the copy to the output buffer is
  enqueued on the shared Magma stream.
- DMABuf export is performed on every output frame so downstream elements
  that prefer DMABuf memory can consume it directly.
- PTS is converted to/from 10 MHz rocDecode units via ``clk_rate = 10000000``.

API
---

.. doxygenfile:: mgmh264dec.hpp
   :project: Magma
