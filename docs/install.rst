Installation
============

Dependencies
------------

- ROCm 7.2.x (``/opt/rocm``)
- GStreamer >= 1.19
- Meson >= 0.60
- Ninja
- Optional: ``librdkafka``, ``protobuf``

Build
-----

.. code-block:: sh

   meson setup build --prefix=/usr --buildtype=release
   ninja -C build
   sudo ninja -C build install

Run
---

.. code-block:: sh

   GST_PLUGIN_PATH=build/gst-plugin \
   LD_LIBRARY_PATH=build/gst-plugin \
   gst-launch-1.0 ...

Both environment variables are required for HIP library resolution.

Tests
-----

.. code-block:: sh

   # All tests
   meson test -C build --print-errorlogs

   # CPU-only tests
   meson test -C build test-infer-object test-inference-meta \
     test-properties test-mgmserialize --print-errorlogs

   # GPU tests
   meson test -C build test-mgmpreproc-tensor --print-errorlogs
