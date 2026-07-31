mgmeglvideosink — EGL/X11 Video Sink
=====================================

Desktop-window display sink using EGL + dmabuf import.  Renders NV12
DMABuf frames to an X11 window via ``EGL_EXT_image_dma_buf_import`` and
GLES 3.x shaders.  Does **not** require DRM master — multiple instances
can coexist.

Pipeline position: terminal sink (``… → mgmeglvideosink``)

.. note::
   Runs a dedicated render thread with an internal frame queue.
   Drops the oldest pending frame when the queue reaches 2 entries
   to stay real-time.

Properties
----------

.. doxygenfile:: mgmeglvideosink.hpp
   :project: Magma

Pipeline examples
-----------------

Test with a colour-bar pattern::

   gst-launch-1.0 videotestsrc pattern=18 ! \\
       video/x-raw,format=NV12,width=640,height=480 ! \\
       mgmvideoconvert ! \\
       mgmeglvideosink window-width=640 window-height=480 sync=false

With debug HUD::

   gst-launch-1.0 videotestsrc pattern=18 ! \\
       video/x-raw,format=NV12,width=640,height=480 ! \\
       mgmvideoconvert ! \\
       mgmeglvideosink show-hud=true sync=false

Comparison with mgmdisplay
--------------------------

======================= ======================  ===========================
Feature                 mgmdisplay              mgmeglvideosink
======================= ======================  ===========================
Output                  DRM/KMS connector       X11 window
DRM master required     Yes                     No
Multiple instances      No                      Yes
Input memory            MagmaHipMeta / DMABuf   DMABuf only
Colour conversion       HIP kernel (GPU→GPU)    GLES YUV shader (GPU→GPU)
Vsync                   DRM page-flip           eglSwapInterval
HUD                     HIP kernel overlay      GLES texture overlay
Window resizing         N/A                     Yes (XCB configure notify)
======================= ======================  ===========================
