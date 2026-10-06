# mgmdisplay — Current State

This document describes `mgmdisplay` as it exists at the time `mgmeglvideosink` is being
introduced. It catalogues the DRM/KMS architecture and explains why the two sinks coexist
rather than one replacing the other.

## File layout

| File | Role |
|------|------|
| `gst-plugin/include/mgmdisplay.hpp` | GObject struct + class declaration |
| `gst-plugin/src/mgmdisplay.cpp` | GstBaseSink implementation (903 lines) |
| `gst-plugin/kernels/nv12_to_xrgb8888.hip` | GPU kernel: NV12→XRGB8888 colour conversion (BT.601 limited→full) |
| `gst-plugin/kernels/fps_overlay.hip` | GPU kernel: 6×8 bitmap font overlay ("FPS: ii.f (aa.f)") |
| `docs/plugins/mgmdisplay.rst` | Element documentation |

## End-to-end data flow

```
GstBuffer (NV12)
  │
  ├─ DMABuf FD ──> hipImportExternalMemory ──> hipDeviceptr_t
  ├─ MagmaHipMeta ──> hipDeviceptr_t (direct)
  └─ System mem ──> hipMemcpy H2D ──> temp hipDeviceptr_t
                          │
                    copy_nv12_to_gbm()
                          │
                    NV12→XRGB8888 GPU kernel
                          │
                      GBM BO
                          │
                    gbm_bo_get_fd()
                          │
                    drmPrimeFDToHandle + drmModeAddFB2
                          │
                    drmModePageFlip or drmModeSetCrtc
                          │
                  ──> Display panel
```

## Why mgmdisplay requires a VT session

`mgmdisplay` opens a DRM card (`/dev/dri/cardN`) and calls `drmSetMaster()` to acquire
exclusive DRM master access for that card. This is the fundamental constraint:

1. **`drmSetMaster()` succeeds only if no other DRM master holds the card.** An active X11
   server or Wayland compositor already holds DRM master on the primary GPU. Running an
   X11 session with a window manager means the X server holds DRM master — any call to
   `drmSetMaster()` from outside that session fails.

2. **`drmModePageFlip()` and `drmModeSetCrtc()` require DRM master.** Without master, no
   framebuffer can be shown on a connected display.

3. **GBM buffer allocation happens via a render node** (`/dev/dri/renderD128+`), which does
   not require master. So `mgmdisplay` uses a split model: render node for GBM/HIP access,
   card node for display/CRTC control.

The practical consequence: `mgmdisplay` must run on a separate virtual terminal where no
display server is active. Launching it from within an X11 desktop session produces:
```
No usable DRM card found (need DRM master — run on separate VT)
```

This is not a bug — it is inherent to the DRM/KMS architecture, which assumes exclusive
ownership of the display hardware.

## Three input paths

`gst_magma_display_render()` (line 725) dispatches based on buffer memory type:

1. **DMABuf** (line 732): `gst_is_dmabuf_memory()` → get FD → `hipImportExternalMemory` →
   NV12→XRGB8888 GPU kernel → GBM BO → display.
2. **MagmaHipMeta** (line 778): Extract `hipDeviceptr_t` from meta → GPU copy to GBM BO →
   NV12→XRGB8888 GPU kernel → display.
3. **System memory** (line 788): `gst_buffer_map` → `hipMemcpy HostToDevice` → same
   conversion pipeline.

## Display submission

- **First frame:** `drmModeSetCrtc()` — sets the mode and shows the first buffer.
- **Sync mode** (`sync=true`, default): `drmModePageFlip()` with `DRM_MODE_PAGE_FLIP_EVENT`
  — non-blocking, async flip with a DRM event callback that destroys the old FB. If a flip
  is already pending, the new frame is dropped (`GST_FLOW_FLUSHING`).
- **Async mode** (`sync=false`): `drmModeSetCrtc()` each frame — immediate, tearing visible,
  no flip queue.

## FPS overlay (HIP kernel)

When `show-fps=true`, `mgmdisplay` JIT-compiles `fps_overlay.hip` and draws a text string
directly into the XRGB8888 framebuffer using a GPU kernel. The FPS measurement is
wall-clock based (`g_get_monotonic_time()`), not PTS-derived. Rolling EMA with α=0.1.

## What is NOT reusable by mgmeglvideosink

- **DRM/KMS calls** — `drmSetMaster`, `drmModePageFlip`, `drmModeSetCrtc` — not applicable
  to X11 window presentation.
- **GBM BO allocation** for scanout — `mgmeglvideosink` uses EGL+dmabuf import, not GBM.
- **HIP kernel NV12→XRGB8888** — conversion done in GL fragment shader instead.
- **HIP kernel FPS overlay** — text rendered via GL textured quads instead.
- **DRM FB management** — `drmModeAddFB2`, framebuffer ID tracking, page flip handlers.

## What IS conceptually reusable (adapted, not copied)

- **Three-path input dispatch** pattern — DMABuf / MagmaHipMeta / system memory.
- **Wall-clock FPS measurement** — `g_get_monotonic_time()` delta with EMA.
- **Bounded frame dropping** — the concept of dropping frames when the display can't keep
  up (page flip pending → drop). In `mgmeglvideosink` this becomes a bounded render queue.

## Coexistence — intentional, not a migration

`mgmdisplay` and `mgmeglvideosink` serve different deployment scenarios:

| Scenario | mgmdisplay | mgmeglvideosink |
|----------|-----------|-----------------|
| Headless / kiosk / dedicated display | ✅ Native, DRM/KMS | ❌ Needs X11 |
| Desktop / dev session (within X11) | ❌ Requires separate VT | ✅ Runs in window |
| Pipeline debugging with HUD | ✅ Fullscreen only | ✅ Windowed, resizable |
| Pipeline throughput measurement | ❌ vsync-locked | ✅ configurable vsync |
| Integration into monitoring app | ❌ DRM master conflicts | ✅ GstVideoOverlay |
| Multi-GPU headless | ✅ Select DRM card directly | ❌ Needs GPU→display match |

Both plugins ship side by side. No code is shared between them — they are independent
implementations using different display backends, linked only by the convention that both
consume NV12 DMABufs from the same upstream pipeline.
