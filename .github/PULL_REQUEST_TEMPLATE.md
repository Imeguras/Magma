## Type

<!-- Tick one (or more if it really mixes). -->
- [ ] New feature: new element, property, capability
- [ ] Bug fix
- [ ] Improvement to something existing: performance, refactor, UX
- [ ] New model / parser addon
- [ ] Docs
- [ ] Infrastructure: CI, build, packaging, Docker, tooling

- [ ] **Breaking change:** renamed/removed properties, changed caps or meta layout, parser API change

## What & why

<!-- What does this change and why? Link issues with "Closes #123". -->

## Testing

<!-- Pipelines you ran and `meson test` output. Performance changes: before/after numbers (FPS, latency). -->

### Environment

<!-- Fill in what you tested on. -->
- GPU (and gfx target):            <!-- e.g. RX 7800 XT (gfx1101) -->
- ROCm version:                    <!-- e.g. 7.2.3 -->
- Linux kernel:                    <!-- `uname -r` -->
- Distro / container:              <!-- e.g. Arch, Debian bookworm Docker image -->
- GStreamer version:               <!-- `gst-launch-1.0 --version` -->
- Mesa / amdgpu driver (if relevant):
- Model(s) and parser plugin used:

<!-- Anything else that could matter for diagnosing (compiler version, other GPUs in the system,
     custom kernel params, iGPU/dGPU setup, display server, ...), add it here. -->

## Checklist

- [ ] Commit subjects start with `Feat:` or `Fix:` (see CONTRIBUTING.md)
- [ ] `meson test -C build --print-errorlogs` passes locally (including the `gpu` suite if kernels/pre-processing/inference changed)
- [ ] Formatted with clang-format 22.1.8
- [ ] New source files have SPDX headers
