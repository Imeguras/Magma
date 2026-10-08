# Security Policy

## Supported versions

Only the latest release (the newest tag on `master`) receives security fixes.

## Reporting a vulnerability

**Please do not open a public issue.** Report privately through GitHub:
**Security** tab → **Report a vulnerability**.

This is a hobby-maintained project, so responses are best effort. Confirmed issues are
fixed in the next release and disclosed through a GitHub security advisory, crediting the
reporter unless they prefer otherwise.

## Scope

Magma processes media and runs code on the GPU, so the trust boundaries are:

**In scope:**
- Crashes, hangs or memory corruption caused by malformed or hostile **input streams**
  (H.264 bitstreams, video frames, caps negotiation) in any `mgm*` element.
- Out-of-bounds reads/writes in Magma's HIP kernels, pre-processing, parser addons, or
  serialization (`mgmserialize`, `mgmkpublish`).

**Out of scope:**
- **Models and parser plugins are trusted input.** `mgminfer` loads the ONNX/MXR files and
  `dlopen`s the parser plugin you configure, and both can execute arbitrary code by design.
  Only load models and plugins you trust.
- Bugs in upstream components (ROCm, MIGraphX, rocDecode, GStreamer, librdkafka); report
  those to their maintainers.
- Denial of service through deliberately huge but valid input (e.g. very large resolutions).
