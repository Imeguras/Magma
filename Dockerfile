FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Build tools
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    meson \
    ninja-build \
    pkg-config \
    python3 \
    wget \
    gpg \
    && rm -rf /var/lib/apt/lists/*

# GStreamer 1.0 development packages (>= 1.19)
RUN apt-get update && apt-get install -y \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer-plugins-bad1.0-dev \
    gstreamer1.0-tools \
    && rm -rf /var/lib/apt/lists/*

# ROCm / HIP SDK (AMD GPU not required — compilation-only works on any machine)
# Pin to ROCm 7.2.x to match host version and avoid dependency conflicts
RUN wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key \
    | gpg --dearmor -o /etc/apt/keyrings/rocm.gpg && \
    echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] \
https://repo.radeon.com/rocm/apt/7.2.3 jammy main" > /etc/apt/sources.list.d/rocm.list && \
    echo "Package: *" > /etc/apt/preferences.d/rocm && \
    echo "Pin: release o=repo.radeon.com" >> /etc/apt/preferences.d/rocm && \
    echo "Pin-Priority: 1001" >> /etc/apt/preferences.d/rocm

RUN apt-get update && apt-get install -y \
    rocm-hip-runtime-dev \
    rocm-device-libs \
    rocm-cmake \
    && rm -rf /var/lib/apt/lists/*

# ROCm environment
ENV PATH=/opt/rocm/bin:$PATH

# Create render group with host GID so group_add in docker-compose resolves cleanly
RUN groupadd -g 989 render

# Already defined in meson.build:
#   add_project_arguments('-D__HIP_PLATFORM_AMD__', language: 'cpp')

WORKDIR /workspace
