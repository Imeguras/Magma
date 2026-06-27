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
RUN wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key \
    | gpg --dearmor -o /etc/apt/keyrings/rocm.gpg && \
    echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] \
https://repo.radeon.com/rocm/apt/latest jammy main" > /etc/apt/sources.list.d/rocm.list

RUN apt-get update && apt-get install -y \
    rocm-hip-sdk \
    rocm-dev \
    && rm -rf /var/lib/apt/lists/*

# ROCm environment
ENV PATH=/opt/rocm/bin:$PATH \
    LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH \
    CPATH=/opt/rocm/include:$CPATH

# Already defined in meson.build:
#   add_project_arguments('-D__HIP_PLATFORM_AMD__', language: 'cpp')

WORKDIR /workspace
