FROM nvidia/cuda:12.6.0-runtime-ubuntu22.04

# Environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_PREFER_BINARY=1
ENV PYTHONUNBUFFERED=1
ENV CMAKE_BUILD_PARALLEL_LEVEL=8
ENV PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync
ENV MODEL_CACHE_DIR=/opt/models
ENV TEMP_DIR=/tmp/video_processing

# Base system deps
RUN apt-get update && apt-get install -y \
    software-properties-common \
    python3 python3-pip git wget ffmpeg libgl1 libglib2.0-0 \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip \
    && apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# -------------------------------------------------------------------
# Build VapourSynth from source (core C library + Python bindings)
# -------------------------------------------------------------------
RUN apt-get update && apt-get install -y \
    build-essential meson ninja-build pkg-config cmake python3-dev git \
    libffms2-dev libzimg-dev libpcre3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/vapoursynth/vapoursynth.git /tmp/vapoursynth \
    && cd /tmp/vapoursynth \
    && meson build \
    && ninja -C build \
    && ninja -C build install \
    && ldconfig \
    && cd .. && rm -rf /tmp/vapoursynth


RUN mkdir -p /opt/models/realesrgan /opt/app $TEMP_DIR

RUN wget -nv -O /opt/models/realesrgan/RealESRGAN_x2plus.pth \
    https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth

WORKDIR /opt/app

# Copy and install Python dependencies
COPY requirements.txt ./

RUN pip install --no-cache-dir torch==2.6.0 torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cu126

RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/

CMD ["python", "-u", "src/handler.py"]