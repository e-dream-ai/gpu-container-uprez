FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_PREFER_BINARY=1
ENV PYTHONUNBUFFERED=1
ENV CMAKE_BUILD_PARALLEL_LEVEL=8
ENV PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync
ENV MODEL_CACHE_DIR=/opt/models
ENV TEMP_DIR=/tmp/video_processing

# System dependencies (including VapourSynth dependencies)
RUN apt-get update && apt-get install -y \
    python3.11 python3-pip git wget ffmpeg libgl1 libglib2.0-0 \
    vapoursynth vapoursynth-dev \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip \
    && apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt/models/realesrgan /opt/app $TEMP_DIR

RUN wget -nv -O /opt/models/realesrgan/RealESRGAN_x2plus.pth \
    https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth

# Set working directory
WORKDIR /opt/app

# Copy requirements and install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/

# Set entrypoint
CMD ["python", "-u", "src/handler.py"]
