FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_PREFER_BINARY=1
ENV PYTHONUNBUFFERED=1
ENV CMAKE_BUILD_PARALLEL_LEVEL=8
ENV PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync
ENV MODEL_CACHE_DIR=/opt/models
ENV TEMP_DIR=/tmp/video_processing

# System dependencies
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip git wget ffmpeg libgl1 libglib2.0-0 \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip \
    && apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Install PyTorch first (required as build dependency for basicsr and others)
RUN pip install --default-timeout=100 --no-cache-dir \
    torch==2.0.1 \
    torchvision==0.15.2 \
    --extra-index-url https://download.pytorch.org/whl/cu118

# Create directories
RUN mkdir -p /opt/models/realesrgan /opt/models/rife /opt/app $TEMP_DIR

# Download Real-ESRGAN weights only (RIFE code/weights are vendored locally)
RUN wget -nv -O /opt/models/realesrgan/RealESRGAN_x2plus.pth \
    https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth

# Set working directory
WORKDIR /opt/app

# Copy application code
COPY src/ ./src/
COPY src/vendor/rife/model /opt/app/src/vendor/rife/model
COPY requirements.txt ./

# Install remaining requirements
RUN pip install --default-timeout=100 --no-cache-dir -r requirements.txt

# Set entrypoint
CMD ["python", "-u", "src/handler.py"]