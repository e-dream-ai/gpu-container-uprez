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

# Install PyTorch with CUDA 11.8
RUN pip install --no-cache-dir torch torchvision torchaudio \
    --extra-index-url https://download.pytorch.org/whl/cu118

# Install Python dependencies
RUN pip install --no-cache-dir \
    runpod \
    requests \
    ffmpeg-python \
    opencv-python-headless \
    numpy \
    Pillow \
    tqdm \
    basicsr \
    gfpgan \
    realesrgan \
    gdown

# Create directories
RUN mkdir -p /opt/models/realesrgan /opt/models/rife /opt/app $TEMP_DIR

# Download models
RUN gdown --id 1gViYvvQrtETBgU1w8axZSsr7YUuw31uy -O /opt/models/rife/rife4.26.pth
RUN wget -nv -O /opt/models/realesrgan/RealESRGAN_x2plus.pth \
    https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth

# Clone RIFE repository for inference code
WORKDIR /opt
RUN git clone https://github.com/megvii-research/ECCV2022-RIFE.git rife-repo

# Set working directory
WORKDIR /opt/app

# Copy application code
COPY src/ ./src/
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Set entrypoint
CMD ["python", "-u", "src/handler.py"]