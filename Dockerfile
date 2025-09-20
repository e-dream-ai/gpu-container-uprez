FROM nvidia/cuda:12.6.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_PREFER_BINARY=1
ENV PYTHONUNBUFFERED=1
ENV CMAKE_BUILD_PARALLEL_LEVEL=8
ENV PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync
ENV MODEL_CACHE_DIR=/opt/models
ENV TEMP_DIR=/tmp/video_processing

RUN apt-get update && apt-get install -y \
    software-properties-common \
    python3 python3-pip git wget ffmpeg libgl1 libglib2.0-0 \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip \
    && apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
    build-essential pkg-config automake autoconf libtool python3-dev git \
    libpcre3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/sekrit-twc/zimg.git /tmp/zimg \
    && cd /tmp/zimg \
    && ./autogen.sh \
    && ./configure \
    && make -j$(nproc) \
    && make install \
    && ldconfig \
    && cd .. && rm -rf /tmp/zimg

RUN git clone https://github.com/vapoursynth/vapoursynth.git /tmp/vapoursynth \
    && cd /tmp/vapoursynth \
    && ./autogen.sh \
    && ./configure \
    && make -j$(nproc) \
    && make install \
    && ldconfig \
    && cd .. && rm -rf /tmp/vapoursynth

RUN mkdir -p /opt/models/realesrgan /opt/app $TEMP_DIR

RUN wget -nv -O /opt/models/realesrgan/RealESRGAN_x2plus.pth \
    https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth

WORKDIR /opt/app

COPY requirements.txt ./

RUN pip install --no-cache-dir torch==2.6.0 torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cu126

RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

CMD ["python", "-u", "src/handler.py"]