#!/bin/bash

set -eox pipefail

echo "🚀 Building and testing video upscaler container"

# Check if we're on a GPU instance
if ! nvidia-smi > /dev/null 2>&1; then
    echo "❌ No GPU detected. This container requires NVIDIA GPU support."
    exit 1
fi

echo "✅ GPU detected:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Build the container
echo "🔨 Building Docker container..."
docker build -t video-upscaler:test .

echo "✅ Container built successfully"

# Test the container with a simple job
echo "🧪 Testing container..."

# Test using existing settings file
echo "Using test_settings.json for testing"

# Run container test
echo "Running container test..."
docker run --gpus all --rm \
  -v $(pwd):/workspace \
  video-upscaler:test \
  python /opt/app/test_with_json.py test_settings.json

echo "✅ Container test completed successfully!"

