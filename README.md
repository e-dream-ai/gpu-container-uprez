# GPU Video Upscaling & Interpolation Container

A RunPod serverless container that performs Real-ESRGAN upscaling and RIFE frame interpolation in a single processing pipeline.

## Architecture

```
Input Video → Frame Extraction → Real-ESRGAN Upscaling → RIFE Interpolation → Output Video
```

## Usage

### Input Parameters

```json
{
  "video_url": "https://example.com/input.mp4",
  "upscale_factor": 2, // 2 only
  "interpolation_factor": 2, // 2, 4, or 8
  "output_fps": 30, // 1-120
  "output_format": "mp4" // mp4, webm, avi
}
```

### Output

```json
{
  "status": "success",
  "download_url": "https://...",
  "processing_info": {
    "upscale_factor": 2,
    "interpolation_factor": 2,
    "output_fps": 30,
    "output_format": "mp4"
  }
}
```

## Building and Deployment

### Local Build

```bash
cd gpu-container-uprez
docker build -t video-upscaler .
```

### RunPod Deployment

1. Build and push to registry:

```bash
docker build -t your-registry/video-upscaler:latest .
docker push your-registry/video-upscaler:latest
```

2. Create RunPod serverless endpoint with:
   - Container Image: `your-registry/video-upscaler:latest`
   - GPU: RTX 4090 or better (recommended)
   - Memory: 16GB+ RAM
   - Storage: 50GB+ for temporary processing

## Technical Details

### Models Used

- **Real-ESRGAN**: RealESRGAN_x2plus.pth
- **RIFE**: rife4.26.pth

### Memory Requirements

- **Minimum**: 8GB GPU memory for 720p videos
- **Recommended**: 16GB+ GPU memory for 1080p+ videos

### Cloudflare R2 Uploads

This container can upload the processed video to Cloudflare R2 and return a presigned URL (similar to the deforum container). Configure these environment variables on your RunPod endpoint:

- `R2_BUCKET_NAME`: Target R2 bucket name
- `R2_ENDPOINT_URL`: R2 S3 endpoint (`https://<account-id>.r2.cloudflarestorage.com`)
- `R2_ACCESS_KEY_ID`: R2 access key
- `R2_SECRET_ACCESS_KEY`: R2 secret key
- `R2_UPLOAD_DIRECTORY` (optional): Prefix for uploaded objects (default: `video-outputs`)
- `R2_PRESIGNED_EXPIRY` (optional): Expiration in seconds for presigned URL (default: `86400`)
