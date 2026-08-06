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
  "upscale_factor": 2, // 1 (skip), 2, or 4
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

1. Automatic build and push

   - Workflow: `.github/workflows/build-and-push.yml`
   - Registry: `docker.io/edreamai/video-upscaler`
   - Tags produced per build:
     - `latest`
     - Timestamped tag like `docker.io/edreamai/video-upscaler:YYYYMMDDHHMMSS-main[-custom]`

2. Update your RunPod Serverless endpoint

   - Open your existing Serverless endpoint in RunPod console
   - Change the Container Image to the newly built tag printed in the GitHub Actions run summary
   - Example: `docker.io/edreamai/video-upscaler:20250206-main`
   - Save and redeploy

3. New endpoint setup (if creating fresh)
   - Container Image: use the latest tag from the workflow output

## Technical Details

### Models Used

- **Real-ESRGAN**: RealESRGAN_x2plus.pth (2x), RealESRGAN_x4plus.pth (4x)
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
