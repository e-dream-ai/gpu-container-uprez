import os
import tempfile
import logging
from typing import Dict, Any
from pathlib import Path

import runpod
import uuid
import boto3
from botocore.config import Config as BotoConfig
from runpod.serverless.utils.rp_validator import validate

from services.video_processor import VideoProcessorService
from utils.cleanup_manager import CleanupManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

INPUT_SCHEMA = {
    'video_url': {
        'type': str,
        'required': False,
        'description': 'URL of the input video to process'
    },
    'video_path': {
        'type': str,
        'required': False,
        'description': 'Local path to input video file'
    },
    'upscale_factor': {
        'type': int,
        'required': False,
        'default': 2,
        'constraints': lambda x: x == 2,
        'description': 'Upscaling factor (2x only)'
    },
    'interpolation_factor': {
        'type': int,
        'required': False,
        'default': 2,
        'constraints': lambda x: x in [2, 4, 8],
        'description': 'Frame interpolation factor (2x, 4x, or 8x)'
    },
    'output_fps': {
        'type': int,
        'required': False,
        'default': 30,
        'constraints': lambda x: 1 <= x <= 120,
        'description': 'Output video frame rate'
    },
    'output_format': {
        'type': str,
        'required': False,
        'default': 'mp4',
        'constraints': lambda x: x in ['mp4', 'webm', 'avi'],
        'description': 'Output video format'
    },
    'tile_size': {
        'type': int,
        'required': False,
        'default': 512,
        'constraints': lambda x: x in [256, 512, 1024, 2048],
        'description': 'Tile size for upscaling (larger = better quality, more memory)'
    },
    'tile_padding': {
        'type': int,
        'required': False,
        'default': 10,
        'constraints': lambda x: 0 <= x <= 50,
        'description': 'Padding between tiles to prevent seams'
    },
    'quality': {
        'type': str,
        'required': False,
        'default': 'high',
        'constraints': lambda x: x in ['low', 'medium', 'high'],
        'description': 'Output quality preset'
    }
}


def download_input_video(video_url: str, temp_dir: Path) -> Path:
    import requests
    
    logger.info(f"Downloading video from: {video_url}")
    
    file_ext = Path(video_url).suffix or '.mp4'
    input_path = temp_dir / f"input{file_ext}"
    
    try:
        response = requests.get(video_url, stream=True, timeout=300)
        response.raise_for_status()
        
        with open(input_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        logger.info(f"Video downloaded successfully: {input_path}")
        return input_path
        
    except Exception as e:
        logger.error(f"Failed to download video: {e}")
        raise RuntimeError(f"Video download failed: {e}")


def upload_output_video(video_path: Path) -> str:
    """Upload processed video to R2 and return a presigned download URL."""
    logger.info(f"Preparing upload for: {video_path}")

    bucket_name = os.environ.get("R2_BUCKET_NAME")
    endpoint_url = os.environ.get("R2_ENDPOINT_URL")
    r2_key = os.environ.get("R2_ACCESS_KEY_ID")
    r2_secret = os.environ.get("R2_SECRET_ACCESS_KEY")
    upload_directory = os.environ.get("R2_UPLOAD_DIRECTORY", "video-outputs")
    expiration_seconds = int(os.environ.get("R2_PRESIGNED_EXPIRY", "86400"))

    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=r2_key,
        aws_secret_access_key=r2_secret,
        region_name="auto",
        config=BotoConfig(s3={"addressing_style": "path"})
    )

    object_key = f"{upload_directory}/{uuid.uuid4()}{video_path.suffix or '.mp4'}"
    logger.info(f"Uploading to r2://{bucket_name}/{object_key}")
    s3.upload_file(str(video_path), bucket_name, object_key)

    try:
        presigned_url = s3.generate_presigned_url(
            'get_object',
            Params={'Bucket': bucket_name, 'Key': object_key},
            ExpiresIn=expiration_seconds
        )
        logger.info("Generated presigned URL for download")
        return presigned_url
    except Exception as e:
        logger.error(f"Failed to generate presigned URL: {e}")
        return f"{endpoint_url}/{bucket_name}/{object_key}"


def get_input_video_path(params: Dict[str, Any], temp_dir: Path) -> Path:
    if 'video_url' in params and params['video_url']:
        return download_input_video(params['video_url'], temp_dir)
    elif 'video_path' in params and params['video_path']:
        video_path = Path(params['video_path'])
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        return video_path
    else:
        raise ValueError("Either 'video_url' or 'video_path' must be provided")


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    cleanup_manager = CleanupManager()
    
    try:
        validated = validate(job, INPUT_SCHEMA)
        
        if validated.get('errors'):
            return {'error': f"Input validation failed: {validated['errors']}"}
        
        params = validated['validated_input']
        logger.info(f"Processing video with parameters: {params}")
        
        temp_dir = Path(tempfile.mkdtemp(prefix='video_proc_'))
        cleanup_manager.add_directory(temp_dir)
        
        input_video_path = get_input_video_path(params, temp_dir)
        
        processor = VideoProcessorService(
            temp_dir=temp_dir,
            cleanup_manager=cleanup_manager
        )
        
        output_video_path = processor.process_video(
            input_path=input_video_path,
            upscale_factor=params['upscale_factor'],
            interpolation_factor=params['interpolation_factor'],
            output_fps=params['output_fps'],
            output_format=params['output_format'],
            tile_size=params['tile_size'],
            tile_padding=params['tile_padding'],
            quality=params['quality']
        )
        
        download_url = upload_output_video(output_video_path)
        
        return {
            'status': 'success',
            'download_url': download_url,
            'processing_info': {
                'upscale_factor': params['upscale_factor'],
                'interpolation_factor': params['interpolation_factor'],
                'output_fps': params['output_fps'],
                'output_format': params['output_format']
            }
        }
        
    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        return {
            'status': 'error',
            'error': str(e)
        }
    
    finally:
        cleanup_manager.cleanup_all()


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
