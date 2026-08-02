import os
import time
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
from edream_sdk.client import create_edream_client

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

BACKEND_URL = os.getenv("BACKEND_URL")
BACKEND_API_KEY = os.getenv("BACKEND_API_KEY")
edream_client = None
if BACKEND_URL and BACKEND_API_KEY:
    edream_client = create_edream_client(backend_url=BACKEND_URL, api_key=BACKEND_API_KEY)
else:
    logger.warning("BACKEND_URL or BACKEND_API_KEY not set; video_uuid will be unsupported.")

INPUT_SCHEMA = {
    'video_url': {
        'type': str,
        'required': False,
        'default': None,
        'description': 'URL of the input video to process'
    },
    'video_uuid': {
        'type': str,
        'required': False,
        'default': None,
        'description': 'Dream UUID to fetch original video from backend'
    },
    'video_path': {
        'type': str,
        'required': False,
        'default': None,
        'description': 'Local path to input video file'
    },
    'upscale_factor': {
        'type': int,
        'required': False,
        'default': 2,
        'constraints': lambda x: x in [1, 2, 4],
        'description': 'Upscaling factor (1x = skip, 2x, or 4x)'
    },
    'interpolation_factor': {
        'type': int,
        'required': False,
        'default': 2,
        'constraints': lambda x: x in [1, 2, 4, 8],
        'description': 'Frame interpolation factor (1x = skip, 2x, 4x, or 8x)'
    },
    'output_fps': {
        'type': int,
        'required': False,
        'default': 0,
        'constraints': lambda x: (x == 0) or (1 <= x <= 120),
        'description': 'Output video frame rate (0 = auto)'
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
    },
    'output_name': {
        'type': str,
        'required': False,
        'default': None,
        'description': 'Output filename (optional)'
    },
    'input_file_path': {
        'type': str,
        'required': False,
        'default': None,
        'description': 'Path to input file (optional)'
    },
    'custom_output_path': {
        'type': str,
        'required': False,
        'default': None,
        'description': 'Custom output path (optional)'
    },
    'runpod_id': {
        'type': str,
        'required': False,
        'default': None,
        'description': 'RunPod job ID (optional)'
    }
}

def download_input_video_using_sdk(video_url: str, temp_dir: Path) -> Path:
    from urllib.parse import urlparse

    parsed = urlparse(video_url)
    file_ext = Path(parsed.path).suffix or '.mp4'
    input_path = temp_dir / f"input{file_ext}"

    if not edream_client:
        raise RuntimeError("EDream client not initialized; cannot download via SDK")

    logger.info(f"Downloading via SDK from: {video_url}")
    success = edream_client.file_client.download_file(video_url, str(input_path))
    if not success:
        raise RuntimeError("SDK download failed")
    logger.info(f"SDK download completed: {input_path}")
    return input_path


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
    video_url = (params.get('video_url') or '').strip() if params.get('video_url') else None
    video_uuid = (params.get('video_uuid') or '').strip() if params.get('video_uuid') else None
    video_path = (params.get('video_path') or '').strip() if params.get('video_path') else None

    provided = [p for p in [video_url, video_uuid, video_path] if p]
    if len(provided) == 0:
        raise ValueError("Provide one of 'video_url', 'video_uuid', or 'video_path'")
    if len(provided) > 1:
        raise ValueError("Provide only one of 'video_url', 'video_uuid', or 'video_path'")

    if video_url:
        return download_input_video_using_sdk(video_url, temp_dir)

    if video_path:
        video_path_obj = Path(video_path)
        if not video_path_obj.exists():
            raise FileNotFoundError(f"Video file not found: {video_path_obj}")
        return video_path_obj

    if not edream_client:
        raise RuntimeError("EDream client not initialized; cannot use 'video_uuid'")

    dream = edream_client.get_dream(uuid=video_uuid)
    if not dream or not dream.get('original_video'):
        raise ValueError(f"Dream not found or missing original_video for uuid: {video_uuid}")

    original = dream['original_video']
    if not isinstance(original, str):
        raise ValueError("original_video is not a valid URL string")
    return download_input_video_using_sdk(original, temp_dir)


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    cleanup_manager = CleanupManager()

    try:
        _input = job.get("input") or {}
        validated = validate(_input, INPUT_SCHEMA)
        
        if validated.get('errors'):
            raise ValueError(f"Input validation failed: {validated['errors']}")
        
        params = validated['validated_input']
        logger.info(f"Processing video with parameters: {params}")
        
        temp_dir = Path(tempfile.mkdtemp(prefix='video_proc_'))
        cleanup_manager.add_directory(temp_dir)

        input_video_path = get_input_video_path(params, temp_dir)
        
        processor = VideoProcessorService(
            temp_dir=temp_dir,
            cleanup_manager=cleanup_manager
        )
        
        start_time = time.perf_counter()
        
        def progress_callback(percent, preview=None):
            elapsed_ms = int((time.perf_counter() - start_time) * 1000)
            countdown_ms = int((elapsed_ms / percent) * (100 - percent)) if percent > 0 else 0
            
            progress_data = {
                "progress": round(float(percent), 1),
                "countdown_ms": countdown_ms
            }
            
            if preview:
                progress_data["preview_frame"] = preview
                
            runpod.serverless.progress_update(job, progress_data)

        output_video_path = processor.process_video(
            input_path=input_video_path,
            upscale_factor=params['upscale_factor'],
            interpolation_factor=params['interpolation_factor'],
            output_fps=params['output_fps'],
            output_format=params['output_format'],
            tile_size=params['tile_size'],
            tile_padding=params['tile_padding'],
            quality=params['quality'],
            progress_callback=progress_callback
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
        # Raising an exception ensures the RunPod job is marked as FAILED
        raise
    
    finally:
        cleanup_manager.cleanup_all()


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
