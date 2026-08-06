from utils.upscale_config import (
    DEFAULT_UPSCALE_FACTOR,
    SKIP_UPSCALE_FACTOR,
    SUPPORTED_UPSCALE_FACTORS,
    supported_factors_text,
)

_UPSCALE_DESCRIPTION = (
    f"Upscaling factor ({SKIP_UPSCALE_FACTOR}x = skip; supported: "
    f"{supported_factors_text()})"
)

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
        'default': DEFAULT_UPSCALE_FACTOR,
        'constraints': lambda x: x in SUPPORTED_UPSCALE_FACTORS,
        'description': _UPSCALE_DESCRIPTION
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
