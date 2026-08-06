import logging
import os
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from services.model_loader import ModelLoader
from services.frame_manager import FrameManager
from services.preview_encoder import PreviewService
from services.upscaler_service import UpscalerService
from services.interpolator_service import InterpolatorService
from utils.cleanup_manager import CleanupManager
from utils.input_validator import InputValidator
from utils.upscale_config import DEFAULT_UPSCALE_FACTOR

logger = logging.getLogger(__name__)

DISK_HEADROOM_BYTES = 2 * (1024 ** 3)


class VideoProcessorService:
    
    def __init__(self, temp_dir: Path, cleanup_manager: CleanupManager):
        self.temp_dir = temp_dir
        self.cleanup_manager = cleanup_manager
        self.model_loader = ModelLoader()
        self.preview_service = PreviewService()
        self.frame_manager = FrameManager(temp_dir, cleanup_manager)
        self.upscaler = UpscalerService(self.model_loader, self.preview_service)
        self.interpolator = InterpolatorService(self.model_loader, self.preview_service)
        
        logger.info("VideoProcessorService initialized")

    def _check_capacity(
        self,
        video_info: Dict[str, Any],
        upscale_factor: int,
        interpolation_factor: int,
        output_format: str,
    ) -> None:
        if not video_info.get('width') or not video_info.get('height'):
            logger.warning(
                "Capacity checks skipped: could not determine source dimensions "
                f"from {video_info}. Encoder and disk limits are unverified."
            )
            return

        check = InputValidator.validate_processing_parameters(
            video_info=video_info,
            upscale_factor=upscale_factor,
            interpolation_factor=interpolation_factor,
            output_format=output_format,
        )

        for warning in check['warnings']:
            logger.warning(f"Capacity: {warning}")

        if not check['valid']:
            raise ValueError("; ".join(check['errors']))

        required = check['estimates'].get('estimated_frame_disk_bytes', 0)
        if not required:
            return

        free = shutil.disk_usage(self.temp_dir).free
        required_gb = required / (1024 ** 3)
        free_gb = free / (1024 ** 3)
        logger.info(
            f"Frame storage estimate: {required_gb:.1f}GB needed, "
            f"{free_gb:.1f}GB free in {self.temp_dir}"
        )

        if required + DISK_HEADROOM_BYTES > free:
            raise RuntimeError(
                f"Insufficient disk space for intermediate frames: "
                f"~{required_gb:.1f}GB needed, {free_gb:.1f}GB free. "
                f"Reduce upscale_factor ({upscale_factor}x), "
                f"interpolation_factor ({interpolation_factor}x), or clip length."
            )

    def process_video(
        self,
        input_path: Path,
        upscale_factor: int = DEFAULT_UPSCALE_FACTOR,
        interpolation_factor: int = 2,
        output_fps: int = 0,
        output_format: str = 'mp4',
        tile_size: int = 1024,
        tile_padding: int = 10,
        quality: str = 'high',
        progress_callback: Callable[[int, Optional[str]], None] = None
    ) -> Path:
        def update_progress(percent, preview=None):
            if progress_callback:
                progress_callback(percent, preview)

        logger.info("Starting video processing pipeline")
        logger.info(f"Input: {input_path}")
        logger.info(f"Upscale: {upscale_factor}x, Interpolation: {interpolation_factor}x")
        
        try:
            update_progress(5.0)
            logger.info("Step 1: Extracting frames from input video")
            original_frames_dir = self.frame_manager.extract_frames(input_path)
            original_frame_paths = self.frame_manager.get_frame_paths(original_frames_dir)
            
            update_progress(15.0)
            logger.info(f"Extracted {len(original_frame_paths)} frames")

            video_info = self.frame_manager.get_video_info(input_path)
            video_info['frame_count'] = len(original_frame_paths)
            self._check_capacity(
                video_info=video_info,
                upscale_factor=upscale_factor,
                interpolation_factor=interpolation_factor,
                output_format=output_format,
            )

            logger.info("Step 2: Upscaling frames with Real-ESRGAN")
            upscaled_frames_dir = self.temp_dir / "frames_upscaled"
            upscaled_frames_dir.mkdir(exist_ok=True)
            self.cleanup_manager.add_directory(upscaled_frames_dir)
            
            # Upscaling: Map 0-100% to 15-50%
            def upscale_progress(p, preview=None):
                update_progress(15.0 + (p * 0.35), preview)

            frame_cache = self.upscaler.upscale_frames(
                input_frames=original_frame_paths,
                output_dir=upscaled_frames_dir,
                upscale_factor=upscale_factor,
                tile_size=tile_size,
                tile_padding=tile_padding,
                progress_callback=upscale_progress
            )
            update_progress(50.0)
            
            upscaled_frame_paths = self.frame_manager.get_frame_paths(upscaled_frames_dir)
            logger.info(f"Upscaled {len(upscaled_frame_paths)} frames")
            
            logger.info("Step 3: Interpolating frames with RIFE")
            interpolated_frames_dir = self.temp_dir / "frames_interpolated"
            interpolated_frames_dir.mkdir(exist_ok=True)
            self.cleanup_manager.add_directory(interpolated_frames_dir)
            
            def interpolation_progress(p, preview=None):
                update_progress(50.0 + (p * 0.40), preview)
            
            self.interpolator.interpolate_frames(
                input_frames=upscaled_frame_paths,
                output_dir=interpolated_frames_dir,
                interpolation_factor=interpolation_factor,
                frame_cache=frame_cache,
                progress_callback=interpolation_progress
            )
            frame_cache = None
            update_progress(90.0)
            
            interpolated_frame_paths = self.frame_manager.get_frame_paths(interpolated_frames_dir)
            logger.info(f"Generated {len(interpolated_frame_paths)} interpolated frames")
            
            logger.info("Step 4: Encoding final video")
            output_path = self.temp_dir / f"output.{output_format}"
            
            video_fps = video_info.get('fps', 30)
            source_fps = int(round(video_fps)) if isinstance(video_fps, (int, float)) else 30
            computed_fps = max(1, int(round(source_fps * max(1, interpolation_factor))))
            final_fps = computed_fps if not output_fps or output_fps <= 0 else output_fps
            logger.info(f"Encoding at {final_fps} fps (source {source_fps} x interp {interpolation_factor})")

            # Encoding: Map 0-100% to 90-100%
            def encoding_progress(p):
                update_progress(90.0 + (p * 0.10))

            self.frame_manager.encode_video(
                frame_dir=interpolated_frames_dir,
                output_path=output_path,
                fps=final_fps,
                format=output_format,
                quality=quality,
                progress_callback=encoding_progress
            )
            update_progress(100.0)

            logger.info(f"Video processing completed: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Video processing failed: {e}", exc_info=True)
            raise RuntimeError(f"Video processing pipeline failed: {e}")
