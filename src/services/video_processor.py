import logging
from pathlib import Path
from typing import Callable, Optional
from services.model_loader import ModelLoader
from services.frame_manager import FrameManager
from services.upscaler_service import UpscalerService
from services.interpolator_service import InterpolatorService
from utils.cleanup_manager import CleanupManager

logger = logging.getLogger(__name__)


class VideoProcessorService:
    
    def __init__(self, temp_dir: Path, cleanup_manager: CleanupManager):
        self.temp_dir = temp_dir
        self.cleanup_manager = cleanup_manager
        
        self.model_loader = ModelLoader()
        self.frame_manager = FrameManager(temp_dir, cleanup_manager)
        self.upscaler = UpscalerService(self.model_loader)
        self.interpolator = InterpolatorService(self.model_loader)
        
        logger.info("VideoProcessorService initialized")
    
    def process_video(
        self,
        input_path: Path,
        upscale_factor: int = 2,
        interpolation_factor: int = 2,
        output_fps: int = 0,
        output_format: str = 'mp4',
        tile_size: int = 512,
        tile_padding: int = 10,
        quality: str = 'high',
        progress_callback: Callable[[int, Optional[str]], None] = None
    ) -> Path:
        def update_progress(percent, preview=None):
            if progress_callback:
                progress_callback(percent, preview)

        logger.info(f"Starting video processing pipeline")
        logger.info(f"Input: {input_path}")
        logger.info(f"Upscale: {upscale_factor}x, Interpolation: {interpolation_factor}x")
        
        try:
            update_progress(5.0)
            logger.info("Step 1: Extracting frames from input video")
            original_frames_dir = self.frame_manager.extract_frames(input_path)
            original_frame_paths = self.frame_manager.get_frame_paths(original_frames_dir)
            
            update_progress(15.0)
            logger.info(f"Extracted {len(original_frame_paths)} frames")
            
            logger.info("Step 2: Upscaling frames with Real-ESRGAN")
            upscaled_frames_dir = self.temp_dir / "frames_upscaled"
            upscaled_frames_dir.mkdir(exist_ok=True)
            self.cleanup_manager.add_directory(upscaled_frames_dir)
            
            # Upscaling: Map 0-100% to 15-50%
            def upscale_progress(p, preview=None):
                update_progress(15.0 + (p * 0.35), preview)

            self.upscaler.upscale_frames(
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
                progress_callback=interpolation_progress
            )
            update_progress(90.0)
            
            interpolated_frame_paths = self.frame_manager.get_frame_paths(interpolated_frames_dir)
            logger.info(f"Generated {len(interpolated_frame_paths)} interpolated frames")
            
            logger.info("Step 4: Encoding final video")
            output_path = self.temp_dir / f"output.{output_format}"
            
            video_info = self.frame_manager.get_video_info(input_path)
            source_fps = int(round(video_info.get('fps', 30))) if isinstance(video_info.get('fps', None), (int, float)) else 30
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