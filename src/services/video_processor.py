import logging
from pathlib import Path
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
        output_fps: int = 30,
        output_format: str = 'mp4',
        tile_size: int = 512,
        tile_padding: int = 10,
        quality: str = 'high'
    ) -> Path:
        logger.info(f"Starting video processing pipeline")
        logger.info(f"Input: {input_path}")
        logger.info(f"Upscale: {upscale_factor}x, Interpolation: {interpolation_factor}x")
        
        try:
            logger.info("Step 1: Extracting frames from input video")
            extraction_fps = max(1.0, float(output_fps) / float(max(1, interpolation_factor)))
            original_frames_dir = self.frame_manager.extract_frames(input_path, fps=extraction_fps)
            original_frame_paths = self.frame_manager.get_frame_paths(original_frames_dir)
            
            logger.info(f"Extracted {len(original_frame_paths)} frames")
            
            logger.info("Step 2: Upscaling frames with Real-ESRGAN")
            upscaled_frames_dir = self.temp_dir / "frames_upscaled"
            upscaled_frames_dir.mkdir(exist_ok=True)
            self.cleanup_manager.add_directory(upscaled_frames_dir)
            
            self.upscaler.upscale_frames(
                input_frames=original_frame_paths,
                output_dir=upscaled_frames_dir,
                upscale_factor=upscale_factor,
                tile_size=tile_size,
                tile_padding=tile_padding
            )
            
            upscaled_frame_paths = self.frame_manager.get_frame_paths(upscaled_frames_dir)
            logger.info(f"Upscaled {len(upscaled_frame_paths)} frames")
            
            logger.info("Step 3: Interpolating frames with RIFE")
            interpolated_frames_dir = self.temp_dir / "frames_interpolated"
            interpolated_frames_dir.mkdir(exist_ok=True)
            self.cleanup_manager.add_directory(interpolated_frames_dir)
            
            self.interpolator.interpolate_frames(
                input_frames=upscaled_frame_paths,
                output_dir=interpolated_frames_dir,
                interpolation_factor=interpolation_factor
            )
            
            interpolated_frame_paths = self.frame_manager.get_frame_paths(interpolated_frames_dir)
            logger.info(f"Generated {len(interpolated_frame_paths)} interpolated frames")
            
            logger.info("Step 4: Encoding final video")
            output_path = self.temp_dir / f"output.{output_format}"
            
            self.frame_manager.encode_video(
                frame_dir=interpolated_frames_dir,
                output_path=output_path,
                fps=output_fps,
                format=output_format,
                quality=quality
            )
            
            logger.info(f"Video processing completed: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Video processing failed: {e}", exc_info=True)
            raise RuntimeError(f"Video processing pipeline failed: {e}")