import logging
import os
import time
from pathlib import Path
from typing import Callable, Dict, Optional

import torch

from services.model_loader import ModelLoader
from services.frame_manager import FrameManager
from services.preview_encoder import PreviewService
from services.upscaler_service import UpscalerService
from services.interpolator_service import InterpolatorService
from utils.cleanup_manager import CleanupManager

logger = logging.getLogger(__name__)

FALSE_ENV_VALUES = {'0', 'false', 'no'}


class VideoProcessorService:
    
    def __init__(self, temp_dir: Path, cleanup_manager: CleanupManager):
        self.temp_dir = temp_dir
        self.cleanup_manager = cleanup_manager
        self.last_benchmark: Dict[str, object] = {}
        
        self.model_loader = ModelLoader()
        self.preview_service = PreviewService()
        self.frame_manager = FrameManager(temp_dir, cleanup_manager)
        self.upscaler = UpscalerService(self.model_loader, self.preview_service)
        self.interpolator = InterpolatorService(self.model_loader, self.preview_service)
        
        logger.info("VideoProcessorService initialized")

    def _finish_stage(
        self,
        stage_timings: Dict[str, float],
        stage_name: str,
        start_time: float,
        **details,
    ) -> None:
        elapsed_s = time.perf_counter() - start_time
        stage_timings[stage_name] = elapsed_s
        detail_text = " ".join(f"{key}={value}" for key, value in details.items())
        logger.info(f"BENCHMARK stage={stage_name} elapsed_s={elapsed_s:.3f} {detail_text}".rstrip())

    def _runtime_info(self) -> Dict[str, object]:
        rife_model = self.model_loader.loaded_models.get("rife")
        info: Dict[str, object] = {
            "device": str(self.model_loader.device),
            "cuda_available": torch.cuda.is_available(),
            "decode_mode": self.frame_manager.last_decode_mode,
            "encode_codec": self.frame_manager.last_encode_codec,
            "rife_fp16": bool(getattr(rife_model, "use_fp16", False)),
            "use_nvenc": os.getenv("USE_NVENC", "1").lower() not in FALSE_ENV_VALUES,
            "use_hwaccel_decode": os.getenv("USE_HWACCEL_DECODE", "1").lower() not in FALSE_ENV_VALUES,
            "torch_compile": os.getenv("TORCH_COMPILE", "1").lower() not in FALSE_ENV_VALUES,
        }

        if torch.cuda.is_available():
            device_index = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(device_index)
            info.update(
                {
                    "gpu_name": torch.cuda.get_device_name(device_index),
                    "gpu_total_memory_gb": round(props.total_memory / (1024 ** 3), 2),
                    "gpu_max_memory_allocated_gb": round(
                        torch.cuda.max_memory_allocated(device_index) / (1024 ** 3),
                        2,
                    ),
                    "gpu_max_memory_reserved_gb": round(
                        torch.cuda.max_memory_reserved(device_index) / (1024 ** 3),
                        2,
                    ),
                }
            )

        return info
    
    def process_video(
        self,
        input_path: Path,
        upscale_factor: int = 2,
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

        pipeline_start = time.perf_counter()
        stage_timings: Dict[str, float] = {}
        self.last_benchmark = {}
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        logger.info("Starting video processing pipeline")
        logger.info(f"Input: {input_path}")
        logger.info(f"Upscale: {upscale_factor}x, Interpolation: {interpolation_factor}x")
        
        try:
            update_progress(5.0)
            logger.info("Step 1: Extracting frames from input video")
            stage_start = time.perf_counter()
            original_frames_dir = self.frame_manager.extract_frames(input_path)
            original_frame_paths = self.frame_manager.get_frame_paths(original_frames_dir)
            self._finish_stage(
                stage_timings,
                "extract_frames",
                stage_start,
                frames=len(original_frame_paths),
                decode_mode=self.frame_manager.last_decode_mode,
            )
            
            update_progress(15.0)
            logger.info(f"Extracted {len(original_frame_paths)} frames")
            
            logger.info("Step 2: Upscaling frames with Real-ESRGAN")
            upscaled_frames_dir = self.temp_dir / "frames_upscaled"
            upscaled_frames_dir.mkdir(exist_ok=True)
            self.cleanup_manager.add_directory(upscaled_frames_dir)
            
            # Upscaling: Map 0-100% to 15-50%
            def upscale_progress(p, preview=None):
                update_progress(15.0 + (p * 0.35), preview)

            stage_start = time.perf_counter()
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
            self._finish_stage(
                stage_timings,
                "upscale",
                stage_start,
                input_frames=len(original_frame_paths),
                output_frames=len(upscaled_frame_paths),
                tile_size=tile_size,
                tile_padding=tile_padding,
                upscale_factor=upscale_factor,
            )
            logger.info(f"Upscaled {len(upscaled_frame_paths)} frames")
            
            logger.info("Step 3: Interpolating frames with RIFE")
            interpolated_frames_dir = self.temp_dir / "frames_interpolated"
            interpolated_frames_dir.mkdir(exist_ok=True)
            self.cleanup_manager.add_directory(interpolated_frames_dir)
            
            def interpolation_progress(p, preview=None):
                update_progress(50.0 + (p * 0.40), preview)
            
            stage_start = time.perf_counter()
            self.interpolator.interpolate_frames(
                input_frames=upscaled_frame_paths,
                output_dir=interpolated_frames_dir,
                interpolation_factor=interpolation_factor,
                progress_callback=interpolation_progress
            )
            update_progress(90.0)
            
            interpolated_frame_paths = self.frame_manager.get_frame_paths(interpolated_frames_dir)
            self._finish_stage(
                stage_timings,
                "interpolate",
                stage_start,
                input_frames=len(upscaled_frame_paths),
                output_frames=len(interpolated_frame_paths),
                interpolation_factor=interpolation_factor,
            )
            logger.info(f"Generated {len(interpolated_frame_paths)} interpolated frames")
            
            logger.info("Step 4: Encoding final video")
            stage_start = time.perf_counter()
            output_path = self.temp_dir / f"output.{output_format}"
            
            video_info = self.frame_manager.get_video_info(input_path)
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
            self._finish_stage(
                stage_timings,
                "encode",
                stage_start,
                frames=len(interpolated_frame_paths),
                fps=final_fps,
                codec=self.frame_manager.last_encode_codec,
                quality=quality,
            )
            update_progress(100.0)

            total_elapsed_s = time.perf_counter() - pipeline_start
            output_size_mb = round(output_path.stat().st_size / (1024 ** 2), 2)
            self.last_benchmark = {
                "total_processing_s": round(total_elapsed_s, 3),
                "stages_s": {key: round(value, 3) for key, value in stage_timings.items()},
                "input": {
                    "path": str(input_path),
                    "width": video_info.get("width"),
                    "height": video_info.get("height"),
                    "fps": video_info.get("fps"),
                    "duration_s": video_info.get("duration"),
                    "codec": video_info.get("codec"),
                    "pix_fmt": video_info.get("pix_fmt"),
                    "frames": len(original_frame_paths),
                },
                "output": {
                    "path": str(output_path),
                    "format": output_format,
                    "fps": final_fps,
                    "frames": len(interpolated_frame_paths),
                    "size_mb": output_size_mb,
                },
                "settings": {
                    "upscale_factor": upscale_factor,
                    "interpolation_factor": interpolation_factor,
                    "tile_size": tile_size,
                    "tile_padding": tile_padding,
                    "quality": quality,
                },
                "runtime": self._runtime_info(),
            }
            logger.info(f"BENCHMARK summary={self.last_benchmark}")
            logger.info(f"Video processing completed: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Video processing failed: {e}", exc_info=True)
            raise RuntimeError(f"Video processing pipeline failed: {e}")
