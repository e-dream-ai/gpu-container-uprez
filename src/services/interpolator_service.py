import logging
from pathlib import Path
from typing import List
import cv2
import numpy as np
import torch
from tqdm import tqdm
import tempfile
import subprocess

from services.model_loader import ModelLoader

logger = logging.getLogger(__name__)


class InterpolatorService:
    def __init__(self, model_loader: ModelLoader):
        self.model_loader = model_loader
        logger.info("InterpolatorService initialized")
    
    def interpolate_frames(
        self,
        input_frames: List[Path],
        output_dir: Path,
        interpolation_factor: int = 2
    ) -> None:
        if len(input_frames) < 2:
            logger.warning("Need at least 2 frames for interpolation")
            return
        
        logger.info(f"Interpolating {len(input_frames)} frames with factor {interpolation_factor}x using vsrife")
        
        # Load the RIFE function from vsrife
        rife_function = self.model_loader.load_rife_model()
        
        try:
            import time
            start_time = time.time()
            
            self._interpolate_with_vsrife(input_frames, output_dir, interpolation_factor, rife_function)
            
            total_time = time.time() - start_time
            logger.info(f"Interpolation completed in {total_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Frame interpolation failed: {e}")
            raise RuntimeError(f"Interpolation process failed: {e}")
    
    def _interpolate_with_vsrife(
        self,
        input_frames: List[Path],
        output_dir: Path,
        interpolation_factor: int,
        rife_function: Any
    ) -> None:
        """Use vsrife with VapourSynth to interpolate frames efficiently."""
        try:
            import vapoursynth as vs
            from vsrife import rife
            
            core = vs.core
            
            temp_video_path = self._create_temp_video_from_frames(input_frames)
            
            try:
                clip = core.ffms2.Source(str(temp_video_path))
                
                interpolated_clip = rife_function(
                    clip=clip,
                    model=4,
                    factor_num=interpolation_factor,
                    factor_den=1,
                    fps_num=None,
                    fps_den=1,
                    scene_thresh=0.15,
                    skip=True,
                    stat_th=60.0,
                    auto_download=True,
                    device_index=0 if torch.cuda.is_available() else -1
                )
                
                self._export_frames_from_clip(interpolated_clip, output_dir)
                
            finally:
                # Clean up temporary video
                if temp_video_path.exists():
                    temp_video_path.unlink()
                    
        except Exception as e:
            logger.error(f"vsrife interpolation failed: {e}")
            self._fallback_frame_copy(input_frames, output_dir, interpolation_factor)
    
    def _create_temp_video_from_frames(self, input_frames: List[Path]) -> Path:
        """Create a temporary video file from input frames for VapourSynth processing."""
        temp_video = Path(tempfile.mktemp(suffix='.mp4'))
        
        # Use ffmpeg to create video from frames
        frame_pattern = str(input_frames[0].parent / "frame_%06d.png")
        cmd = [
            'ffmpeg', '-y',
            '-framerate', '30',  # Input framerate
            '-i', frame_pattern,
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-crf', '18',  # High quality
            str(temp_video)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to create temp video: {result.stderr}")
        
        return temp_video
    
    def _export_frames_from_clip(self, clip, output_dir: Path) -> None:
        """Export frames from VapourSynth clip to output directory."""
        import vapoursynth as vs
        
        frame_count = clip.num_frames
        logger.info(f"Exporting {frame_count} interpolated frames")
        
        with tqdm(total=frame_count, desc="Exporting frames") as pbar:
            for i in range(frame_count):
                frame = clip.get_frame(i)
                frame_array = np.asarray(frame[0]) 
                
                if len(frame_array.shape) == 2: 
                    frame_bgr = cv2.cvtColor(frame_array, cv2.COLOR_GRAY2BGR)
                else:  # RGB
                    frame_bgr = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
                
                output_path = output_dir / f"frame_{i:06d}.png"
                cv2.imwrite(str(output_path), frame_bgr)
                pbar.update(1)
    
    def _fallback_frame_copy(
        self,
        input_frames: List[Path],
        output_dir: Path,
        interpolation_factor: int
    ) -> None:
        logger.warning("Using fallback frame copying (no actual interpolation)")
        
        output_frame_idx = 0
        for i, frame_path in enumerate(input_frames):
            self._copy_frame(frame_path, output_dir, output_frame_idx)
            output_frame_idx += 1
            
            if i < len(input_frames) - 1: 
                for _ in range(interpolation_factor - 1):
                    self._copy_frame(frame_path, output_dir, output_frame_idx)
                    output_frame_idx += 1
    
    def _copy_frame(self, source_path: Path, output_dir: Path, frame_idx: int) -> None:
        output_path = output_dir / f"frame_{frame_idx:06d}.png"
        
        frame = cv2.imread(str(source_path))
        if frame is not None:
            cv2.imwrite(str(output_path), frame)
    
    def _calculate_output_frame_count(self, input_count: int, interpolation_factor: int) -> int:
        if input_count < 2:
            return input_count
        
        intermediate_count = (input_count - 1) * (interpolation_factor - 1)
        return input_count + intermediate_count