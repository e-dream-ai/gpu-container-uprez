import logging
from pathlib import Path
from typing import List
import cv2
import numpy as np
import torch
from tqdm import tqdm

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
        
        logger.info(f"Interpolating {len(input_frames)} frames with factor {interpolation_factor}x")
        
        rife_model = self.model_loader.load_rife_model()
        
        try:
            import time
            start_time = time.time()
            
            total_output_frames = self._calculate_output_frame_count(
                len(input_frames), interpolation_factor
            )
            
            with tqdm(total=total_output_frames, desc="Interpolating frames") as pbar:
                output_frame_idx = 0
                
                for i in range(len(input_frames) - 1):
                    current_frame = input_frames[i]
                    next_frame = input_frames[i + 1]
                    
                    self._copy_frame(current_frame, output_dir, output_frame_idx)
                    output_frame_idx += 1
                    pbar.update(1)
                    
                    intermediate_frames = self._generate_intermediate_frames(
                        current_frame, next_frame, rife_model, interpolation_factor
                    )
                    
                    for intermediate_frame in intermediate_frames:
                        output_path = output_dir / f"frame_{output_frame_idx:06d}.png"
                        cv2.imwrite(str(output_path), intermediate_frame)
                        output_frame_idx += 1
                        pbar.update(1)
                
                self._copy_frame(input_frames[-1], output_dir, output_frame_idx)
                pbar.update(1)
            
            total_time = time.time() - start_time
            logger.info(f"Interpolation completed in {total_time:.2f}s")
            logger.info(f"Generated {total_output_frames} frames from {len(input_frames)} input frames")
            
        except Exception as e:
            logger.error(f"Frame interpolation failed: {e}")
            raise RuntimeError(f"Interpolation process failed: {e}")
    
    def _generate_intermediate_frames(
        self,
        frame1_path: Path,
        frame2_path: Path,
        rife_model,
        interpolation_factor: int
    ) -> List[np.ndarray]:

        try:
            frame1 = cv2.imread(str(frame1_path))
            frame2 = cv2.imread(str(frame2_path))
            
            if frame1 is None or frame2 is None:
                logger.error(f"Could not load frames: {frame1_path}, {frame2_path}")
                return []
            
            frame1_tensor = self._frame_to_tensor(frame1)
            frame2_tensor = self._frame_to_tensor(frame2)
            
            intermediate_frames = []
            
            if interpolation_factor == 2:
                mid_frame = self._interpolate_single_frame(
                    frame1_tensor, frame2_tensor, rife_model, 0.5
                )
                intermediate_frames.append(mid_frame)
                
            elif interpolation_factor == 4:
                for t in [0.25, 0.5, 0.75]:
                    mid_frame = self._interpolate_single_frame(
                        frame1_tensor, frame2_tensor, rife_model, t
                    )
                    intermediate_frames.append(mid_frame)
                    
            elif interpolation_factor == 8:
                for t in [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875]:
                    mid_frame = self._interpolate_single_frame(
                        frame1_tensor, frame2_tensor, rife_model, t
                    )
                    intermediate_frames.append(mid_frame)
            
            return intermediate_frames
            
        except Exception as e:
            logger.error(f"Failed to generate intermediate frames: {e}")
            return []
    
    def _interpolate_single_frame(
        self,
        frame1_tensor: torch.Tensor,
        frame2_tensor: torch.Tensor,
        rife_model,
        timestep: float
    ) -> np.ndarray:
        with torch.no_grad():
            timestep_tensor = torch.tensor([timestep]).to(frame1_tensor.device).float()
            
            mid_frame_tensor = rife_model.inference(
                frame1_tensor, frame2_tensor, timestep_tensor
            )
            
            mid_frame = self._tensor_to_frame(mid_frame_tensor)
            
            return mid_frame
    
    def _frame_to_tensor(self, frame: np.ndarray) -> torch.Tensor:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        frame_tensor = torch.from_numpy(frame_rgb.astype(np.float32) / 255.0)
        
        frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0)
        
        if torch.cuda.is_available():
            frame_tensor = frame_tensor.cuda()
        
        return frame_tensor
    
    def _tensor_to_frame(self, tensor: torch.Tensor) -> np.ndarray:
       
        frame_tensor = tensor.squeeze(0).cpu()
        
        frame_tensor = frame_tensor.permute(1, 2, 0)
        
        frame_np = (frame_tensor.numpy() * 255.0).astype(np.uint8)
        
        frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
        
        return frame_bgr
    
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