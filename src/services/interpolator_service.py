import logging
from contextlib import nullcontext
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm

from services.model_loader import ModelLoader
from services.preview_encoder import PreviewService

logger = logging.getLogger(__name__)

FAST_PNG_WRITE_PARAMS = [cv2.IMWRITE_PNG_COMPRESSION, 0]


class InterpolatorService:
    def __init__(self, model_loader: ModelLoader, preview_service: PreviewService):
        self.model_loader = model_loader
        self.preview_service = preview_service
        self._target_size: Optional[Tuple[int, int]] = None
        self._output_size: Optional[Tuple[int, int]] = None
        self._use_fp16 = False
        logger.info("InterpolatorService initialized")

    def interpolate_frames(
        self,
        input_frames: List[Path],
        output_dir: Path,
        interpolation_factor: int = 2,
        progress_callback: Callable[[int, Optional[str]], None] = None,
        preview_max_side: Optional[int] = None,
        preview_jpeg_quality: Optional[int] = None,
    ) -> None:
        if len(input_frames) < 2:
            logger.warning("Need at least 2 frames for interpolation")
            return

        if interpolation_factor == 1:
            logger.info(
                "Interpolation factor is 1x; skipping interpolation and copying frames"
            )
            for idx, frame_path in enumerate(input_frames):
                try:
                    img = cv2.imread(str(frame_path))
                    if img is None:
                        logger.warning(f"Could not read frame: {frame_path}")
                        continue

                    output_path = output_dir / f"frame_{idx:06d}.png"
                    self._write_frame(output_path, img)

                    if progress_callback:
                        preview_base64 = (
                            self.preview_service.encode_bgr_to_base64_jpeg(
                                img,
                                max_side=preview_max_side,
                                jpeg_quality=preview_jpeg_quality,
                            )
                        )
                        progress_callback(
                            int(((idx + 1) / len(input_frames) * 100)),
                            preview_base64,
                        )
                except Exception as e:
                    logger.error(f"Failed to copy frame {frame_path}: {e}")
            return

        logger.info(
            f"Interpolating {len(input_frames)} frames with factor {interpolation_factor}x"
        )

        logger.info("Loading RIFE model...")
        rife_model = self.model_loader.load_rife_model()
        self._use_fp16 = bool(getattr(rife_model, 'use_fp16', False))
        logger.info("RIFE model loaded successfully")

        try:
            import time

            start_time = time.time()

            self._target_size = self._determine_safe_size(input_frames)
            if self._target_size is None:
                raise RuntimeError(
                    "Failed to determine a safe target size for interpolation"
                )
            logger.info(
                f"Using target frame size (HxW): {self._target_size[0]}x{self._target_size[1]}"
            )

            try:
                first_img = cv2.imread(str(input_frames[0]))
                if first_img is not None:
                    h0, w0 = first_img.shape[:2]
                    self._output_size = (h0, w0)
                    logger.info(f"Using output frame size (HxW): {h0}x{w0}")
            except Exception:
                self._output_size = None

            desired_total_frames = len(input_frames) * max(1, interpolation_factor)

            with tqdm(total=desired_total_frames, desc="Interpolating frames") as pbar:
                output_frame_idx = 0

                for i in range(len(input_frames) - 1):
                    current_frame = input_frames[i]
                    next_frame = input_frames[i + 1]

                    logger.debug(f"Processing frame pair {i}/{len(input_frames)-1}")

                    self._copy_frame(current_frame, output_dir, output_frame_idx)
                    output_frame_idx += 1
                    pbar.update(1)

                    try:
                        intermediate_frames = self._generate_intermediate_frames(
                            current_frame,
                            next_frame,
                            rife_model,
                            interpolation_factor,
                        )
                        logger.debug(
                            f"Generated {len(intermediate_frames)} intermediate frames for pair {i}"
                        )
                    except Exception as e:
                        logger.error(
                            "Failed to generate intermediate frames for pair "
                            f"{i} ({current_frame} -> {next_frame}): {e}"
                        )
                        raise RuntimeError(
                            f"Interpolation failed at frame pair {i}: {e}"
                        )

                    last_intermediate = None
                    for intermediate_frame in intermediate_frames:
                        output_path = output_dir / f"frame_{output_frame_idx:06d}.png"
                        success = self._write_frame(output_path, intermediate_frame)
                        if not success:
                            raise RuntimeError(
                                f"Failed to write interpolated frame to {output_path}"
                            )
                        last_intermediate = intermediate_frame
                        output_frame_idx += 1
                        pbar.update(1)

                    if progress_callback:
                        preview_base64 = (
                            self.preview_service.encode_bgr_to_base64_jpeg(
                                last_intermediate,
                                max_side=preview_max_side,
                                jpeg_quality=preview_jpeg_quality,
                            )
                        )
                        progress_callback(
                            int((output_frame_idx / desired_total_frames * 100)),
                            preview_base64,
                        )

                self._copy_frame(input_frames[-1], output_dir, output_frame_idx)
                output_frame_idx += 1
                pbar.update(1)

                while output_frame_idx < desired_total_frames:
                    output_path = output_dir / f"frame_{output_frame_idx:06d}.png"
                    last_frame = cv2.imread(str(input_frames[-1]))
                    if last_frame is not None and self._output_size is not None:
                        out_h, out_w = self._output_size
                        last_frame = self._resize_to_exact(last_frame, out_h, out_w)
                    if last_frame is not None:
                        self._write_frame(output_path, last_frame)
                    output_frame_idx += 1
                    pbar.update(1)

                if progress_callback:
                    progress_callback(100, None)

            total_time = time.time() - start_time
            logger.info(f"Interpolation completed in {total_time:.2f}s")
            logger.info(
                f"Generated {desired_total_frames} frames from {len(input_frames)} input frames"
            )

            actual_frames = list(output_dir.glob("frame_*.png"))
            if len(actual_frames) == 0:
                raise RuntimeError(
                    f"No interpolated frames were generated! Expected {desired_total_frames} frames"
                )
            if len(actual_frames) < desired_total_frames:
                logger.warning(
                    f"Generated fewer frames than expected: {len(actual_frames)} vs {desired_total_frames}"
                )

        except Exception as e:
            logger.error(f"Frame interpolation failed: {e}")
            raise RuntimeError(f"Interpolation process failed: {e}")

    def _generate_intermediate_frames(
        self,
        frame1_path: Path,
        frame2_path: Path,
        rife_model,
        interpolation_factor: int,
    ) -> List[np.ndarray]:
        frame1 = cv2.imread(str(frame1_path))
        frame2 = cv2.imread(str(frame2_path))

        if frame1 is None or frame2 is None:
            error_msg = f"Could not load frames: {frame1_path}, {frame2_path}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        if self._target_size is None:
            raise RuntimeError(
                "Target size not initialized before generating intermediate frames"
            )
        target_h, target_w = self._target_size
        frame1 = self._fit_to_size(frame1, target_h, target_w)
        frame2 = self._fit_to_size(frame2, target_h, target_w)

        frame1_tensor = self._frame_to_tensor(frame1)
        frame2_tensor = self._frame_to_tensor(frame2)

        intermediate_frames: List[np.ndarray] = []

        if interpolation_factor == 2:
            mid_frame = self._interpolate_single_frame(
                frame1_tensor, frame2_tensor, rife_model, 0.5
            )
            if self._output_size is not None:
                out_h, out_w = self._output_size
                mid_frame = self._resize_to_exact(mid_frame, out_h, out_w)
            intermediate_frames.append(mid_frame)

        elif interpolation_factor == 4:
            for t in [0.25, 0.5, 0.75]:
                mid_frame = self._interpolate_single_frame(
                    frame1_tensor, frame2_tensor, rife_model, t
                )
                if self._output_size is not None:
                    out_h, out_w = self._output_size
                    mid_frame = self._resize_to_exact(mid_frame, out_h, out_w)
                intermediate_frames.append(mid_frame)

        elif interpolation_factor == 8:
            for t in [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875]:
                mid_frame = self._interpolate_single_frame(
                    frame1_tensor, frame2_tensor, rife_model, t
                )
                if self._output_size is not None:
                    out_h, out_w = self._output_size
                    mid_frame = self._resize_to_exact(mid_frame, out_h, out_w)
                intermediate_frames.append(mid_frame)

        return intermediate_frames

    def _interpolate_single_frame(
        self,
        frame1_tensor: torch.Tensor,
        frame2_tensor: torch.Tensor,
        rife_model,
        timestep: float,
    ) -> np.ndarray:
        autocast_context = (
            torch.autocast(device_type='cuda', dtype=torch.float16)
            if self._use_fp16 and frame1_tensor.device.type == 'cuda'
            else nullcontext()
        )

        with torch.inference_mode(), autocast_context:
            timestep_tensor = torch.tensor(
                [timestep],
                device=frame1_tensor.device,
                dtype=frame1_tensor.dtype,
            )
            mid_frame_tensor = rife_model.inference(
                frame1_tensor, frame2_tensor, timestep_tensor
            )
            mid_frame = self._tensor_to_frame(mid_frame_tensor)
            return mid_frame

    def _frame_to_tensor(self, frame: np.ndarray) -> torch.Tensor:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = torch.from_numpy(frame_rgb)
        frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0)

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        dtype = torch.float16 if self._use_fp16 and device.type == 'cuda' else torch.float32
        return frame_tensor.to(device=device, dtype=dtype).div_(255.0)

    def _tensor_to_frame(self, tensor: torch.Tensor) -> np.ndarray:
        frame_tensor = tensor.squeeze(0).detach().float().clamp_(0.0, 1.0).cpu()
        frame_tensor = frame_tensor.permute(1, 2, 0)
        frame_np = (frame_tensor.numpy() * 255.0).astype(np.uint8)
        frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
        return frame_bgr

    def _copy_frame(self, source_path: Path, output_dir: Path, frame_idx: int) -> None:
        output_path = output_dir / f"frame_{frame_idx:06d}.png"
        frame = cv2.imread(str(source_path))
        if frame is not None:
            if self._output_size is not None:
                out_h, out_w = self._output_size
                frame = self._resize_to_exact(frame, out_h, out_w)
            self._write_frame(output_path, frame)

    def _write_frame(self, output_path: Path, frame: np.ndarray) -> bool:
        if output_path.suffix.lower() == '.png':
            return cv2.imwrite(str(output_path), frame, FAST_PNG_WRITE_PARAMS)
        return cv2.imwrite(str(output_path), frame)

    def _calculate_output_frame_count(self, input_count: int, interpolation_factor: int) -> int:
        if input_count < 2:
            return input_count
        intermediate_count = (input_count - 1) * (interpolation_factor - 1)
        return input_count + intermediate_count

    def _determine_safe_size(self, frames: List[Path]) -> Optional[Tuple[int, int]]:
        max_h: Optional[int] = None
        max_w: Optional[int] = None
        for frame_path in frames:
            img = cv2.imread(str(frame_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            if max_h is None or h > max_h:
                max_h = h
            if max_w is None or w > max_w:
                max_w = w
        if max_h is None or max_w is None:
            return None

        def ceil_to_multiple(value: int, multiple: int) -> int:
            return ((value + multiple - 1) // multiple) * multiple

        safe_h = max(64, ceil_to_multiple(max_h, 64))
        safe_w = max(64, ceil_to_multiple(max_w, 64))
        return (safe_h, safe_w)

    def _fit_to_size(self, img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
        h, w = img.shape[:2]
        if h == target_h and w == target_w:
            return img

        top = 0
        left = 0
        bottom = h
        right = w

        if h > target_h:
            crop_top = (h - target_h) // 2
            top = crop_top
            bottom = crop_top + target_h

        if w > target_w:
            crop_left = (w - target_w) // 2
            left = crop_left
            right = crop_left + target_w

        cropped = img[top:bottom, left:right]
        ch, cw = cropped.shape[:2]

        pad_top = max(0, (target_h - ch) // 2)
        pad_bottom = max(0, target_h - ch - pad_top)
        pad_left = max(0, (target_w - cw) // 2)
        pad_right = max(0, target_w - cw - pad_left)

        if pad_top or pad_bottom or pad_left or pad_right:
            cropped = cv2.copyMakeBorder(
                cropped,
                pad_top,
                pad_bottom,
                pad_left,
                pad_right,
                borderType=cv2.BORDER_REPLICATE,
            )
        return cropped

    def _resize_to_exact(self, img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
        h, w = img.shape[:2]
        if h == target_h and w == target_w:
            return img

        top = max(0, (h - target_h) // 2)
        left = max(0, (w - target_w) // 2)
        bottom = min(h, top + target_h)
        right = min(w, left + target_w)
        cropped = img[top:bottom, left:right]
        ch, cw = cropped.shape[:2]

        pad_top = max(0, (target_h - ch) // 2)
        pad_bottom = max(0, target_h - ch - pad_top)
        pad_left = max(0, (target_w - cw) // 2)
        pad_right = max(0, target_w - cw - pad_left)

        if pad_top or pad_bottom or pad_left or pad_right:
            cropped = cv2.copyMakeBorder(
                cropped,
                pad_top,
                pad_bottom,
                pad_left,
                pad_right,
                borderType=cv2.BORDER_REPLICATE,
            )
        return cropped