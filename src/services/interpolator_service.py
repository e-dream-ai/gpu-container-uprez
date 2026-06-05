import logging
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm

from services.model_loader import ModelLoader
from services.preview_encoder import PreviewService

logger = logging.getLogger(__name__)

FAST_PNG_WRITE_PARAMS = [cv2.IMWRITE_PNG_COMPRESSION, 0]

DEFAULT_RIFE_BATCH_SIZE = max(1, int(os.getenv('RIFE_BATCH_SIZE', '3')))

TIMESTEPS = {
    2: [0.5],
    4: [0.25, 0.5, 0.75],
    8: [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875],
}


class InterpolatorService:
    def __init__(self, model_loader: ModelLoader, preview_service: PreviewService):
        self.model_loader = model_loader
        self.preview_service = preview_service
        self._target_size: Optional[Tuple[int, int]] = None
        self._output_size: Optional[Tuple[int, int]] = None
        self._use_fp16 = False
        self._bgr_cache: Optional[Dict[Path, np.ndarray]] = None
        self._cpu_tensor_cache: Optional[Dict[int, torch.Tensor]] = None
        logger.info("InterpolatorService initialized")

    def interpolate_frames(
        self,
        input_frames: List[Path],
        output_dir: Path,
        interpolation_factor: int = 2,
        batch_size: Optional[int] = None,
        frame_cache: Optional[Dict[Path, np.ndarray]] = None,
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

        self._bgr_cache = frame_cache

        if batch_size is None:
            batch_size = DEFAULT_RIFE_BATCH_SIZE
        batch_size = max(1, batch_size)

        logger.info(
            f"Interpolating {len(input_frames)} frames with factor "
            f"{interpolation_factor}x (batch_size={batch_size})"
        )

        logger.info("Loading RIFE model...")
        rife_model = self.model_loader.load_rife_model()
        self._use_fp16 = bool(getattr(rife_model, 'use_fp16', False))
        logger.info("RIFE model loaded successfully")

        try:
            self._target_size = self._determine_safe_size(input_frames)
            if self._target_size is None:
                raise RuntimeError(
                    "Failed to determine a safe target size for interpolation"
                )
            logger.info(
                f"Using target frame size (HxW): {self._target_size[0]}x{self._target_size[1]}"
            )

            self._output_size = None
            try:
                first_img = (
                    self._bgr_cache.get(input_frames[0])
                    if self._bgr_cache is not None else None
                )
                if first_img is None:
                    first_img = cv2.imread(str(input_frames[0]))
                if first_img is not None:
                    h0, w0 = first_img.shape[:2]
                    self._output_size = (h0, w0)
                    logger.info(f"Using output frame size (HxW): {h0}x{w0}")
            except Exception:
                self._output_size = None

            num_inputs = len(input_frames)
            factor = max(1, interpolation_factor)
            desired_total_frames = num_inputs * factor
            timesteps = TIMESTEPS.get(interpolation_factor, [0.5])

            for i in range(num_inputs):
                self._copy_frame(input_frames[i], output_dir, i * factor)
            for pos in range((num_inputs - 1) * factor + 1, desired_total_frames):
                self._copy_frame(input_frames[-1], output_dir, pos)

            work_items: List[Tuple[int, int, float, int]] = []
            for i in range(num_inputs - 1):
                for k, t in enumerate(timesteps):
                    out_pos = i * factor + 1 + k
                    work_items.append((i, i + 1, t, out_pos))

            _infer_device = (
                torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            )
            _dtype = torch.float16 if self._use_fp16 else torch.float32
            self._cpu_tensor_cache = {}
            for idx in range(num_inputs):
                path = input_frames[idx]
                frame = self._bgr_cache.get(path) if self._bgr_cache is not None else None
                if frame is None:
                    frame = cv2.imread(str(path))
                if frame is None:
                    raise RuntimeError(f"Could not load frame for tensor cache: {path}")
                frame = self._fit_to_size(frame, *self._target_size)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                t = torch.from_numpy(frame_rgb).permute(2, 0, 1).to(dtype=_dtype).div_(255.0)
                if _infer_device.type == 'cuda':
                    t = t.pin_memory()
                self._cpu_tensor_cache[idx] = t

            total_items = len(work_items)
            with tqdm(total=desired_total_frames, desc="Interpolating frames") as pbar:
                pbar.update(desired_total_frames - total_items)

                done_items = 0
                for start in range(0, total_items, batch_size):
                    chunk = work_items[start : start + batch_size]
                    last_frame_bgr = self._process_interpolation_chunk(
                        chunk, input_frames, output_dir, rife_model
                    )
                    done_items += len(chunk)
                    pbar.update(len(chunk))

                    if progress_callback and last_frame_bgr is not None:
                        preview_base64 = (
                            self.preview_service.encode_bgr_to_base64_jpeg(
                                last_frame_bgr,
                                max_side=preview_max_side,
                                jpeg_quality=preview_jpeg_quality,
                            )
                        )
                        progress_callback(
                            int(done_items / max(1, total_items) * 100),
                            preview_base64,
                        )

                if progress_callback:
                    progress_callback(100, None)

            logger.info(
                f"Generated {desired_total_frames} frames from {num_inputs} input frames"
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
            self._bgr_cache = None
            self._cpu_tensor_cache = None

        except Exception as e:
            logger.error(f"Frame interpolation failed: {e}")
            raise RuntimeError(f"Interpolation process failed: {e}")

    def _process_interpolation_chunk(
        self,
        chunk: List[Tuple[int, int, float, int]],
        input_frames: List[Path],
        output_dir: Path,
        rife_model,
    ) -> Optional[np.ndarray]:
        _device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        def get_tensor(idx: int) -> torch.Tensor:
            if self._cpu_tensor_cache is not None and idx in self._cpu_tensor_cache:
                return self._cpu_tensor_cache[idx].to(_device, non_blocking=True)
            return self._load_frame_tensor(input_frames[idx])

        img0 = torch.stack([get_tensor(item[0]) for item in chunk])
        img1 = torch.stack([get_tensor(item[1]) for item in chunk])
        timesteps = torch.tensor(
            [item[2] for item in chunk],
            device=img0.device,
            dtype=img0.dtype,
        ).view(-1, 1, 1, 1)

        output = self._run_rife_with_oom_split(rife_model, img0, img1, timesteps)

        last_frame_bgr: Optional[np.ndarray] = None
        for b, item in enumerate(chunk):
            out_pos = item[3]
            frame = self._tensor_to_frame(output[b : b + 1])
            if self._output_size is not None:
                frame = self._resize_to_exact(frame, *self._output_size)
            output_path = output_dir / f"frame_{out_pos:06d}.png"
            if not self._write_frame(output_path, frame):
                raise RuntimeError(
                    f"Failed to write interpolated frame to {output_path}"
                )
            last_frame_bgr = frame

        return last_frame_bgr

    def _run_rife_with_oom_split(
        self,
        rife_model,
        img0: torch.Tensor,
        img1: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        autocast_context = (
            torch.autocast(device_type='cuda', dtype=torch.float16)
            if self._use_fp16 and img0.device.type == 'cuda'
            else nullcontext()
        )
        try:
            with torch.inference_mode(), autocast_context:
                return rife_model.inference(img0, img1, timesteps)
        except torch.cuda.OutOfMemoryError:
            if img0.shape[0] == 1:
                raise
            torch.cuda.empty_cache()
            mid = img0.shape[0] // 2
            logger.warning(
                f"CUDA OOM interpolating batch of {img0.shape[0]}; "
                f"retrying as {mid}+{img0.shape[0] - mid}"
            )
            first = self._run_rife_with_oom_split(
                rife_model, img0[:mid], img1[:mid], timesteps[:mid]
            )
            second = self._run_rife_with_oom_split(
                rife_model, img0[mid:], img1[mid:], timesteps[mid:]
            )
            return torch.cat((first, second), dim=0)

    def _load_frame_tensor(self, path: Path) -> torch.Tensor:
        frame = self._bgr_cache.get(path) if self._bgr_cache is not None else None
        if frame is None:
            frame = cv2.imread(str(path))
        if frame is None:
            raise RuntimeError(f"Could not load frame: {path}")
        if self._target_size is None:
            raise RuntimeError("Target size not initialised before loading frames")

        target_h, target_w = self._target_size
        frame = self._fit_to_size(frame, target_h, target_w)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1)

        device = (
            torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        )
        dtype = (
            torch.float16
            if self._use_fp16 and device.type == 'cuda'
            else torch.float32
        )
        return tensor.to(device=device, dtype=dtype).div_(255.0)

    def _tensor_to_frame(self, tensor: torch.Tensor) -> np.ndarray:
        frame_tensor = tensor.squeeze(0).detach().float().clamp_(0.0, 1.0).cpu()
        frame_tensor = frame_tensor.permute(1, 2, 0)
        frame_np = (frame_tensor.numpy() * 255.0).astype(np.uint8)
        frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
        return frame_bgr

    def _copy_frame(self, source_path: Path, output_dir: Path, frame_idx: int) -> None:
        output_path = output_dir / f"frame_{frame_idx:06d}.png"
        frame = self._bgr_cache.get(source_path) if self._bgr_cache is not None else None
        if frame is None:
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

    def _determine_safe_size(self, frames: List[Path]) -> Optional[Tuple[int, int]]:
        max_h: Optional[int] = None
        max_w: Optional[int] = None
        for frame_path in frames:
            img = self._bgr_cache.get(frame_path) if self._bgr_cache is not None else None
            if img is None:
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
