import logging
import os
from pathlib import Path
from typing import Callable, Iterator, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from services.model_loader import ModelLoader
from services.preview_encoder import PreviewService

logger = logging.getLogger(__name__)

FAST_PNG_WRITE_PARAMS = [cv2.IMWRITE_PNG_COMPRESSION, 0]

DEFAULT_UPSCALE_BATCH_SIZE = max(1, int(os.getenv('UPSCALE_BATCH_SIZE', '4')))

_MOD_SCALE = 2


class UpscalerService:
    def __init__(
        self,
        model_loader: ModelLoader,
        preview_service: PreviewService,
    ):
        self.model_loader = model_loader
        self.preview_service = preview_service
        logger.info("UpscalerService initialized")

    def upscale_frames(
        self,
        input_frames: List[Path],
        output_dir: Path,
        upscale_factor: int = 2,
        tile_size: int = 1024,
        tile_padding: int = 10,
        batch_size: Optional[int] = None,
        progress_callback: Callable[[int, Optional[str]], None] = None,
        preview_max_side: Optional[int] = None,
        preview_jpeg_quality: Optional[int] = None,
    ) -> None:
        if not input_frames:
            logger.warning("No input frames provided for upscaling")
            return

        if upscale_factor == 1:
            logger.info(
                "Upscale factor is 1x; skipping upscaling step and copying frames"
            )
            for idx, frame_path in enumerate(input_frames):
                try:
                    img = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
                    if img is None:
                        logger.warning(f"Could not read frame: {frame_path}")
                        continue

                    output_path = output_dir / frame_path.name
                    self._write_frame(output_path, img)

                    if progress_callback:
                        preview_base64 = self.preview_service.encode_bgr_to_base64_jpeg(
                            img,
                            max_side=preview_max_side,
                            jpeg_quality=preview_jpeg_quality,
                        )
                        progress_callback(
                            int(((idx + 1) / len(input_frames) * 100)),
                            preview_base64,
                        )
                except Exception as e:
                    logger.error(f"Failed to copy frame {frame_path}: {e}")
            return

        if upscale_factor != 2:
            logger.warning(
                f"Unsupported upscale_factor={upscale_factor}. Falling back to 2x."
            )
            upscale_factor = 2

        if batch_size is None:
            batch_size = DEFAULT_UPSCALE_BATCH_SIZE
        batch_size = max(1, batch_size)

        logger.info(
            f"Upscaling {len(input_frames)} frames with factor {upscale_factor}x "
            f"(batch_size={batch_size})"
        )

        upsampler = self.model_loader.load_realesrgan_model(
            upscale_factor=upscale_factor,
            tile_size=tile_size,
            tile_padding=tile_padding,
        )

        try:
            import time

            start_time = time.time()
            total = len(input_frames)
            done = 0

            with tqdm(total=total, desc="Upscaling frames") as pbar:
                for batch in self._iter_size_batches(input_frames, batch_size):
                    paths = [path for path, _ in batch]
                    imgs = [img for _, img in batch]

                    upscaled = self._upscale_image_batch(upsampler, imgs)

                    last_img = None
                    for path, up_img in zip(paths, upscaled):
                        self._write_frame(output_dir / path.name, up_img)
                        last_img = up_img

                    done += len(batch)
                    pbar.update(len(batch))

                    if progress_callback and last_img is not None:
                        preview_base64 = self.preview_service.encode_bgr_to_base64_jpeg(
                            last_img,
                            max_side=preview_max_side,
                            jpeg_quality=preview_jpeg_quality,
                        )
                        progress_callback(int(done / total * 100), preview_base64)

            total_time = time.time() - start_time
            logger.info(f"Upscaling completed in {total_time:.2f}s")
            logger.info(f"Average time per frame: {total_time/len(input_frames):.3f}s")

        except Exception as e:
            logger.error(f"Frame upscaling failed: {e}")
            raise RuntimeError(f"Upscaling process failed: {e}")

    def _iter_size_batches(
        self, paths: List[Path], batch_size: int
    ) -> Iterator[List[Tuple[Path, np.ndarray]]]:
        batch: List[Tuple[Path, np.ndarray]] = []
        ref_shape: Optional[Tuple[int, int]] = None

        for path in paths:
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None:
                logger.warning(f"Could not read frame: {path}")
                continue

            shape = img.shape[:2]
            if batch and (len(batch) >= batch_size or shape != ref_shape):
                yield batch
                batch = []

            if not batch:
                ref_shape = shape
            batch.append((path, img))

        if batch:
            yield batch

    def _upscale_image_batch(
        self, upsampler, imgs: List[np.ndarray]
    ) -> List[np.ndarray]:
        if not imgs:
            return []

        model = upsampler.model
        device = upsampler.device
        netscale = upsampler.scale
        use_half = bool(upsampler.half)

        rgb = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in imgs]
        x = torch.from_numpy(np.stack(rgb)).permute(0, 3, 1, 2).contiguous()
        x = x.to(device=device, dtype=torch.float32).div_(255.0)
        if use_half:
            x = x.half()

        _, _, h, w = x.shape
        pad_h = (_MOD_SCALE - h % _MOD_SCALE) % _MOD_SCALE
        pad_w = (_MOD_SCALE - w % _MOD_SCALE) % _MOD_SCALE
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

        out = self._forward_with_oom_split(model, x)

        if pad_h or pad_w:
            out = out[:, :, : h * netscale, : w * netscale]

        out = out.detach().clamp_(0.0, 1.0).float().mul_(255.0).round_().byte().cpu()
        out_np = out.permute(0, 2, 3, 1).numpy()  # B, H, W, 3 (RGB)
        return [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in out_np]

    def _forward_with_oom_split(
        self, model, x: torch.Tensor
    ) -> torch.Tensor:
        try:
            with torch.inference_mode():
                return model(x)
        except torch.cuda.OutOfMemoryError:
            if x.shape[0] == 1:
                raise
            torch.cuda.empty_cache()
            mid = x.shape[0] // 2
            logger.warning(
                f"CUDA OOM upscaling batch of {x.shape[0]}; "
                f"retrying as {mid}+{x.shape[0] - mid}"
            )
            first = self._forward_with_oom_split(model, x[:mid])
            second = self._forward_with_oom_split(model, x[mid:])
            return torch.cat((first, second), dim=0)

    def upscale_single_frame(
        self, input_path: Path, output_path: Path, upscale_factor: int = 2
    ) -> bool:
        try:
            if upscale_factor != 2:
                logger.warning(
                    f"Unsupported upscale_factor={upscale_factor}. Falling back to 2x."
                )
                upscale_factor = 2

            upsampler = self.model_loader.load_realesrgan_model(upscale_factor)

            img = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
            if img is None:
                logger.error(f"Could not read image: {input_path}")
                return False

            img_upscaled, _ = upsampler.enhance(img, outscale=upscale_factor)

            self._write_frame(output_path, img_upscaled)
            return True

        except Exception as e:
            logger.error(f"Failed to upscale frame {input_path}: {e}")
            return False

    def _write_frame(self, output_path: Path, img) -> bool:
        if output_path.suffix.lower() == '.png':
            return cv2.imwrite(str(output_path), img, FAST_PNG_WRITE_PARAMS)
        return cv2.imwrite(str(output_path), img)

    def get_output_dimensions(
        self, input_width: int, input_height: int, upscale_factor: int
    ) -> Tuple[int, int]:
        return (input_width * upscale_factor, input_height * upscale_factor)
