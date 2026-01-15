import logging
from pathlib import Path
from typing import List, Callable, Optional, Tuple

import cv2
from tqdm import tqdm

from services.model_loader import ModelLoader
from services.preview_encoder import PreviewService

logger = logging.getLogger(__name__)


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
        tile_size: int = 512,
        tile_padding: int = 10,
        batch_size: int = 1,
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
                    cv2.imwrite(str(output_path), img)

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

        logger.info(f"Upscaling {len(input_frames)} frames with factor {upscale_factor}x")

        upsampler = self.model_loader.load_realesrgan_model(
            upscale_factor=upscale_factor,
            tile_size=tile_size,
            tile_padding=tile_padding,
        )

        try:
            import time

            start_time = time.time()

            with tqdm(total=len(input_frames), desc="Upscaling frames") as pbar:
                for i in range(0, len(input_frames), batch_size):
                    batch = input_frames[i : i + batch_size]
                    last_upscaled_img = self._process_frame_batch(
                        batch, output_dir, upsampler, upscale_factor
                    )
                    pbar.update(len(batch))

                    if progress_callback:
                        preview_base64 = self.preview_service.encode_bgr_to_base64_jpeg(
                            last_upscaled_img,
                            max_side=preview_max_side,
                            jpeg_quality=preview_jpeg_quality,
                        )
                        progress_callback(
                            int(((i + len(batch)) / len(input_frames) * 100)),
                            preview_base64,
                        )

            total_time = time.time() - start_time
            logger.info(f"Upscaling completed in {total_time:.2f}s")
            logger.info(f"Average time per frame: {total_time/len(input_frames):.3f}s")

        except Exception as e:
            logger.error(f"Frame upscaling failed: {e}")
            raise RuntimeError(f"Upscaling process failed: {e}")

    def _process_frame_batch(
        self,
        frame_batch: List[Path],
        output_dir: Path,
        upsampler,
        upscale_factor: int,
    ):
        last_img = None
        for frame_path in frame_batch:
            try:
                img = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
                if img is None:
                    logger.warning(f"Could not read frame: {frame_path}")
                    continue

                img_upscaled, _ = upsampler.enhance(img, outscale=upscale_factor)
                last_img = img_upscaled

                output_path = output_dir / frame_path.name
                cv2.imwrite(str(output_path), img_upscaled)

            except Exception as e:
                logger.error(f"Failed to process frame {frame_path}: {e}")
                continue
        return last_img

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

            cv2.imwrite(str(output_path), img_upscaled)
            return True

        except Exception as e:
            logger.error(f"Failed to upscale frame {input_path}: {e}")
            return False

    def get_output_dimensions(
        self, input_width: int, input_height: int, upscale_factor: int
    ) -> Tuple[int, int]:
        return (input_width * upscale_factor, input_height * upscale_factor)