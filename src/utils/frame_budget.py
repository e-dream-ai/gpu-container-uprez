import logging
from math import ceil

logger = logging.getLogger(__name__)

REFERENCE_FRAME_PIXELS = 3840 * 2160

def pixel_scaled_batch_size(batch_size: int, frame_pixels: int, stage: str) -> int:
    if frame_pixels <= 0:
        return batch_size

    factor = ceil(frame_pixels / REFERENCE_FRAME_PIXELS)
    if factor <= 1:
        return batch_size

    scaled = max(1, batch_size // factor)
    if scaled != batch_size:
        logger.info(
            f"{stage}: reducing batch size {batch_size} -> {scaled} for "
            f"{frame_pixels / 1e6:.1f}MP frames"
        )
    return scaled
