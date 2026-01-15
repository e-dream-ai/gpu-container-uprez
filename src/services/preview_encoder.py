import base64
import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class PreviewService:
    def __init__(self, max_side: int = 512, jpeg_quality: int = 85):
        self.max_side = int(max_side)
        self.jpeg_quality = int(jpeg_quality)

    def encode_bgr_to_base64_jpeg(
        self,
        img_bgr: Optional[np.ndarray],
        max_side: Optional[int] = None,
        jpeg_quality: Optional[int] = None,
    ) -> Optional[str]:
        try:
            if img_bgr is None:
                return None

            h, w = img_bgr.shape[:2]
            if h == 0 or w == 0:
                return None

            max_side = int(max_side if max_side is not None else self.max_side)
            jpeg_quality = int(
                jpeg_quality if jpeg_quality is not None else self.jpeg_quality
            )

            scale = float(max_side) / float(max(h, w))
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))

            preview = cv2.resize(
                img_bgr,
                (new_w, new_h),
                interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
            )

            ok, buf = cv2.imencode(
                ".jpg",
                preview,
                [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
            )
            if not ok:
                return None

            return base64.b64encode(buf.tobytes()).decode("utf-8")
        except Exception as e:
            logger.warning(f"Preview generation failed: {e}")
            return None