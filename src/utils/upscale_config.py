from pathlib import Path
from types import MappingProxyType
from typing import Mapping

REALESRGAN_MODEL_DIR = Path('/opt/models/realesrgan')

REALESRGAN_WEIGHT_FILES: Mapping[int, str] = MappingProxyType({
    2: 'RealESRGAN_x2plus.pth',
    4: 'RealESRGAN_x4plus.pth',
})

MODEL_UPSCALE_FACTORS: frozenset = frozenset(REALESRGAN_WEIGHT_FILES)

SKIP_UPSCALE_FACTOR = 1

DEFAULT_UPSCALE_FACTOR = 2

SUPPORTED_UPSCALE_FACTORS: frozenset = MODEL_UPSCALE_FACTORS | {SKIP_UPSCALE_FACTOR}

HEAVY_UPSCALE_THRESHOLD = 2

def realesrgan_weight_path(upscale_factor: int) -> Path:
    return REALESRGAN_MODEL_DIR / REALESRGAN_WEIGHT_FILES[upscale_factor]


def supported_factors_text() -> str:
    factors = sorted(SUPPORTED_UPSCALE_FACTORS)
    if len(factors) == 1:
        return str(factors[0])
    return f"{', '.join(str(f) for f in factors[:-1])}, or {factors[-1]}"
