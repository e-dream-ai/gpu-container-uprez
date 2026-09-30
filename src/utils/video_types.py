from collections.abc import Callable
from typing import TypeAlias, TypedDict


CodecParams: TypeAlias = dict[str, str | int | float]
ProgressCallback: TypeAlias = Callable[[float, str | None], None]


class VideoInfo(TypedDict):
    width: int
    height: int
    fps: float
    duration: float
    frame_count: int
    codec: str
    pix_fmt: str
