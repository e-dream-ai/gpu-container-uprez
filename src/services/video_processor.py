import logging
import os
import shutil
from collections.abc import Iterator
from contextlib import closing
from dataclasses import dataclass
from math import ceil, isfinite
from pathlib import Path
from tempfile import TemporaryDirectory

from services.model_loader import ModelLoader
from services.frame_manager import FrameChunk, FrameManager
from services.preview_encoder import PreviewService
from services.upscaler_service import UpscalerService
from services.interpolator_service import InterpolatorService
from utils.cleanup_manager import CleanupManager
from utils.input_validator import InputValidator
from utils.upscale_config import DEFAULT_UPSCALE_FACTOR
from utils.video_types import CodecParams, ProgressCallback, VideoInfo

logger = logging.getLogger(__name__)

DISK_HEADROOM_BYTES = 2 * (1024 ** 3)


@dataclass(frozen=True)
class ProcessingOptions:
    upscale_factor: int
    interpolation_factor: int
    fps: int
    output_format: str
    tile_size: int
    tile_padding: int
    quality: str


def stage_progress(callback: ProgressCallback, start: float, end: float) -> ProgressCallback:
    def report(percent: float, preview: str | None = None) -> None:
        callback(start + (end - start) * percent / 100.0, preview)
    return report


class VideoProcessorService:

    def __init__(self, temp_dir: Path, cleanup_manager: CleanupManager) -> None:
        self.temp_dir = temp_dir
        self.cleanup_manager = cleanup_manager
        self.max_chunk_frames = int(os.getenv('VIDEO_CHUNK_FRAMES', '1800'))
        disk_budget_gb = float(os.getenv('VIDEO_CHUNK_DISK_GB', '8'))
        if self.max_chunk_frames < 2:
            raise ValueError('VIDEO_CHUNK_FRAMES must be at least 2')
        if not isfinite(disk_budget_gb) or disk_budget_gb <= 0:
            raise ValueError('VIDEO_CHUNK_DISK_GB must be finite and positive')
        self.chunk_disk_budget_bytes = int(disk_budget_gb * (1024 ** 3))
        self.model_loader = ModelLoader()
        self.preview_service = PreviewService()
        self.frame_manager = FrameManager(temp_dir, cleanup_manager)
        self.upscaler = UpscalerService(self.model_loader, self.preview_service)
        self.interpolator = InterpolatorService(self.model_loader, self.preview_service)
        logger.info('VideoProcessorService initialized')

    def _frame_storage_per_input(self, video_info: VideoInfo, options: ProcessingOptions) -> int:
        check = InputValidator.validate_processing_parameters(
            video_info={**video_info, 'frame_count': 1},
            upscale_factor=options.upscale_factor,
            interpolation_factor=options.interpolation_factor,
            output_format=options.output_format,
        )
        for warning in check['warnings']:
            logger.warning(f"Capacity: {warning}")
        if not check['valid']:
            raise ValueError('; '.join(check['errors']))
        source_bytes = video_info['width'] * video_info['height'] * 3
        processed_bytes = int(check['estimates']['estimated_frame_disk_bytes'])
        return ceil((source_bytes + processed_bytes) * 1.05)

    def _chunk_frame_limit(self, bytes_per_input: int, encoded_bytes: int) -> int:
        free = shutil.disk_usage(self.temp_dir).free
        available = free - DISK_HEADROOM_BYTES - encoded_bytes
        budget = min(self.chunk_disk_budget_bytes, available // 2)
        frame_limit = min(self.max_chunk_frames, budget // bytes_per_input)
        if frame_limit < 2:
            raise RuntimeError(
                f'Insufficient disk budget for a two-frame processing chunk: '
                f'{free / (1024 ** 3):.1f}GB free, reserving space for encoded '
                f'chunks, final output and {DISK_HEADROOM_BYTES / (1024 ** 3):.0f}GB headroom. '
                f'Reduce processing factors or increase available storage or VIDEO_CHUNK_DISK_GB.'
            )
        logger.info(f'Processing up to {frame_limit} source frames per chunk')
        return frame_limit

    def _remove_directory(self, path: Path) -> None:
        if not self.cleanup_manager.cleanup_directory(path):
            raise RuntimeError(f'Could not reclaim intermediate frame storage: {path}')
        self.cleanup_manager.remove_directory(path)

    def _require_frames(self, directory: Path, count: int) -> list[Path]:
        frames = self.frame_manager.get_frame_paths(directory)
        if len(frames) != count or any(
            path.name != f'frame_{index:06d}.png' for index, path in enumerate(frames)
        ):
            raise RuntimeError(f'Expected {count} consecutive frames in {directory}, found {len(frames)}')
        return frames

    def _prepare_chunk(
        self,
        chunk: FrameChunk,
        options: ProcessingOptions,
        boundary_path: Path,
        progress: ProgressCallback,
    ) -> Path:
        original_dir = chunk.frames[0].parent
        chunk_dir = original_dir.parent
        upscaled_dir = chunk_dir / 'upscaled'
        interpolated_dir = chunk_dir / 'interpolated'
        upscaled_dir.mkdir()
        interpolated_dir.mkdir()
        input_frames = chunk.frames
        if boundary_path.exists():
            boundary_path.replace(upscaled_dir / 'frame_000000.png')
            input_frames = input_frames[1:]
        frame_cache = self.upscaler.upscale_frames(
            input_frames=input_frames,
            output_dir=upscaled_dir,
            upscale_factor=options.upscale_factor,
            tile_size=options.tile_size,
            tile_padding=options.tile_padding,
            progress_callback=stage_progress(progress, 0.1, 0.45),
        )
        try:
            upscaled_frames = self._require_frames(upscaled_dir, len(chunk.frames))
            self._remove_directory(original_dir)
            self.interpolator.interpolate_frames(
                input_frames=upscaled_frames,
                output_dir=interpolated_dir,
                interpolation_factor=options.interpolation_factor,
                frame_cache=frame_cache,
                progress_callback=stage_progress(progress, 0.45, 0.85),
            )
            self._require_frames(interpolated_dir, len(chunk.frames) * options.interpolation_factor)
            if not chunk.is_final:
                upscaled_frames[-1].replace(boundary_path)
        finally:
            frame_cache = None
        self._remove_directory(upscaled_dir)
        return interpolated_dir

    @staticmethod
    def _chunk_progress(
        callback: ProgressCallback,
        processed_frames: int,
        chunk_frames: int,
        estimated_frames: int,
    ) -> ProgressCallback:
        denominator = max(estimated_frames, processed_frames + chunk_frames)

        def report(fraction: float, preview: str | None = None) -> None:
            percent = 5.0 + 89.0 * (processed_frames + chunk_frames * fraction) / denominator
            callback(min(94.0, percent), preview)
        return report

    def _process_chunks(
        self,
        decoded_frames: Iterator[bytes],
        first_frame: bytes,
        workspace: Path,
        video_info: VideoInfo,
        options: ProcessingOptions,
        progress: ProgressCallback,
    ) -> list[tuple[Path, int]]:
        bytes_per_input = self._frame_storage_per_input(video_info, options)
        estimated_frames = video_info['frame_count'] or ceil(video_info['duration'] * video_info['fps'])
        encoded_dir = workspace / 'encoded'
        encoded_dir.mkdir()
        chunks: list[tuple[Path, int]] = []
        encoded_bytes = 0
        processed_frames = 0
        codec_params: CodecParams | None = None
        pending_frames: tuple[bytes, ...] = (first_frame,)
        chunk_dir = workspace / 'frames'
        while True:
            frame_limit = self._chunk_frame_limit(bytes_per_input, encoded_bytes)
            chunk = self.frame_manager.extract_chunk(
                decoded_frames, chunk_dir / 'original', frame_limit, pending_frames,
            )
            owned_frames = len(chunk.frames) if chunk.is_final else len(chunk.frames) - 1
            output_frames = owned_frames * options.interpolation_factor
            chunk_progress = self._chunk_progress(progress, processed_frames, owned_frames, estimated_frames)
            chunk_progress(0.1, None)
            logger.info(f'Processing chunk {len(chunks) + 1}: {len(chunk.frames)} source frames')
            interpolated_dir = self._prepare_chunk(chunk, options, workspace / 'boundary.png', chunk_progress)
            chunk_path = encoded_dir / f'chunk_{len(chunks):06d}.{options.output_format}'
            encoding_progress = stage_progress(chunk_progress, 0.85, 1.0)
            self.frame_manager.encode_video(
                frame_dir=interpolated_dir,
                output_path=chunk_path,
                fps=options.fps,
                format=options.output_format,
                quality=options.quality,
                frame_count=output_frames,
                codec_params=codec_params,
                progress_callback=lambda percent: encoding_progress(percent, None),
            )
            codec_params = self.frame_manager.last_encode_params
            if codec_params is None:
                raise RuntimeError('Encoder settings were not retained for concatenation')
            chunks.append((chunk_path, output_frames))
            encoded_bytes += chunk_path.stat().st_size
            processed_frames += owned_frames
            self._remove_directory(chunk_dir)
            if chunk.is_final:
                return chunks
            pending_frames = chunk.pending_frames

    def _assemble_video(self, chunks: list[tuple[Path, int]], output_path: Path, fps: int) -> None:
        encoded_bytes = sum(path.stat().st_size for path, _ in chunks)
        if len(chunks) > 1 and shutil.disk_usage(self.temp_dir).free < encoded_bytes + DISK_HEADROOM_BYTES:
            raise RuntimeError('Insufficient disk space to concatenate encoded chunks into the final video')
        self.frame_manager.concatenate_videos(chunks, output_path, fps)

    def process_video(
        self,
        input_path: Path,
        upscale_factor: int = DEFAULT_UPSCALE_FACTOR,
        interpolation_factor: int = 2,
        output_fps: int = 0,
        output_format: str = 'mp4',
        tile_size: int = 1024,
        tile_padding: int = 10,
        quality: str = 'high',
        progress_callback: ProgressCallback | None = None,
    ) -> Path:
        last_progress = 0.0

        def update_progress(percent: float, preview: str | None = None) -> None:
            nonlocal last_progress
            last_progress = max(last_progress, percent)
            if progress_callback is not None:
                progress_callback(last_progress, preview)

        output_path = self.temp_dir / f'output.{output_format}'
        logger.info(f'Starting chunked video processing: {input_path}')
        try:
            video_info = self.frame_manager.get_video_info(input_path)
            source_fps = max(1, int(round(video_info['fps'])))
            final_fps = output_fps if output_fps > 0 else source_fps * interpolation_factor
            options = ProcessingOptions(
                upscale_factor=upscale_factor,
                interpolation_factor=interpolation_factor,
                fps=final_fps,
                output_format=output_format,
                tile_size=tile_size,
                tile_padding=tile_padding,
                quality=quality,
            )
            update_progress(5.0)
            with TemporaryDirectory(prefix='chunks_', dir=self.temp_dir) as workspace:
                with closing(self.frame_manager.iter_frames(input_path)) as decoded_frames:
                    first_frame = next(decoded_frames, None)
                    if first_frame is None:
                        raise ValueError('Input video contains no frames')
                    video_info['width'], video_info['height'] = self.frame_manager.get_frame_dimensions(first_frame)
                    chunks = self._process_chunks(
                        decoded_frames, first_frame, Path(workspace), video_info, options, update_progress,
                    )
                update_progress(95.0)
                self._assemble_video(chunks, output_path, final_fps)
            update_progress(100.0)
            logger.info(f'Video processing completed: {output_path}')
            return output_path
        except Exception as error:
            logger.error(f'Video processing failed: {error}', exc_info=True)
            self.cleanup_manager.cleanup_file(output_path)
            raise RuntimeError(f'Video processing pipeline failed: {error}') from error
