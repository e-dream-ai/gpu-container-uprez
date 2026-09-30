import logging
import os
import re
import subprocess
from collections import deque
from collections.abc import Generator, Iterator
from dataclasses import dataclass
from fractions import Fraction
from itertools import chain, islice
from math import isfinite
from pathlib import Path
from tempfile import TemporaryFile
from typing import IO, Callable, List, Optional, Tuple

import ffmpeg

from utils.cleanup_manager import CleanupManager
from utils.video_types import CodecParams, VideoInfo

logger = logging.getLogger(__name__)

FALSE_ENV_VALUES = {'0', 'false', 'no'}
FAST_PNG_COMPRESSION_LEVEL = 0
NVENC_HEVC_ENCODER = 'hevc_nvenc'

FFMPEG_DECODE_THREADS = max(1, int(os.getenv('FFMPEG_DECODE_THREADS', '4')))
FFMPEG_ENCODE_THREADS = max(1, int(os.getenv('FFMPEG_ENCODE_THREADS', '4')))
USE_HWACCEL_DECODE = os.getenv('USE_HWACCEL_DECODE', '1').lower() not in FALSE_ENV_VALUES


@dataclass(frozen=True)
class FrameChunk:
    frames: list[Path]
    pending_frames: tuple[bytes, ...]

    @property
    def is_final(self) -> bool:
        return not self.pending_frames


class VideoEncodingError(RuntimeError):
    pass


class FrameManager:
    def __init__(self, temp_dir: Path, cleanup_manager: CleanupManager) -> None:
        self.temp_dir = temp_dir
        self.cleanup_manager = cleanup_manager
        self.last_decode_mode = "unknown"
        self.last_encode_codec = "unknown"
        self.last_encode_params: CodecParams | None = None
        logger.info("FrameManager initialized")

    def iter_frames(self, video_path: Path) -> Generator[bytes, None, None]:
        decode_modes = [True, False] if USE_HWACCEL_DECODE else [False]
        dimensions: tuple[int, int] | None = None
        for use_hwaccel in decode_modes:
            input_options = {'hwaccel': 'cuda'} if use_hwaccel else {}
            stream = ffmpeg.input(str(video_path), **input_options)['v:0']
            stream = ffmpeg.output(
                stream,
                'pipe:1',
                format='image2pipe',
                vcodec='png',
                pix_fmt='rgb24',
                vsync='0',
                compression_level=FAST_PNG_COMPRESSION_LEVEL,
                threads=FFMPEG_DECODE_THREADS,
            ).global_args('-nostdin', '-loglevel', 'error')
            yielded_frame = False
            with TemporaryFile(dir=self.temp_dir) as stderr:
                process = subprocess.Popen(
                    ['ffmpeg'] + stream.get_args(),
                    stdout=subprocess.PIPE,
                    stderr=stderr,
                    stdin=subprocess.DEVNULL,
                )
                try:
                    if process.stdout is None:
                        raise RuntimeError('FFmpeg frame output pipe is unavailable')
                    while True:
                        frame = self._read_png(process.stdout)
                        if frame is None:
                            break
                        frame_dimensions = self.get_frame_dimensions(frame)
                        if dimensions is None:
                            dimensions = frame_dimensions
                        elif frame_dimensions != dimensions:
                            raise ValueError('Source video changes dimensions between frames')
                        self.last_decode_mode = 'cuda' if use_hwaccel else 'software'
                        yielded_frame = True
                        yield frame
                    if process.wait() != 0:
                        stderr.seek(max(0, stderr.tell() - 8192))
                        raise RuntimeError(stderr.read().decode('utf-8', errors='replace'))
                    if not yielded_frame:
                        raise RuntimeError('Input video contains no decodable frames')
                    return
                except RuntimeError as error:
                    if not use_hwaccel or yielded_frame:
                        raise
                    logger.warning(f"CUDA decode failed before the first frame; retrying in software: {error}")
                finally:
                    if process.stdout is not None:
                        process.stdout.close()
                    self._stop_process(process)

    @staticmethod
    def _stop_process(process: subprocess.Popen[bytes] | subprocess.Popen[str]) -> None:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

    @staticmethod
    def _read_png(stream: IO[bytes]) -> bytes | None:
        signature = stream.read(8)
        if not signature:
            return None
        if signature != b'\x89PNG\r\n\x1a\n':
            raise RuntimeError('Invalid PNG frame from FFmpeg')
        frame = bytearray(signature)
        while True:
            header = stream.read(8)
            if len(header) != 8:
                raise RuntimeError('Truncated PNG frame header from FFmpeg')
            size = int.from_bytes(header[:4], 'big') + 4
            data = stream.read(size)
            if len(data) != size:
                raise RuntimeError('Truncated PNG frame data from FFmpeg')
            frame.extend(header)
            frame.extend(data)
            if header[4:] == b'IEND':
                return bytes(frame)
    
    @staticmethod
    def get_frame_dimensions(frame: bytes) -> tuple[int, int]:
        if len(frame) < 24 or frame[12:16] != b'IHDR':
            raise ValueError('Missing PNG frame dimensions')
        width = int.from_bytes(frame[16:20], 'big')
        height = int.from_bytes(frame[20:24], 'big')
        if width <= 0 or height <= 0:
            raise ValueError('Invalid PNG frame dimensions')
        return width, height

    def extract_chunk(
        self,
        decoded_frames: Iterator[bytes],
        output_dir: Path,
        frame_limit: int,
        pending_frames: tuple[bytes, ...],
    ) -> FrameChunk:
        if frame_limit < max(2, len(pending_frames)):
            raise ValueError('Chunk size must accommodate the boundary frames')
        output_dir.mkdir(parents=True)
        frame_stream = chain(pending_frames, decoded_frames)
        frames: list[Path] = []
        last_frame: bytes | None = None
        for index, frame in enumerate(islice(frame_stream, frame_limit)):
            path = output_dir / f'frame_{index:06d}.png'
            path.write_bytes(frame)
            frames.append(path)
            last_frame = frame
        if last_frame is None:
            raise ValueError('Input video contains no frames')
        next_frame = next(frame_stream, None)
        pending = () if next_frame is None else (last_frame, next_frame)
        return FrameChunk(frames=frames, pending_frames=pending)

    def get_frame_paths(self, frame_dir: Path) -> List[Path]:
        frame_paths = sorted(frame_dir.glob("frame_*.png"))
        
        if not frame_paths:
            frame_paths = sorted(frame_dir.glob("*.png"))
            if not frame_paths:
                frame_paths = sorted(frame_dir.glob("*.jpg"))
        
        logger.debug(f"Found {len(frame_paths)} frames in {frame_dir}")
        return frame_paths
    
    def encode_video(
        self,
        frame_dir: Path,
        output_path: Path,
        fps: int = 30,
        format: str = 'mp4',
        quality: str = 'high',
        progress_callback: Optional[Callable[[int], None]] = None,
        frame_count: Optional[int] = None,
        codec_params: CodecParams | None = None,
    ) -> Path:

        try:
            logger.info(f"Encoding video from {frame_dir} to {output_path}")
            
            frame_files = sorted(frame_dir.glob("frame_*.png"))
            if not frame_files:
                frame_files = sorted(frame_dir.glob("*.png"))
                if not frame_files:
                    raise RuntimeError(f"No frames found in {frame_dir} for encoding")
                
                logger.info(f"Found {len(frame_files)} frames without standard naming, creating temporary symlinks")
                temp_frame_dir = self.temp_dir / "temp_frames_for_encoding"
                temp_frame_dir.mkdir(exist_ok=True)
                self.cleanup_manager.add_directory(temp_frame_dir)
                
                for i, frame_file in enumerate(frame_files):
                    symlink_path = temp_frame_dir / f"frame_{i:06d}.png"
                    symlink_path.symlink_to(frame_file.absolute())
                
                frame_pattern = temp_frame_dir / "frame_%06d.png"
            else:
                logger.info(f"Found {len(frame_files)} frames in standard format")
                frame_pattern = frame_dir / "frame_%06d.png"
            
            total_frames = len(frame_files) if frame_count is None else frame_count
            if total_frames <= 0 or total_frames > len(frame_files):
                raise ValueError(f"Invalid encoding frame count: {total_frames}")
            if codec_params is None:
                codec_params, uses_nvenc = self._get_codec_params(format, quality)
            else:
                uses_nvenc = False
            self.last_encode_params = dict(codec_params)
            self.last_encode_codec = str(codec_params.get('vcodec', 'unknown'))

            try:
                self._run_encode(
                    frame_pattern=frame_pattern,
                    output_path=output_path,
                    fps=fps,
                    codec_params=codec_params,
                    total_frames=total_frames,
                    progress_callback=progress_callback,
                )
            except VideoEncodingError as e:
                if not uses_nvenc:
                    raise

                logger.warning(f"NVENC encode failed; retrying with software encoder: {e}")
                software_params = self._get_software_codec_params(format, quality)
                self.last_encode_params = dict(software_params)
                self.last_encode_codec = str(software_params.get('vcodec', 'unknown'))
                self._run_encode(
                    frame_pattern=frame_pattern,
                    output_path=output_path,
                    fps=fps,
                    codec_params=software_params,
                    total_frames=total_frames,
                    progress_callback=progress_callback,
                )
            
            if not output_path.exists():
                raise RuntimeError("Output video file was not created")
            
            logger.info(f"Video encoded successfully: {output_path}")
            return output_path
            
        except ffmpeg.Error as e:
            error_msg = e.stderr.decode() if e.stderr else str(e)
            error_lines = error_msg.strip().split('\n')
            relevant_error = '\n'.join(error_lines[-5:]) if len(error_lines) > 5 else error_msg
            logger.error(f"Video encoding failed: {relevant_error}")
            raise RuntimeError(f"Failed to encode video: {relevant_error}")

    def _run_encode(
        self,
        frame_pattern: Path,
        output_path: Path,
        fps: int,
        codec_params: CodecParams,
        total_frames: int,
        progress_callback: Optional[Callable[[int], None]] = None,
    ) -> None:
        input_stream = ffmpeg.input(
            str(frame_pattern),
            framerate=fps,
            start_number=0,
            threads=FFMPEG_DECODE_THREADS,
        )
        output_stream = ffmpeg.output(
            input_stream,
            str(output_path),
            vframes=total_frames,
            **codec_params,
        )

        args = output_stream.global_args('-nostdin').overwrite_output().get_args()
        cmd = ['ffmpeg'] + args

        logger.info(f"Running FFmpeg encode with codec: {codec_params.get('vcodec')}")

        process = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            encoding='utf-8',
        )

        frame_regex = re.compile(r'frame=\s*(\d+)')
        stderr_lines: deque[str] = deque(maxlen=32)

        try:
            if process.stderr is None:
                raise RuntimeError('FFmpeg progress pipe is unavailable')
            while True:
                line = process.stderr.readline()
                if not line and process.poll() is not None:
                    break

                if line:
                    stderr_lines.append(line)
                    match = frame_regex.search(line)
                    if match and progress_callback:
                        current_frame = int(match.group(1))
                        percent = min(100, int((current_frame / total_frames) * 100))
                        progress_callback(percent)

            if process.returncode != 0:
                stderr = ''.join(stderr_lines)
                error_lines = stderr.strip().split('\n')
                relevant_error = '\n'.join(error_lines[-8:]) if len(error_lines) > 8 else stderr
                raise VideoEncodingError(
                    f"FFmpeg failed with return code {process.returncode}. Stderr: {relevant_error}"
                )
        finally:
            if process.stderr is not None:
                process.stderr.close()
            self._stop_process(process)

    def concatenate_videos(
        self,
        chunks: List[Tuple[Path, int]],
        output_path: Path,
        fps: int,
    ) -> Path:
        if not chunks:
            raise ValueError('No encoded chunks to concatenate')
        if len(chunks) == 1:
            chunks[0][0].replace(output_path)
            return output_path

        manifest = chunks[0][0].parent / 'concat.txt'
        with manifest.open('w', encoding='utf-8') as file:
            file.write('ffconcat version 1.0\n')
            for path, frame_count in chunks:
                file.write(f"file '{path.name}'\nduration {frame_count / fps:.12f}\n")

        output_options = {'c': 'copy'}
        if output_path.suffix == '.mp4':
            output_options['movflags'] = '+faststart'
            if self.last_encode_codec == NVENC_HEVC_ENCODER:
                output_options['vtag'] = 'hvc1'
        try:
            (
                ffmpeg.input(str(manifest), format='concat')
                .output(str(output_path), **output_options)
                .global_args('-nostdin', '-loglevel', 'error')
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
        except ffmpeg.Error as error:
            detail = error.stderr.decode('utf-8', errors='replace') if error.stderr else str(error)
            raise RuntimeError(f"Failed to concatenate video chunks: {detail[-8192:]}") from error
        return output_path

    def _get_codec_params(self, format: str, quality: str) -> tuple[CodecParams, bool]:
        use_nvenc = os.getenv('USE_NVENC', '1').lower() not in FALSE_ENV_VALUES
        if use_nvenc and format == 'mp4' and self._supports_encoder(NVENC_HEVC_ENCODER):
            return self._get_nvenc_codec_params(quality), True

        return self._get_software_codec_params(format, quality), False

    def _get_nvenc_codec_params(self, quality: str) -> CodecParams:
        quality_settings: dict[str, CodecParams] = {
            'low': {'cq': 28, 'preset': 'p3'},
            'medium': {'cq': 23, 'preset': 'p5'},
            'high': {'cq': 18, 'preset': 'p6'},
        }

        settings = quality_settings.get(quality, quality_settings['high'])
        return {
            'vcodec': NVENC_HEVC_ENCODER,
            'pix_fmt': 'yuv420p',
            'preset': settings['preset'],
            'tune': 'hq',
            'rc': 'vbr',
            'cq:v': settings['cq'],
            'b:v': '0',
            'movflags': '+faststart',
            'vtag': 'hvc1',
        }

    def _get_software_codec_params(self, format: str, quality: str) -> CodecParams:
        quality_settings: dict[str, CodecParams] = {
            'low': {'crf': 28, 'preset': 'fast'},
            'medium': {'crf': 23, 'preset': 'medium'},
            'high': {'crf': 18, 'preset': 'slow'}
        }
        
        settings = quality_settings.get(quality, quality_settings['high'])

        base: CodecParams = {'pix_fmt': 'yuv420p', 'threads': FFMPEG_ENCODE_THREADS}

        if format == 'webm':
            return {
                **base,
                'vcodec': 'libvpx-vp9',
                'crf': settings['crf'],
                'b:v': '2M'
            }

        if format == 'avi':
            return {
                **base,
                'vcodec': 'libx264',
                'crf': settings['crf']
            }

        params: CodecParams = {
            **base,
            'vcodec': 'libx264',
            'crf': settings['crf'],
            'preset': settings['preset'],
        }
        if format == 'mp4':
            params['movflags'] = '+faststart'
        return params

    def _supports_encoder(self, encoder_name: str) -> bool:
        try:
            result = subprocess.run(
                [
                    'ffmpeg', '-hide_banner',
                    '-f', 'lavfi', '-i', 'testsrc=duration=1:size=64x64:rate=1',
                    '-frames:v', '1',
                    '-c:v', encoder_name,
                    '-f', 'null', '-',
                ],
                capture_output=True,
                check=False,
            )
            if result.returncode != 0:
                reason = result.stderr.decode(errors='replace').strip().split('\n')[-1]
                logger.warning(f"Encoder {encoder_name} not available: {reason}")
                return False
            return True
        except OSError as e:
            logger.warning(f"Could not test encoder {encoder_name}: {e}")
            return False
    
    def get_video_info(self, video_path: Path) -> VideoInfo:
        try:
            probe = ffmpeg.probe(str(video_path))
            video_stream = next(
                (stream for stream in probe['streams'] if stream['codec_type'] == 'video'),
                None
            )
            
            if not video_stream:
                raise ValueError("No video stream found")

            rate = video_stream['r_frame_rate']
            if rate == '0/0':
                rate = video_stream.get('avg_frame_rate', rate)
            duration = video_stream.get('duration') or probe.get('format', {}).get('duration', 0)
            frame_count = video_stream.get('nb_frames', '0')
            info: VideoInfo = {
                'width': int(video_stream['width']),
                'height': int(video_stream['height']),
                'fps': float(Fraction(rate)),
                'duration': float(duration),
                'frame_count': int(frame_count) if str(frame_count).isdigit() else 0,
                'codec': video_stream['codec_name'],
                'pix_fmt': video_stream.get('pix_fmt', 'unknown')
            }
            if info['width'] <= 0 or info['height'] <= 0:
                raise ValueError('Invalid video dimensions')
            if not isfinite(info['fps']) or info['fps'] <= 0:
                raise ValueError('Invalid video frame rate')
            if not isfinite(info['duration']) or info['duration'] < 0:
                raise ValueError('Invalid video duration')
            return info
        except (ffmpeg.Error, KeyError, TypeError, ValueError, ZeroDivisionError) as error:
            raise ValueError(f'Failed to get video info: {error}') from error
