import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import ffmpeg

from utils.cleanup_manager import CleanupManager

logger = logging.getLogger(__name__)

FALSE_ENV_VALUES = {'0', 'false', 'no'}
FAST_PNG_COMPRESSION_LEVEL = 0
NVENC_HEVC_ENCODER = 'hevc_nvenc'
USE_HWACCEL_DECODE = os.getenv('USE_HWACCEL_DECODE', '1').lower() not in FALSE_ENV_VALUES


class FrameManager:
    def __init__(self, temp_dir: Path, cleanup_manager: CleanupManager):
        self.temp_dir = temp_dir
        self.cleanup_manager = cleanup_manager
        self.last_decode_mode = "unknown"
        self.last_encode_codec = "unknown"
        logger.info("FrameManager initialized")
    
    def extract_frames(self, video_path: Path, output_dir: Optional[Path] = None, fps: Optional[float] = None) -> Path:

        if output_dir is None:
            output_dir = self.temp_dir / "frames_original"
        
        output_dir.mkdir(exist_ok=True)
        self.cleanup_manager.add_directory(output_dir)
        
        frame_pattern = output_dir / "frame_%06d.png"
        
        if fps is not None:
            logger.info(f"Ignoring requested extraction fps={fps}; extracting at native rate")

        try:
            logger.info(f"Extracting frames from {video_path}")
            self._run_extract(video_path, frame_pattern, use_hwaccel=USE_HWACCEL_DECODE)
            frame_count = len(list(output_dir.glob("frame_*.png")))
            logger.info(f"Extracted {frame_count} frames to {output_dir}")
            return output_dir

        except ffmpeg.Error as e:
            error_msg = e.stderr.decode() if e.stderr else str(e)
            error_lines = error_msg.strip().split('\n')
            relevant_error = '\n'.join(error_lines[-5:]) if len(error_lines) > 5 else error_msg
            logger.error(f"Frame extraction failed: {relevant_error}")
            raise RuntimeError(f"Failed to extract frames: {relevant_error}")
    
    def _run_extract(self, video_path: Path, frame_pattern: Path, use_hwaccel: bool) -> None:
        def _build_stream(hwaccel: bool):
            kwargs = {'hwaccel': 'cuda'} if hwaccel else {}
            input_stream = ffmpeg.input(str(video_path), **kwargs)
            return ffmpeg.output(
                input_stream,
                str(frame_pattern),
                pix_fmt='rgb24',
                vsync='0',
                compression_level=FAST_PNG_COMPRESSION_LEVEL,
                start_number=0,
            ).overwrite_output()

        if use_hwaccel:
            try:
                _build_stream(True).run(capture_stdout=True, capture_stderr=True)
                self.last_decode_mode = "cuda"
                logger.debug("Frame extraction used CUDA hwaccel")
                return
            except ffmpeg.Error as e:
                warn = (e.stderr.decode() if e.stderr else str(e)).strip().split('\n')[-1]
                logger.warning(f"CUDA hwaccel decode failed, retrying with software: {warn}")

        _build_stream(False).run(capture_stdout=True, capture_stderr=True)
        self.last_decode_mode = "software"

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
        progress_callback: Optional[Callable[[int], None]] = None
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
            
            total_frames = len(frame_files)
            codec_params, uses_nvenc = self._get_codec_params(format, quality)
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
            except RuntimeError as e:
                if not uses_nvenc:
                    raise

                logger.warning(f"NVENC encode failed; retrying with software encoder: {e}")
                software_params = self._get_software_codec_params(format, quality)
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
        codec_params: Dict[str, object],
        total_frames: int,
        progress_callback: Optional[Callable[[int], None]] = None,
    ) -> None:
        input_stream = ffmpeg.input(str(frame_pattern), framerate=fps, start_number=0)
        output_stream = ffmpeg.output(
            input_stream,
            str(output_path),
            **codec_params,
        )

        args = output_stream.overwrite_output().get_args()
        cmd = ['ffmpeg'] + args

        logger.info(f"Running FFmpeg encode with codec: {codec_params.get('vcodec')}")

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            encoding='utf-8',
        )

        frame_regex = re.compile(r'frame=\s*(\d+)')
        stderr_lines = []

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

        remaining_stderr = process.stderr.read()
        if remaining_stderr:
            stderr_lines.append(remaining_stderr)

        if process.returncode != 0:
            stderr = ''.join(stderr_lines)
            error_lines = stderr.strip().split('\n')
            relevant_error = '\n'.join(error_lines[-8:]) if len(error_lines) > 8 else stderr
            raise RuntimeError(
                f"FFmpeg failed with return code {process.returncode}. Stderr: {relevant_error}"
            )

    def _get_codec_params(self, format: str, quality: str) -> Tuple[Dict[str, object], bool]:
        use_nvenc = os.getenv('USE_NVENC', '1').lower() not in FALSE_ENV_VALUES
        if use_nvenc and format == 'mp4' and self._supports_encoder(NVENC_HEVC_ENCODER):
            return self._get_nvenc_codec_params(quality), True

        return self._get_software_codec_params(format, quality), False

    def _get_nvenc_codec_params(self, quality: str) -> Dict[str, object]:
        quality_settings = {
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

    def _get_software_codec_params(self, format: str, quality: str) -> Dict[str, object]:
        quality_settings = {
            'low': {'crf': 28, 'preset': 'fast'},
            'medium': {'crf': 23, 'preset': 'medium'},
            'high': {'crf': 18, 'preset': 'slow'}
        }
        
        settings = quality_settings.get(quality, quality_settings['high'])
        
        if format == 'mp4':
            return {
                'vcodec': 'libx264',
                'pix_fmt': 'yuv420p',
                'crf': settings['crf'],
                'preset': settings['preset'],
                'movflags': '+faststart',
            }
        elif format == 'webm':
            return {
                'vcodec': 'libvpx-vp9',
                'pix_fmt': 'yuv420p',
                'crf': settings['crf'],
                'b:v': '2M'
            }
        elif format == 'avi':
            return {
                'vcodec': 'libx264',
                'pix_fmt': 'yuv420p',
                'crf': settings['crf']
            }
        else:
            return {
                'vcodec': 'libx264',
                'pix_fmt': 'yuv420p',
                'crf': settings['crf'],
                'preset': settings['preset']
            }

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
            return result.returncode == 0
        except OSError as e:
            logger.warning(f"Could not test encoder {encoder_name}: {e}")
            return False
    
    def get_video_info(self, video_path: Path) -> dict:
        try:
            probe = ffmpeg.probe(str(video_path))
            video_stream = next(
                (stream for stream in probe['streams'] if stream['codec_type'] == 'video'),
                None
            )
            
            if not video_stream:
                raise ValueError("No video stream found")
            
            return {
                'width': int(video_stream['width']),
                'height': int(video_stream['height']),
                'fps': eval(video_stream['r_frame_rate']),
                'duration': float(video_stream.get('duration', 0)),
                'codec': video_stream['codec_name'],
                'pix_fmt': video_stream.get('pix_fmt', 'unknown')
            }
            
        except Exception as e:
            logger.error(f"Failed to get video info: {e}")
            return {}
