import logging
from pathlib import Path
from typing import List, Optional
import ffmpeg

from utils.cleanup_manager import CleanupManager

logger = logging.getLogger(__name__)


class FrameManager:
    def __init__(self, temp_dir: Path, cleanup_manager: CleanupManager):
        self.temp_dir = temp_dir
        self.cleanup_manager = cleanup_manager
        logger.info("FrameManager initialized")
    
    def extract_frames(self, video_path: Path, output_dir: Optional[Path] = None, fps: Optional[float] = None) -> Path:

        if output_dir is None:
            output_dir = self.temp_dir / "frames_original"
        
        output_dir.mkdir(exist_ok=True)
        self.cleanup_manager.add_directory(output_dir)
        
        frame_pattern = output_dir / "frame_%06d.png"
        
        try:
            logger.info(f"Extracting frames from {video_path}")
            
            input_stream = ffmpeg.input(str(video_path))
            # Extract at native frame rate; avoid resampling/duplication. Ensure no vsync-induced dup/drop.
            if fps is not None:
                logger.info(f"Ignoring requested extraction fps={fps}; extracting at native rate")
            output_stream = (
                ffmpeg
                .output(
                    input_stream,
                    str(frame_pattern),
                    pix_fmt='rgb24',
                    vsync='0',
                    start_number=0
                )
            )

            (
                output_stream
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
            
            frame_count = len(list(output_dir.glob("frame_*.png")))
            logger.info(f"Extracted {frame_count} frames to {output_dir}")
            return output_dir
            
        except ffmpeg.Error as e:
            error_msg = e.stderr.decode() if e.stderr else str(e)
            error_lines = error_msg.strip().split('\n')
            relevant_error = '\n'.join(error_lines[-5:]) if len(error_lines) > 5 else error_msg
            logger.error(f"Frame extraction failed: {relevant_error}")
            raise RuntimeError(f"Failed to extract frames: {relevant_error}")
    
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
        quality: str = 'high'
    ) -> Path:

        try:
            logger.info(f"Encoding video from {frame_dir} to {output_path}")
            
            frame_pattern = frame_dir / "frame_%06d.png"
            if not any(frame_dir.glob("frame_*.png")):
                frame_files = sorted(frame_dir.glob("*.png"))
                if frame_files:
                    temp_frame_dir = self.temp_dir / "temp_frames_for_encoding"
                    temp_frame_dir.mkdir(exist_ok=True)
                    self.cleanup_manager.add_directory(temp_frame_dir)
                    
                    for i, frame_file in enumerate(frame_files):
                        symlink_path = temp_frame_dir / f"frame_{i+1:06d}.png"
                        symlink_path.symlink_to(frame_file.absolute())
                    
                    frame_pattern = temp_frame_dir / "frame_%06d.png"
            
            codec_params = self._get_codec_params(format, quality)
            
            input_stream = ffmpeg.input(str(frame_pattern), framerate=fps, start_number=0)
            output_stream = ffmpeg.output(
                input_stream,
                str(output_path),
                **codec_params
            )
            
            ffmpeg.run(output_stream, overwrite_output=True, capture_stdout=True, capture_stderr=True)
            
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
    
    def _get_codec_params(self, format: str, quality: str) -> dict:
        quality_settings = {
            'low': {'crf': 28, 'preset': 'fast'},
            'medium': {'crf': 23, 'preset': 'medium'},
            'high': {'crf': 18, 'preset': 'slow'}
        }
        
        settings = quality_settings.get(quality, quality_settings['high'])
        
        if format == 'mp4':
            return {
                'vcodec': 'libx265',
                'pix_fmt': 'yuv420p',
                'crf': settings['crf'],
                'preset': settings['preset'],
                'movflags': '+faststart',
                'vtag': 'hvc1'
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
                'vcodec': 'libx265',
                'pix_fmt': 'yuv420p',
                'crf': settings['crf']
            }
        else:
            return {
                'vcodec': 'libx265',
                'pix_fmt': 'yuv420p',
                'crf': settings['crf'],
                'preset': settings['preset']
            }
    
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