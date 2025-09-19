import logging
from pathlib import Path
from typing import Dict, Any
import validators
import cv2

logger = logging.getLogger(__name__)


class InputValidator:

    
    @staticmethod
    def validate_video_url(url: str) -> Dict[str, Any]:
        result = {'valid': False, 'errors': [], 'warnings': []}
        
        if not url or not isinstance(url, str):
            result['errors'].append("Video URL is required and must be a string")
            return result
        
        if not validators.url(url):
            result['errors'].append("Invalid URL format")
            return result
        
        supported_protocols = ['http', 'https', 'ftp', 'ftps']
        protocol = url.split('://')[0].lower()
        
        if protocol not in supported_protocols:
            result['errors'].append(f"Unsupported protocol: {protocol}. Supported: {supported_protocols}")
            return result
        
        supported_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv']
        url_path = Path(url).suffix.lower()
        
        if url_path and url_path not in supported_extensions:
            result['warnings'].append(f"Unusual video extension: {url_path}. Supported: {supported_extensions}")
        
        result['valid'] = True
        return result
    
    @staticmethod
    def validate_upscale_factor(factor: Any) -> Dict[str, Any]:
        result = {'valid': False, 'errors': [], 'warnings': []}
        
        if not isinstance(factor, int):
            try:
                factor = int(factor)
            except (ValueError, TypeError):
                result['errors'].append("Upscale factor must be an integer")
                return result
        
        if factor != 2:
            result['errors'].append("Upscale factor must be 2")
            return result
                
        result['valid'] = True
        result['value'] = factor
        return result
    
    @staticmethod
    def validate_interpolation_factor(factor: Any) -> Dict[str, Any]:
        result = {'valid': False, 'errors': [], 'warnings': []}
        
        if not isinstance(factor, int):
            try:
                factor = int(factor)
            except (ValueError, TypeError):
                result['errors'].append("Interpolation factor must be an integer")
                return result
        
        if factor not in [2, 4, 8]:
            result['errors'].append("Interpolation factor must be 2, 4, or 8")
            return result
        
        if factor >= 4:
            result['warnings'].append(f"{factor}x interpolation will significantly increase processing time")
        
        result['valid'] = True
        result['value'] = factor
        return result
    
    @staticmethod
    def validate_output_fps(fps: Any) -> Dict[str, Any]:
        result = {'valid': False, 'errors': [], 'warnings': []}
        
        if not isinstance(fps, (int, float)):
            try:
                fps = float(fps)
            except (ValueError, TypeError):
                result['errors'].append("Output FPS must be a number")
                return result
        
        if fps <= 0:
            result['errors'].append("Output FPS must be greater than 0")
            return result
        
        if fps > 120:
            result['errors'].append("Output FPS cannot exceed 120")
            return result
        
        result['valid'] = True
        result['value'] = int(fps)
        return result
    
    @staticmethod
    def validate_output_format(format: Any) -> Dict[str, Any]:
        result = {'valid': False, 'errors': [], 'warnings': []}
        
        if not isinstance(format, str):
            result['errors'].append("Output format must be a string")
            return result
        
        format = format.lower().strip()
        supported_formats = ['mp4', 'webm', 'avi']
        
        if format not in supported_formats:
            result['errors'].append(f"Unsupported format: {format}. Supported: {supported_formats}")
            return result
        
        if format == 'webm':
            result['warnings'].append("WebM format may have limited compatibility")
        
        result['valid'] = True
        result['value'] = format
        return result
    
    @staticmethod
    def validate_video_file(file_path: Path) -> Dict[str, Any]:
        result = {
            'valid': False,
            'errors': [],
            'warnings': [],
            'video_info': {}
        }
        
        if not file_path.exists():
            result['errors'].append(f"Video file does not exist: {file_path}")
            return result
        
        if not file_path.is_file():
            result['errors'].append(f"Path is not a file: {file_path}")
            return result
        
        try:
            cap = cv2.VideoCapture(str(file_path))
            
            if not cap.isOpened():
                result['errors'].append("Cannot open video file - unsupported format or corrupted")
                return result
            
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            cap.release()
            
            result['video_info'] = {
                'frame_count': frame_count,
                'fps': fps,
                'width': width,
                'height': height,
                'duration': frame_count / fps if fps > 0 else 0
            }
            
            if frame_count <= 0:
                result['errors'].append("Video has no frames")
                return result
            
            if fps <= 0:
                result['errors'].append("Invalid video frame rate")
                return result
            
            if width <= 0 or height <= 0:
                result['errors'].append("Invalid video dimensions")
                return result
            
            if frame_count < 10:
                result['warnings'].append("Video is very short (less than 10 frames)")
            
            if fps > 60:
                result['warnings'].append(f"High frame rate detected: {fps:.2f} FPS")
            
            if width * height > 3840 * 2160:
                result['warnings'].append("Very high resolution video may require significant processing time")
            
            if result['video_info']['duration'] > 300:
                result['warnings'].append("Long video may require significant processing time")
            
            result['valid'] = True
            
        except Exception as e:
            result['errors'].append(f"Failed to analyze video file: {e}")
        
        return result
    
    @staticmethod
    def validate_processing_parameters(
        video_info: Dict[str, Any],
        upscale_factor: int,
        interpolation_factor: int
    ) -> Dict[str, Any]:
        result = {'valid': True, 'errors': [], 'warnings': [], 'estimates': {}}
        
        if not video_info:
            result['errors'].append("Video information is required")
            result['valid'] = False
            return result
        
        width = video_info.get('width', 0)
        height = video_info.get('height', 0)
        frame_count = video_info.get('frame_count', 0)
        
        output_width = width * upscale_factor
        output_height = height * upscale_factor
        output_frame_count = frame_count * interpolation_factor
        
        upscale_time_per_frame = 0.1 * upscale_factor  # seconds
        interpolation_time_per_frame = 0.05 * interpolation_factor
        estimated_processing_time = frame_count * (upscale_time_per_frame + interpolation_time_per_frame)
        
        frame_size_mb = (width * height * 3) / (1024 * 1024)
        estimated_memory_gb = frame_size_mb * upscale_factor * upscale_factor / 1024 * 2
        
        result['estimates'] = {
            'output_width': output_width,
            'output_height': output_height,
            'output_frame_count': output_frame_count,
            'estimated_processing_time_minutes': estimated_processing_time / 60,
            'estimated_memory_usage_gb': estimated_memory_gb
        }
        
        if output_width > 7680 or output_height > 4320:
            result['warnings'].append("Output resolution exceeds 8K - may cause memory issues")
        
        if estimated_processing_time > 3600:
            result['warnings'].append("Estimated processing time exceeds 1 hour")
        
        if estimated_memory_gb > 8:
            result['warnings'].append(f"Estimated memory usage: {estimated_memory_gb:.1f}GB - may require high-memory GPU")
        
        if output_frame_count > 10000:
            result['warnings'].append("Output will have >10k frames - consider shorter clips for testing")
        
        return result
    
    @staticmethod
    def validate_complete_input(input_data: Dict[str, Any]) -> Dict[str, Any]:
        result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'validated_params': {},
            'video_info': {},
            'estimates': {}
        }
        
        validators_map = {
            'video_url': InputValidator.validate_video_url,
            'upscale_factor': InputValidator.validate_upscale_factor,
            'interpolation_factor': InputValidator.validate_interpolation_factor,
            'output_fps': InputValidator.validate_output_fps,
            'output_format': InputValidator.validate_output_format
        }
        
        for param_name, validator_func in validators_map.items():
            if param_name in input_data:
                validation_result = validator_func(input_data[param_name])
                
                if not validation_result['valid']:
                    result['valid'] = False
                    result['errors'].extend([f"{param_name}: {error}" for error in validation_result['errors']])
                else:
                    result['validated_params'][param_name] = validation_result.get('value', input_data[param_name])
                
                result['warnings'].extend([f"{param_name}: {warning}" for warning in validation_result.get('warnings', [])])
        
        return result