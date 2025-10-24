"""
Camera Service Module
Handles camera stream capture and image processing
"""

import subprocess
import cv2
import numpy as np
import logging
import io
import time
from PIL import Image
from typing import Optional, Dict, Any
from config import Config

logger = logging.getLogger(__name__)


class CameraService:
    """Service for handling camera operations"""
    
    def __init__(self):
        self.retry_count = Config.CAMERA_RETRY_COUNT
        self.retry_delay = Config.CAMERA_RETRY_DELAY
        self.timeout = Config.CAMERA_TIMEOUT
    
    def validate_stream_url(self, stream_url: str) -> str:
        """Validate stream URL to prevent injection attacks"""
        if not isinstance(stream_url, str):
            raise ValueError("Stream URL must be a string")
        
        if not stream_url.strip():
            raise ValueError("Stream URL cannot be empty")
        
        # Use configuration for allowed patterns
        import re
        if not any(re.match(pattern, stream_url) for pattern in Config.ALLOWED_CAMERA_DOMAINS):
            raise ValueError(f"Invalid or unauthorized stream URL: {stream_url}")
        
        return stream_url
    
    def capture_frame_ffmpeg(self, stream_url: str, timeout: Optional[float] = None) -> Optional[np.ndarray]:
        """Capture single frame from HLS stream with security validation"""
        try:
            # Validate inputs
            validated_url = self.validate_stream_url(stream_url)
            
            # Use instance timeout if not provided
            if timeout is None:
                timeout = self.timeout
            
            if not isinstance(timeout, (int, float)) or timeout <= 0 or timeout > 60:
                raise ValueError("Timeout must be a positive number between 1 and 60 seconds")
            
            # Construct command with validated inputs
            cmd = [
                'ffmpeg', 
                '-loglevel', 'error', 
                '-i', validated_url, 
                '-vframes', '1',
                '-f', 'image2pipe', 
                '-vcodec', 'png', 
                '-'
            ]
            
            logger.info(f"Capturing frame from validated URL: {validated_url}")
            result = subprocess.run(cmd, capture_output=True, timeout=timeout, check=False)
            
            if result.returncode == 0:
                if not result.stdout:
                    logger.warning("FFmpeg returned empty output")
                    return None
                    
                image = Image.open(io.BytesIO(result.stdout))
                frame = np.array(image)
                
                if len(frame.shape) == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                logger.debug(f"Successfully captured frame: {frame.shape}")
                return frame
            else:
                logger.error(f"FFmpeg failed with return code {result.returncode}: {result.stderr.decode('utf-8', errors='ignore')}")
                return None
                
        except subprocess.TimeoutExpired:
            logger.warning(f"FFmpeg timeout after {timeout} seconds for URL: {stream_url}")
            return None
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg process error: {e}")
            return None
        except (OSError, FileNotFoundError) as e:
            logger.error(f"FFmpeg not found or system error: {e}")
            return None
        except (ValueError, TypeError) as e:
            logger.error(f"Input validation error: {e}")
            return None
        except Exception as e:
            logger.exception(f"Unexpected error in frame capture: {e}")
            return None
    
    def capture_frame_with_retry(self, stream_url: str, timeout: Optional[float] = None) -> Optional[np.ndarray]:
        """Capture frame with retry logic"""
        for attempt in range(self.retry_count):
            try:
                frame = self.capture_frame_ffmpeg(stream_url, timeout)
                if frame is not None:
                    return frame
                
                if attempt < self.retry_count - 1:
                    logger.info(f"Retry {attempt + 1}/{self.retry_count} for camera capture")
                    time.sleep(self.retry_delay)
                    
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {e}")
                if attempt < self.retry_count - 1:
                    time.sleep(self.retry_delay)
        
        logger.error(f"Failed to capture frame after {self.retry_count} attempts")
        return None
    
    def preprocess_frame(self, frame: np.ndarray, enhance: bool = None) -> Optional[np.ndarray]:
        """Enhance image for better detection with proper error handling"""
        # Use configuration default if enhance not specified
        if enhance is None:
            enhance = Config.ENHANCEMENT_ENABLED
        
        # Input validation
        if frame is None:
            logger.warning("Received None frame for preprocessing")
            return None
        
        if not isinstance(frame, np.ndarray):
            logger.error(f"Invalid frame type: {type(frame)}, expected numpy.ndarray")
            return None
        
        if len(frame.shape) != 3 or frame.shape[2] != 3:
            logger.error(f"Invalid frame shape: {frame.shape}, expected (height, width, 3)")
            return frame
        
        if not enhance:
            return frame
        
        try:
            # Enhance image using CLAHE in LAB color space
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to lightness channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # Merge channels and convert back to BGR
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            
            logger.debug("Successfully enhanced frame")
            return enhanced
            
        except cv2.error as e:
            logger.error(f"OpenCV error during frame enhancement: {e}")
            return frame
        except (ValueError, TypeError) as e:
            logger.error(f"Invalid parameters for frame enhancement: {e}")
            return frame
        except Exception as e:
            logger.exception(f"Unexpected error during frame enhancement: {e}")
            return frame
    
    def get_camera_info(self, camera_id: str) -> Optional[Dict[str, Any]]:
        """Get camera configuration by ID"""
        return Config.get_camera_config(camera_id)
    
    def get_all_cameras(self) -> Dict[str, Dict[str, Any]]:
        """Get all available camera configurations"""
        return Config.get_all_cameras()
    
    def test_camera_connection(self, camera_id: str) -> bool:
        """Test if camera is accessible"""
        try:
            camera_config = self.get_camera_info(camera_id)
            if not camera_config:
                logger.error(f"Camera configuration not found for ID: {camera_id}")
                return False
            
            stream_url = camera_config['url']
            frame = self.capture_frame_ffmpeg(stream_url, timeout=5)  # Short timeout for testing
            
            if frame is not None and frame.size > 0:
                logger.info(f"Camera {camera_id} connection test successful")
                return True
            else:
                logger.warning(f"Camera {camera_id} connection test failed - no frame received")
                return False
                
        except Exception as e:
            logger.error(f"Camera {camera_id} connection test failed: {e}")
            return False
    
    def get_frame_stats(self, frame: np.ndarray) -> Dict[str, Any]:
        """Get basic statistics about a frame"""
        if frame is None or not isinstance(frame, np.ndarray):
            return {}
        
        try:
            stats = {
                'shape': frame.shape,
                'size': frame.size,
                'dtype': str(frame.dtype),
                'min_value': int(frame.min()),
                'max_value': int(frame.max()),
                'mean_value': float(frame.mean())
            }
            
            if len(frame.shape) == 3:
                stats['channels'] = frame.shape[2]
                stats['width'] = frame.shape[1]
                stats['height'] = frame.shape[0]
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get frame statistics: {e}")
            return {}


# Global service instance
_camera_service = None


def get_camera_service() -> CameraService:
    """Get or create camera service instance"""
    global _camera_service
    if _camera_service is None:
        _camera_service = CameraService()
    return _camera_service


# Convenience functions for backward compatibility
def capture_frame_ffmpeg(stream_url: str, timeout: Optional[float] = None) -> Optional[np.ndarray]:
    """Convenience function for frame capture"""
    service = get_camera_service()
    return service.capture_frame_ffmpeg(stream_url, timeout)


def preprocess_frame(frame: np.ndarray, enhance: bool = None) -> Optional[np.ndarray]:
    """Convenience function for frame preprocessing"""
    service = get_camera_service()
    return service.preprocess_frame(frame, enhance)