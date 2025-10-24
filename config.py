"""
Configuration management for Bridge_V3 application
Handles environment variables and application settings
"""

import os
import logging
from typing import Dict, Any, Optional


class Config:
    """Application configuration with environment variable support"""
    
    # Camera Configuration
    CAMERA_TIMEOUT: int = int(os.getenv('CAMERA_TIMEOUT', '10'))
    CAMERA_RETRY_COUNT: int = int(os.getenv('CAMERA_RETRY_COUNT', '3'))
    CAMERA_RETRY_DELAY: float = float(os.getenv('CAMERA_RETRY_DELAY', '2.0'))
    
    # YOLO Model Configuration
    YOLO_CONFIDENCE: float = float(os.getenv('YOLO_CONFIDENCE', '0.15'))
    YOLO_IOU_THRESHOLD: float = float(os.getenv('YOLO_IOU_THRESHOLD', '0.45'))
    YOLO_IMAGE_SIZE: int = int(os.getenv('YOLO_IMAGE_SIZE', '640'))
    YOLO_MODEL_PATH: str = os.getenv('YOLO_MODEL_PATH', 'yolov8n.pt')
    
    # Performance Configuration
    ENHANCEMENT_ENABLED: bool = os.getenv('ENHANCEMENT_ENABLED', 'true').lower() == 'true'
    MAX_CONCURRENT_STREAMS: int = int(os.getenv('MAX_CONCURRENT_STREAMS', '1'))
    FRAME_CACHE_SIZE: int = int(os.getenv('FRAME_CACHE_SIZE', '10'))
    
    # File Storage Configuration
    SNAPSHOT_DIR: str = os.getenv('SNAPSHOT_DIR', 'bridge_snapshots')
    MAX_SNAPSHOT_AGE_DAYS: int = int(os.getenv('MAX_SNAPSHOT_AGE_DAYS', '30'))
    
    # Logging Configuration
    LOG_LEVEL: str = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FORMAT: str = os.getenv('LOG_FORMAT', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    LOG_FILE: Optional[str] = os.getenv('LOG_FILE')
    
    # Security Configuration
    ALLOWED_CAMERA_DOMAINS: list = [
        r'^https://s\d+\.nysdot\.skyvdn\.com:443/rtplive/.*\.m3u8$',
        r'^https://[\w\-\.]+\.nysdot\.gov/.*\.m3u8$'
    ]
    
    # Traffic Analysis Configuration
    VEHICLE_WEIGHTS: Dict[str, int] = {
        'car': int(os.getenv('VEHICLE_WEIGHT_CAR', '4000')),
        'truck': int(os.getenv('VEHICLE_WEIGHT_TRUCK', '35000')),
        'bus': int(os.getenv('VEHICLE_WEIGHT_BUS', '25000')),
        'motorcycle': int(os.getenv('VEHICLE_WEIGHT_MOTORCYCLE', '500'))
    }
    
    VEHICLE_CLASSES: Dict[int, str] = {
        2: 'car',
        3: 'motorcycle', 
        5: 'bus',
        7: 'truck'
    }
    
    # Bridge Specifications
    BRIDGE_SPECS: Dict[str, Any] = {
        'name': os.getenv('BRIDGE_NAME', 'Twin Bridges - I-87 Northway'),
        'main_span_ft': float(os.getenv('BRIDGE_MAIN_SPAN_FT', '599.8')),
        'total_length_ft': float(os.getenv('BRIDGE_TOTAL_LENGTH_FT', '778.9')),
        'main_span_m': float(os.getenv('BRIDGE_MAIN_SPAN_M', '182.8')),
        'total_length_m': float(os.getenv('BRIDGE_TOTAL_LENGTH_M', '237.4')),
        'age_years': int(os.getenv('BRIDGE_AGE_YEARS', '66')),
        'material': os.getenv('BRIDGE_MATERIAL', 'Steel through arch')
    }
    
    # Camera URLs - Allow override via environment variables for security
    CAMERA_CONFIGS: Dict[str, Dict[str, str]] = {
        "5821": {
            "name": "5821 - North of Mohawk (Twin Bridges)",
            "url": os.getenv('CAMERA_5821_URL', "https://s51.nysdot.skyvdn.com:443/rtplive/R1_003/playlist.m3u8"),
            "approaching_bridge": "right",
            "camera_location": "North of bridge looking south",
            "left_lane_label": "Southbound (PAST LOAD)",
            "right_lane_label": "Northbound (CURRENT LOAD)"
        },
        "3645": {
            "name": "3645 - South of Mohawk (Twin Bridges)",
            "url": os.getenv('CAMERA_3645_URL', "https://s51.nysdot.skyvdn.com:443/rtplive/R1_001/playlist.m3u8"),
            "approaching_bridge": "left",
            "camera_location": "South of bridge looking north",
            "left_lane_label": "Northbound (CURRENT LOAD)",
            "right_lane_label": "Southbound (PAST LOAD)"
        }
    }
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate configuration values"""
        logger = logging.getLogger(__name__)
        valid = True
        
        # Validate timeout values
        if not 1 <= cls.CAMERA_TIMEOUT <= 60:
            logger.error(f"Invalid CAMERA_TIMEOUT: {cls.CAMERA_TIMEOUT}, must be between 1 and 60")
            valid = False
            
        # Validate YOLO confidence
        if not 0.0 <= cls.YOLO_CONFIDENCE <= 1.0:
            logger.error(f"Invalid YOLO_CONFIDENCE: {cls.YOLO_CONFIDENCE}, must be between 0.0 and 1.0")
            valid = False
            
        # Validate file paths
        if not os.path.exists(cls.YOLO_MODEL_PATH):
            logger.warning(f"YOLO model file not found: {cls.YOLO_MODEL_PATH}")
            
        # Validate camera URLs
        import re
        for camera_id, config in cls.CAMERA_CONFIGS.items():
            url = config['url']
            if not any(re.match(pattern, url) for pattern in cls.ALLOWED_CAMERA_DOMAINS):
                logger.error(f"Invalid camera URL for {camera_id}: {url}")
                valid = False
                
        return valid
    
    @classmethod
    def setup_logging(cls) -> None:
        """Setup logging configuration"""
        logging.basicConfig(
            level=getattr(logging, cls.LOG_LEVEL.upper()),
            format=cls.LOG_FORMAT,
            filename=cls.LOG_FILE
        )
        
        # Ensure snapshot directory exists
        os.makedirs(cls.SNAPSHOT_DIR, exist_ok=True)
    
    @classmethod
    def get_camera_config(cls, camera_id: str) -> Optional[Dict[str, str]]:
        """Get configuration for specific camera"""
        return cls.CAMERA_CONFIGS.get(camera_id)
    
    @classmethod
    def get_all_cameras(cls) -> Dict[str, Dict[str, str]]:
        """Get all camera configurations"""
        return cls.CAMERA_CONFIGS.copy()


# Initialize logging when module is imported
Config.setup_logging()