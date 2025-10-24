"""
Vehicle Detection Module
Handles YOLO-based vehicle detection and classification
"""

import cv2
import numpy as np
import logging
from datetime import datetime
from typing import Dict, Tuple, Optional, Any
from config import Config

logger = logging.getLogger(__name__)

# Global model instance for efficiency
_model_instance = None


def get_yolo_model():
    """Get or initialize YOLO model instance"""
    global _model_instance
    
    try:
        from ultralytics import YOLO
        
        if _model_instance is None:
            logger.info(f"Loading YOLO model from {Config.YOLO_MODEL_PATH}")
            _model_instance = YOLO(Config.YOLO_MODEL_PATH)
            
        return _model_instance
        
    except ImportError:
        logger.warning("Ultralytics YOLO not available, falling back to basic detection")
        return None
    except Exception as e:
        logger.error(f"Failed to load YOLO model: {e}")
        return None


def validate_frame(frame: np.ndarray) -> bool:
    """Validate frame for processing"""
    if frame is None:
        logger.warning("Received None frame for detection")
        return False
    
    if not isinstance(frame, np.ndarray):
        logger.error(f"Invalid frame type: {type(frame)}, expected numpy.ndarray")
        return False
    
    if len(frame.shape) != 3 or frame.shape[2] != 3:
        logger.error(f"Invalid frame shape: {frame.shape}, expected (height, width, 3)")
        return False
    
    return True


def define_lane_regions(frame: np.ndarray, lane_divider_percent: float = 0.43) -> Dict[str, Any]:
    """Define lane regions with divider"""
    if not validate_frame(frame):
        return {}
    
    height, width = frame.shape[:2]
    divider_x = int(width * lane_divider_percent)
    
    regions = {
        'left_lane': (0, 0, divider_x, height),
        'right_lane': (divider_x, 0, width, height),
        'divider_x': divider_x,
        'total_area': width * height
    }
    
    logger.debug(f"Defined lane regions: left={regions['left_lane']}, right={regions['right_lane']}")
    return regions


def detect_vehicles_yolo(frame: np.ndarray, model, confidence: float = None) -> Dict[str, Any]:
    """Detect vehicles using YOLO model"""
    if confidence is None:
        confidence = Config.YOLO_CONFIDENCE
    
    try:
        results = model(
            frame, 
            conf=confidence, 
            iou=Config.YOLO_IOU_THRESHOLD, 
            imgsz=Config.YOLO_IMAGE_SIZE,
            verbose=False, 
            classes=[2, 3, 5, 7]  # car, motorcycle, bus, truck
        )
        
        vehicle_data = {
            'approaching_bridge': {'car': 0, 'truck': 0, 'bus': 0, 'motorcycle': 0},
            'past_bridge': {'car': 0, 'truck': 0, 'bus': 0, 'motorcycle': 0},
            'total': 0,
            'detections': []
        }
        
        if not results or not results[0].boxes:
            logger.debug("No vehicles detected")
            return vehicle_data
        
        # Define lane regions
        lane_regions = define_lane_regions(frame)
        if not lane_regions:
            return vehicle_data
        
        divider_x = lane_regions['divider_x']
        boxes = results[0].boxes
        
        for box in boxes:
            class_id = int(box.cls[0])
            
            if class_id in Config.VEHICLE_CLASSES:
                vehicle_type = Config.VEHICLE_CLASSES[class_id]
                
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                center_x = (x1 + x2) // 2
                confidence_score = float(box.conf[0])
                
                # Determine which lane and direction
                if center_x < divider_x:
                    lane = 'left'
                    direction = 'approaching_bridge'  # Default, can be configured per camera
                else:
                    lane = 'right'
                    direction = 'past_bridge'  # Default, can be configured per camera
                
                vehicle_data[direction][vehicle_type] += 1
                vehicle_data['total'] += 1
                
                # Store detection details
                detection = {
                    'type': vehicle_type,
                    'bbox': (x1, y1, x2, y2),
                    'center': (center_x, (y1 + y2) // 2),
                    'confidence': confidence_score,
                    'lane': lane,
                    'direction': direction
                }
                vehicle_data['detections'].append(detection)
        
        logger.debug(f"Detected {vehicle_data['total']} vehicles")
        return vehicle_data
        
    except Exception as e:
        logger.error(f"YOLO detection failed: {e}")
        return {
            'approaching_bridge': {'car': 0, 'truck': 0, 'bus': 0, 'motorcycle': 0},
            'past_bridge': {'car': 0, 'truck': 0, 'bus': 0, 'motorcycle': 0},
            'total': 0,
            'detections': []
        }


def count_vehicles_simple(frame: np.ndarray) -> Dict[str, Any]:
    """Simple vehicle detection fallback when YOLO is not available"""
    if not validate_frame(frame):
        return {
            'approaching_bridge': {'car': 0, 'truck': 0, 'bus': 0, 'motorcycle': 0},
            'past_bridge': {'car': 0, 'truck': 0, 'bus': 0, 'motorcycle': 0},
            'total': 0,
            'detections': []
        }
    
    # Simple motion detection - very basic fallback
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (21, 21), 0)
        
        # Use static background subtraction (simplified)
        thresh = cv2.threshold(blurred, 25, 255, cv2.THRESH_BINARY)[1]
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        vehicle_count = 0
        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Minimum area for vehicle
                vehicle_count += 1
        
        # Distribute as cars (simplified)
        result = {
            'approaching_bridge': {'car': vehicle_count, 'truck': 0, 'bus': 0, 'motorcycle': 0},
            'past_bridge': {'car': 0, 'truck': 0, 'bus': 0, 'motorcycle': 0},
            'total': vehicle_count,
            'detections': []
        }
        
        logger.debug(f"Simple detection found {vehicle_count} vehicles")
        return result
        
    except Exception as e:
        logger.error(f"Simple vehicle detection failed: {e}")
        return {
            'approaching_bridge': {'car': 0, 'truck': 0, 'bus': 0, 'motorcycle': 0},
            'past_bridge': {'car': 0, 'truck': 0, 'bus': 0, 'motorcycle': 0},
            'total': 0,
            'detections': []
        }


def draw_detection_overlay(frame: np.ndarray, vehicle_data: Dict[str, Any], 
                          camera_config: Dict[str, str]) -> np.ndarray:
    """Draw detection overlay on frame"""
    if not validate_frame(frame):
        return frame
    
    try:
        output_frame = frame.copy()
        height, width = frame.shape[:2]
        
        # Draw lane regions
        lane_regions = define_lane_regions(frame)
        if lane_regions:
            divider_x = lane_regions['divider_x']
            cv2.line(output_frame, (divider_x, 0), (divider_x, height), (0, 255, 255), 2)
        
        # Draw vehicle detections
        for detection in vehicle_data.get('detections', []):
            x1, y1, x2, y2 = detection['bbox']
            vehicle_type = detection['type']
            confidence = detection['confidence']
            
            # Color coding by vehicle type
            colors = {
                'car': (0, 255, 0),      # Green
                'truck': (0, 0, 255),    # Red
                'bus': (255, 0, 255),    # Magenta
                'motorcycle': (255, 255, 0)  # Cyan
            }
            color = colors.get(vehicle_type, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{vehicle_type}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(output_frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(output_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Calculate and display load
        approaching_count = sum(vehicle_data['approaching_bridge'].values())
        load_lbs = sum(vehicle_data['approaching_bridge'][v] * Config.VEHICLE_WEIGHTS.get(v, 4000) 
                      for v in ['car', 'truck', 'bus', 'motorcycle'])
        load_tons = load_lbs / 2000
        
        # Draw summary box
        summary_height = 100
        cv2.rectangle(output_frame, (0, height - summary_height), (width, height), (0, 0, 0), -1)
        
        # Add text information
        info_lines = [
            f"Vehicles: {approaching_count}",
            f"Load: {load_tons:.1f} tons ({load_lbs:,.0f} lbs)",
            f"Camera: {camera_config.get('camera_location', 'Unknown')}"
        ]
        
        y_offset = height - 75
        for line in info_lines:
            cv2.putText(output_frame, line, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 25
        
        return output_frame
        
    except Exception as e:
        logger.error(f"Failed to draw detection overlay: {e}")
        return frame


def detect_vehicles_with_overlay(frame: np.ndarray, camera_config: Dict[str, str], 
                                confidence: float = None) -> Tuple[Dict[str, Any], np.ndarray]:
    """Main detection function with overlay - combines YOLO and fallback detection"""
    if not validate_frame(frame):
        empty_result = {
            'approaching_bridge': {'car': 0, 'truck': 0, 'bus': 0, 'motorcycle': 0},
            'past_bridge': {'car': 0, 'truck': 0, 'bus': 0, 'motorcycle': 0},
            'total': 0,
            'detections': []
        }
        return empty_result, frame
    
    try:
        # Try YOLO detection first
        model = get_yolo_model()
        if model is not None:
            vehicle_data = detect_vehicles_yolo(frame, model, confidence)
        else:
            # Fallback to simple detection
            vehicle_data = count_vehicles_simple(frame)
        
        # Calculate additional metrics
        approaching_count = sum(vehicle_data['approaching_bridge'].values())
        load_lbs = sum(vehicle_data['approaching_bridge'][v] * Config.VEHICLE_WEIGHTS.get(v, 4000) 
                      for v in ['car', 'truck', 'bus', 'motorcycle'])
        load_tons = load_lbs / 2000
        
        # Add metadata
        vehicle_data['load_tons'] = load_tons
        vehicle_data['load_lbs'] = load_lbs
        vehicle_data['density'] = approaching_count * 5 / Config.BRIDGE_SPECS['total_length_m']
        vehicle_data['timestamp'] = datetime.now()
        
        # Draw overlay
        output_frame = draw_detection_overlay(frame, vehicle_data, camera_config)
        
        return vehicle_data, output_frame
        
    except Exception as e:
        logger.exception(f"Vehicle detection failed: {e}")
        empty_result = {
            'approaching_bridge': {'car': 0, 'truck': 0, 'bus': 0, 'motorcycle': 0},
            'past_bridge': {'car': 0, 'truck': 0, 'bus': 0, 'motorcycle': 0},
            'total': 0,
            'detections': [],
            'load_tons': 0,
            'load_lbs': 0,
            'density': 0,
            'timestamp': datetime.now()
        }
        return empty_result, frame