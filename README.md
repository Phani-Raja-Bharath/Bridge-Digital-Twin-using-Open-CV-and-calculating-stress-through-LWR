# Bridge_V3
Yolo Implementation 


## Project Overview

Bridge_V3 is a professional-grade hybrid digital twin application for monitoring and analyzing structural fatigue of the Twin Bridges (I-87 over Mohawk River) using real-time traffic data and advanced analytics. The application has been refactored for production use with modular architecture, comprehensive security, and enterprise-grade logging.

## Development Commands

### Setup
```bash
pip install -r requirements.txt
```

### Run Application
```bash
streamlit run bridge_camera_app.py
```

### Development Tools
```bash
# Code formatting
black src/ *.py

# Linting
flake8 src/ *.py

# Type checking
mypy src/ *.py

# Testing
pytest tests/ -v --cov=src
```

### Multi-Camera Monitoring (Different Ports)
```bash
streamlit run bridge_camera_app.py --server.port 8501  # Camera 5821
streamlit run bridge_camera_app.py --server.port 8502  # Camera 3645
```

### System Requirements
- FFmpeg must be installed on the system for HLS stream capture
- Internet connection required for NY511 camera streams
- YOLOv8 model file (`yolov8n.pt`) included in repository

## Modular Architecture

### Directory Structure
```
/src
  ├── models/
  │   └── vehicle_detection.py     # YOLO-based vehicle detection and classification
  ├── services/
  │   └── camera_service.py        # Camera stream capture and image processing
  ├── ui/
  │   └── (future UI components)
  └── utils/
      └── logging_config.py        # Comprehensive logging configuration
```

### Main Application (`bridge_camera_app.py`)
Refactored Streamlit application with improved structure:

1. **Configuration & Security** - Environment-based configuration with input validation
2. **Modular Imports** - Clean separation of concerns using dedicated modules
3. **Simulation Functions** - LWR traffic flow simulation and Monte Carlo analysis
4. **Visualization Functions** - Plotly interactive charts and HTML report generation
5. **Main Interface** - Three operational modes with enhanced error handling

### Technology Stack
- **Streamlit** - Web application framework
- **YOLOv8 (Ultralytics)** - Vehicle detection and classification
- **OpenCV + FFmpeg** - Video stream processing with security validation
- **scikit-learn** - Random Forest regression for predictions
- **Plotly** - Interactive visualizations
- **NumPy/Pandas/SciPy** - Data science stack

### Data Flow
```
NY511 Traffic Cameras (Validated HLS Stream) 
→ Secure FFmpeg Capture 
→ YOLO Vehicle Detection (with fallback)
→ LWR Traffic Simulation 
→ Monte Carlo Analysis 
→ Random Forest ML Model 
→ Streamlit Dashboard & Reports
```

## Configuration System (`config.py`)

### Environment Variables
The application uses environment-based configuration for security and flexibility:

```bash
# Camera Configuration
CAMERA_TIMEOUT=10
CAMERA_RETRY_COUNT=3
CAMERA_5821_URL="https://..."

# YOLO Configuration  
YOLO_CONFIDENCE=0.15
YOLO_MODEL_PATH="yolov8n.pt"

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE=logs/bridge_v3.log

# Vehicle Weights (lbs)
VEHICLE_WEIGHT_CAR=4000
VEHICLE_WEIGHT_TRUCK=35000
```

### Configuration Classes
- `Config` - Main configuration class with validation
- `Config.validate_config()` - Validates all configuration values
- `Config.setup_logging()` - Initializes logging system

## Security Features

### Input Validation
- **URL Validation**: Only allows authorized NY511 camera domains
- **Parameter Validation**: Type checking and range validation for all inputs
- **Command Injection Prevention**: Secure subprocess handling with validated inputs

### Error Handling
- **Specific Exception Types**: No bare `except:` clauses
- **Graceful Degradation**: Fallback mechanisms for optional features
- **Comprehensive Logging**: All errors logged with context

## Logging System

### Log Files
- `logs/bridge_v3.log` - Main application log (rotating, 10MB max)
- `logs/bridge_v3_errors.log` - Error-only log for monitoring
- `logs/bridge_v3_performance.log` - Performance metrics and timing

### Logging Features
- **Color-coded Console Output**: Visual log level differentiation
- **Structured Logging**: Consistent format with function names and line numbers
- **Performance Tracking**: Automatic timing for slow operations
- **Context Managers**: `LogContext` for operation tracking

### Usage Examples
```python
from src.utils.logging_config import get_logger, LogContext

logger = get_logger(__name__)

with LogContext(logger, "camera_capture", camera_id="5821"):
    # Operation automatically timed and logged
    frame = capture_frame_ffmpeg(url)
```

## Application Modes

1. **Live Monitoring Dashboard**: Real-time traffic detection with enhanced security
2. **Full Analysis Pipeline**: Complete workflow with performance monitoring
3. **Visualizations & Export**: Professional report generation with error handling

## Data Management

### File Structure
- `bridge_snapshots/` - Auto-captured traffic images with validation
- `logs/` - Comprehensive application logging
- CSV exports with enhanced data validation
- HTML reports with embedded visualizations

### Data Processing Pipeline
Enhanced pipeline with monitoring and validation:
- Secure camera stream capture with retry logic
- Robust vehicle detection with fallback mechanisms
- Validated traffic flow simulation
- Monitored Monte Carlo analysis
- Logged machine learning predictions

## Dependencies

### Production Dependencies
- streamlit>=1.39, ultralytics>=8.0, plotly>=5.20
- scikit-learn>=1.3, scipy>=1.10, opencv-python>=4.8
- pillow>=10.0, numpy>=1.24, pandas>=2.0

### Development Dependencies
- pytest>=7.0 (testing), black>=23.0 (formatting)
- flake8>=6.0 (linting), mypy>=1.0 (type checking)

### Feature Detection
The application includes runtime detection for optional dependencies with graceful fallbacks:
- YOLO detection → Basic motion detection
- Plotly visualizations → Text-based reports
- ML features → Statistical analysis only

## Development Guidelines

### Code Quality
- All functions include proper type hints and docstrings
- Comprehensive error handling with specific exception types
- Performance monitoring for operations >100ms
- Security validation for all external inputs

### Testing Strategy
- Unit tests for all core functions
- Integration tests for complete workflows
- Mock external dependencies (cameras, ML models)
- Performance benchmarks for critical operations

This refactored architecture provides enterprise-grade reliability, security, and maintainability while preserving all original functionality.