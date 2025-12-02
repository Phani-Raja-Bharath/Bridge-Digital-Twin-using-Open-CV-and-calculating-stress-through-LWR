
import sys
import os
from unittest.mock import MagicMock

# Mock streamlit and other dependencies
sys.modules['streamlit'] = MagicMock()
sys.modules['cv2'] = MagicMock()
sys.modules['numpy'] = MagicMock()
sys.modules['pandas'] = MagicMock()
sys.modules['PIL'] = MagicMock()
sys.modules['ultralytics'] = MagicMock()

# Add current directory to path
sys.path.append(os.getcwd())

try:
    import bridge_modified
    print("Successfully imported bridge_modified")
    
    # Test get_report_css
    if hasattr(bridge_modified, 'get_report_css'):
        css = bridge_modified.get_report_css()
        if "body {" in css and ".header {" in css:
            print("get_report_css() works and returns expected CSS")
        else:
            print("get_report_css() returned unexpected content")
    else:
        print("get_report_css function missing")

    # Test detect_vehicles signature
    import inspect
    sig = inspect.signature(bridge_modified.detect_vehicles)
    if 'vehicle_weights' in sig.parameters:
        print("detect_vehicles accepts vehicle_weights")
    else:
        print("detect_vehicles missing vehicle_weights parameter")

except Exception as e:
    print(f"Error: {e}")
