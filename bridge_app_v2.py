"""
HYBRID DIGITAL TWIN - BRIDGE FATIGUE MONITORING
Proof-of-Concept: Low-cost monitoring using existing cameras + weather data

Features:
- Live camera feed with YOLO vehicle detection
- Real-time weather from Open-Meteo API
- LWR traffic simulation + Monte Carlo
- Interactive scenario dashboard
- Combined traffic + environmental fatigue

Case Study: Twin Bridges (I-87 over Mohawk River, NY)

Author: Bharath
For: IABMAS 2026 / RSS 2026 Conference Submissions
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import time
import subprocess
from PIL import Image
import io
import logging
import requests
from scipy import stats
import base64

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Optional imports
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_absolute_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class BridgeConfig:
    """Twin Bridges specifications"""
    name: str = "Twin Bridges - I-87 Northway"
    location: str = "I-87 over Mohawk River, New York"
    latitude: float = 42.79
    longitude: float = -73.85
    year_built: int = 1959
    main_span_m: float = 182.8
    total_length_m: float = 237.4
    material: str = "Steel through arch"
    daily_traffic: int = 115000
    
    @property
    def age_years(self) -> int:
        return datetime.now().year - self.year_built


# Camera configurations
CAMERAS = {
    "5821 - North of Mohawk": {
        "url": "https://s51.nysdot.skyvdn.com:443/rtplive/R1_003/playlist.m3u8",
        "approaching_side": "right",
        "left_label": "Southbound (PAST)",
        "right_label": "Northbound (LOAD)"
    },
    "3645 - South of Mohawk": {
        "url": "https://s51.nysdot.skyvdn.com:443/rtplive/R1_001/playlist.m3u8",
        "approaching_side": "left",
        "left_label": "Northbound (LOAD)",
        "right_label": "Southbound (PAST)"
    },
}

VEHICLE_WEIGHTS = {'car': 4000, 'truck': 35000, 'bus': 25000, 'motorcycle': 500}
VEHICLE_CLASSES = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}


# =============================================================================
# WEATHER API (Open-Meteo - Free, No API Key)
# =============================================================================

def fetch_weather(lat: float, lon: float) -> Dict:
    """Fetch current weather from Open-Meteo API"""
    try:
        url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}"
            f"&current=temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m"
            f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum"
            f"&past_days=7&forecast_days=1"
            f"&timezone=America/New_York"
        )
        
        response = requests.get(url, timeout=10)
        data = response.json()
        
        current = data.get("current", {})
        daily = data.get("daily", {})
        
        # Count freeze-thaw cycles (temp crossing 0°C)
        freeze_thaw = 0
        if daily.get("temperature_2m_min") and daily.get("temperature_2m_max"):
            for tmin, tmax in zip(daily["temperature_2m_min"], daily["temperature_2m_max"]):
                if tmin is not None and tmax is not None:
                    if tmin < 0 < tmax:
                        freeze_thaw += 1
        
        return {
            "temperature": current.get("temperature_2m", 0),
            "humidity": current.get("relative_humidity_2m", 50),
            "precipitation": current.get("precipitation", 0),
            "wind_speed": current.get("wind_speed_10m", 0),
            "freeze_thaw_7day": freeze_thaw,
            "success": True
        }
    except Exception as e:
        logger.error(f"Weather fetch failed: {e}")
        return {
            "temperature": 10,
            "humidity": 50,
            "precipitation": 0,
            "wind_speed": 5,
            "freeze_thaw_7day": 0,
            "success": False
        }


def calculate_environmental_stress(
    temperature: float,
    humidity: float,
    precipitation: float,
    freeze_thaw_cycles: int,
    wind_speed: float,
    bridge_age: int
) -> Dict:
    """
    Calculate environmental stress factors for steel bridge corrosion.
    Based on ISO 9223 corrosivity categories and engineering literature.
    
    Returns normalized scores (0-100) for each factor and combined score.
    """
    
    # 1. Freeze-thaw stress (0-100)
    # 50+ cycles/year is severe for NY climate
    freeze_thaw_annual = freeze_thaw_cycles * 52 / 7  # Extrapolate to annual
    freeze_thaw_score = min(100, freeze_thaw_annual * 1.5)
    
    # 2. Humidity corrosion factor (0-100)
    # Steel corrosion accelerates above 60% RH
    if humidity < 60:
        humidity_score = humidity * 0.5
    else:
        humidity_score = 30 + (humidity - 60) * 1.75
    humidity_score = min(100, humidity_score)
    
    # 3. Temperature stress (0-100)
    # Thermal cycling and extreme temps
    if temperature < -10:
        temp_score = 80 + abs(temperature + 10) * 2
    elif temperature < 0:
        temp_score = 50 + abs(temperature) * 3
    elif temperature > 35:
        temp_score = 50 + (temperature - 35) * 3
    else:
        temp_score = 20 + abs(temperature - 15) * 1.5
    temp_score = min(100, temp_score)
    
    # 4. Precipitation/salt factor (0-100)
    # Winter precipitation = road salt = chloride corrosion
    is_winter = datetime.now().month in [11, 12, 1, 2, 3]
    salt_multiplier = 3.0 if (is_winter and temperature < 5) else 1.0
    precip_score = min(100, precipitation * 10 * salt_multiplier)
    
    # 5. Wind loading (0-100)
    # Dynamic stress on structure
    wind_score = min(100, wind_speed * 2)
    
    # 6. Age degradation factor
    # Older bridges more susceptible
    age_factor = 1.0 + (bridge_age - 50) * 0.01 if bridge_age > 50 else 1.0
    
    # Combined environmental score (weighted)
    combined = (
        freeze_thaw_score * 0.30 +  # Major factor for NY
        humidity_score * 0.20 +
        temp_score * 0.15 +
        precip_score * 0.25 +       # Salt corrosion critical
        wind_score * 0.10
    ) * age_factor
    
    combined = min(100, combined)
    
    return {
        "freeze_thaw_score": round(freeze_thaw_score, 1),
        "humidity_score": round(humidity_score, 1),
        "temperature_score": round(temp_score, 1),
        "precipitation_score": round(precip_score, 1),
        "wind_score": round(wind_score, 1),
        "age_factor": round(age_factor, 2),
        "combined": round(combined, 1)
    }


# =============================================================================
# CAMERA FUNCTIONS
# =============================================================================

def capture_frame(stream_url: str, timeout: int = 10) -> Optional[np.ndarray]:
    """Capture frame from HLS stream using ffmpeg"""
    try:
        cmd = [
            'ffmpeg', '-loglevel', 'error',
            '-i', stream_url,
            '-vframes', '1',
            '-f', 'image2pipe',
            '-vcodec', 'png', '-'
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=timeout)
        
        if result.returncode == 0 and result.stdout:
            image = Image.open(io.BytesIO(result.stdout))
            frame = np.array(image)
            if len(frame.shape) == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            return frame
        return None
    except Exception as e:
        logger.error(f"Frame capture error: {e}")
        return None


def detect_vehicles(
    frame: np.ndarray,
    model,
    camera_config: Dict,
    lane_divider: float = 0.43,
    confidence: float = 0.15,
    bridge_config: BridgeConfig = None
) -> Tuple[Dict, np.ndarray]:
    """
    Detect vehicles and calculate bridge load.
    Clean overlay: boxes + labels only, no text clutter.
    """
    
    if frame is None or model is None:
        return {}, frame
    
    try:
        output_frame = frame.copy()
        height, width = frame.shape[:2]
        divider_x = int(width * lane_divider)
        approaching_side = camera_config["approaching_side"]
        
        # Initialize counts
        vehicle_data = {
            "approaching": {"car": 0, "truck": 0, "bus": 0, "motorcycle": 0, "total": 0},
            "leaving": {"car": 0, "truck": 0, "bus": 0, "motorcycle": 0, "total": 0},
        }
        
        # Run YOLO
        results = model(frame, conf=confidence, iou=0.45, verbose=False, classes=[2, 3, 5, 7])
        
        # Draw lane divider (subtle)
        cv2.line(output_frame, (divider_x, 0), (divider_x, height), (0, 200, 255), 2)
        
        # Process detections
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                if class_id not in VEHICLE_CLASSES:
                    continue
                    
                vehicle_type = VEHICLE_CLASSES[class_id]
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                center_x = (x1 + x2) // 2
                
                # Determine if approaching or leaving
                if center_x < divider_x:
                    category = "approaching" if approaching_side == "left" else "leaving"
                else:
                    category = "approaching" if approaching_side == "right" else "leaving"
                
                vehicle_data[category][vehicle_type] += 1
                vehicle_data[category]["total"] += 1
                
                # Draw box - clean style
                if category == "approaching":
                    color = (0, 255, 0)  # Green
                    thickness = 2
                else:
                    color = (100, 100, 100)  # Gray
                    thickness = 1
                
                cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, thickness)
                
                # Small label background + text
                label = vehicle_type[:3].upper()
                font_scale = 0.4
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
                
                # Label above box
                label_y = y1 - 4 if y1 > 20 else y2 + th + 4
                cv2.rectangle(output_frame, (x1, label_y - th - 2), (x1 + tw + 4, label_y + 2), color, -1)
                cv2.putText(output_frame, label, (x1 + 2, label_y),
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1)
        
        # Calculate load
        load_lbs = sum(
            vehicle_data["approaching"][v] * VEHICLE_WEIGHTS.get(v, 4000)
            for v in ["car", "truck", "bus", "motorcycle"]
        )
        load_tons = load_lbs / 2000
        
        # Calculate density
        bridge_length = bridge_config.total_length_m if bridge_config else 237.4
        density = vehicle_data["approaching"]["total"] * 5 / bridge_length
        
        vehicle_data["load_tons"] = round(load_tons, 1)
        vehicle_data["load_lbs"] = load_lbs
        vehicle_data["density"] = round(density, 4)
        
        # NO text overlay on image - stats shown in Streamlit UI
        
        return vehicle_data, output_frame
        
    except Exception as e:
        logger.error(f"Detection error: {e}")
        return {}, frame


# =============================================================================
# SIMULATION & ML
# =============================================================================

def run_lwr_simulation(
    initial_density: float,
    road_length_m: float,
    v_max_mps: float,
    total_time: int = 300
) -> Dict:
    """LWR traffic flow simulation"""
    
    dx, dt = 10.0, 1.0
    rho_max = 0.2
    num_sections = int(road_length_m / dx)
    num_steps = int(total_time / dt)
    
    rho = np.ones(num_sections) * initial_density * rho_max
    rho += np.random.normal(0, 0.01 * rho_max, num_sections)
    rho = np.clip(rho, 0, rho_max)
    
    stress_history = []
    
    for step in range(num_steps):
        velocity = v_max_mps * (1 - rho / rho_max)
        flow = rho * velocity
        wave_speed = v_max_mps * (1 - 2 * rho / rho_max)
        
        rho_new = rho.copy()
        for i in range(1, num_sections - 1):
            if wave_speed[i] > 0:
                rho_new[i] = rho[i] - (dt / dx) * (flow[i] - flow[i-1])
            else:
                rho_new[i] = rho[i] - (dt / dx) * (flow[i+1] - flow[i])
        
        rho_new[0] = initial_density * rho_max
        rho = np.clip(rho_new, 0, rho_max)
        
        # Inject jam at midpoint
        if step == num_steps // 2:
            mid = num_sections // 2
            rho[mid:mid+10] = rho_max * 0.9
        
        stress = np.mean(rho) / rho_max * 100
        stress_history.append(stress)
    
    cumulative = np.trapz(stress_history, dx=dt)
    fatigue = min(cumulative / 100, 100)
    
    shockwave_speed = np.mean(np.abs(np.gradient(rho))) * v_max_mps
    
    return {
        "fatigue": fatigue,
        "shockwave_speed": shockwave_speed,
        "avg_density": np.mean(rho),
        "max_stress": max(stress_history)
    }


def run_monte_carlo(
    num_runs: int,
    road_length_m: float,
    live_density: Optional[float] = None
) -> pd.DataFrame:
    """Monte Carlo simulation"""
    
    results = []
    
    for run in range(num_runs):
        if live_density and run % 5 == 0:
            density = live_density
        else:
            density = np.random.uniform(0.2, 0.6)
        
        v_max = np.random.choice([40, 60, 80, 100]) / 3.6
        alpha = np.random.uniform(0.00005, 0.001)
        
        sim = run_lwr_simulation(density, road_length_m, v_max)
        
        results.append({
            "density": density,
            "v_max": v_max * 3.6,
            "alpha": alpha,
            "shockwave_speed": sim["shockwave_speed"],
            "fatigue": sim["fatigue"] * (alpha / 0.0001)
        })
    
    return pd.DataFrame(results)


def train_model(data: pd.DataFrame) -> Tuple:
    """Train Random Forest model"""
    
    if not SKLEARN_AVAILABLE or len(data) < 20:
        return None, {}
    
    features = ["density", "v_max", "alpha", "shockwave_speed"]
    X = data[features]
    y = data["fatigue"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    metrics = {
        "r2": r2_score(y_test, y_pred),
        "mae": mean_absolute_error(y_test, y_pred),
        "feature_importance": dict(zip(features, model.feature_importances_)),
        "y_test": y_test,
        "y_pred": y_pred
    }
    
    return model, metrics


def predict_fatigue(
    model,
    density: float,
    avg_shockwave: float
) -> float:
    """Predict traffic fatigue from current conditions"""
    
    if model is None:
        return density * 100  # Fallback
    
    X = pd.DataFrame([{
        "density": density,
        "v_max": 80,
        "alpha": 0.0005,
        "shockwave_speed": avg_shockwave
    }])
    
    return model.predict(X)[0]


# =============================================================================
# SCENARIO ANALYSIS
# =============================================================================

def calculate_scenario_fatigue(
    base_traffic_fatigue: float,
    base_env_stress: float,
    traffic_multiplier: float,
    truck_percentage: float,
    freeze_thaw_cycles: int,
    temperature: float,
    precipitation: float
) -> Dict:
    """Calculate fatigue for a given scenario"""
    
    # Adjust traffic fatigue
    truck_factor = 1 + (truck_percentage - 0.15) * 2  # Trucks have higher impact
    adjusted_traffic = base_traffic_fatigue * traffic_multiplier * truck_factor
    adjusted_traffic = min(100, adjusted_traffic)
    
    # Recalculate environmental
    env_stress = calculate_environmental_stress(
        temperature=temperature,
        humidity=50,
        precipitation=precipitation,
        freeze_thaw_cycles=freeze_thaw_cycles,
        wind_speed=10,
        bridge_age=datetime.now().year - 1959
    )
    
    # Combined fatigue (weighted)
    # Traffic is primary, environment is modifier
    combined = adjusted_traffic * 0.7 + env_stress["combined"] * 0.3
    combined = min(100, combined)
    
    return {
        "traffic_fatigue": round(adjusted_traffic, 1),
        "environmental_stress": round(env_stress["combined"], 1),
        "combined_fatigue": round(combined, 1),
        "env_breakdown": env_stress
    }


def get_status(score: float) -> Tuple[str, str]:
    """Get status label and color from fatigue score"""
    if score < 50:
        return "✅ SAFE OPERATION", "green"
    elif score < 70:
        return "⚠️ MONITOR CLOSELY", "orange"
    elif score < 85:
        return "🔶 SCHEDULE INSPECTION", "red"
    else:
        return "🔴 IMMEDIATE ACTION", "darkred"


# =============================================================================
# VISUALIZATIONS
# =============================================================================

def create_fatigue_breakdown_chart(traffic: float, environmental: float) -> go.Figure:
    """Stacked bar showing fatigue components"""
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=["Fatigue Score"],
        y=[traffic * 0.7],
        name="Traffic Stress (70%)",
        marker_color="#3498db"
    ))
    
    fig.add_trace(go.Bar(
        x=["Fatigue Score"],
        y=[environmental * 0.3],
        name="Environmental (30%)",
        marker_color="#e74c3c"
    ))
    
    fig.update_layout(
        barmode="stack",
        height=300,
        title="Fatigue Score Breakdown",
        yaxis_title="Score",
        showlegend=True,
        template="plotly_white"
    )
    
    return fig


def create_sensitivity_chart(base_traffic: float, base_env: float) -> go.Figure:
    """Show how fatigue changes with different scenarios"""
    
    traffic_mult = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5]
    scores = []
    
    for mult in traffic_mult:
        adj_traffic = base_traffic * mult
        combined = adj_traffic * 0.7 + base_env * 0.3
        scores.append(min(100, combined))
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=[f"{int((m-1)*100):+d}%" for m in traffic_mult],
        y=scores,
        mode="lines+markers",
        name="Combined Fatigue",
        line=dict(color="#2ecc71", width=3)
    ))
    
    # Threshold lines
    fig.add_hline(y=50, line_dash="dash", line_color="green", annotation_text="Safe")
    fig.add_hline(y=70, line_dash="dash", line_color="orange", annotation_text="Monitor")
    fig.add_hline(y=85, line_dash="dash", line_color="red", annotation_text="Critical")
    
    fig.update_layout(
        height=300,
        title="Sensitivity: Traffic Change Impact",
        xaxis_title="Traffic Change",
        yaxis_title="Fatigue Score",
        template="plotly_white"
    )
    
    return fig


def create_environmental_breakdown_chart(env_data: Dict) -> go.Figure:
    """Pie chart of environmental factors"""
    
    labels = ["Freeze-Thaw", "Humidity", "Temperature", "Precipitation", "Wind"]
    values = [
        env_data["freeze_thaw_score"],
        env_data["humidity_score"],
        env_data["temperature_score"],
        env_data["precipitation_score"],
        env_data["wind_score"]
    ]
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        marker_colors=["#3498db", "#9b59b6", "#e74c3c", "#1abc9c", "#f39c12"]
    )])
    
    fig.update_layout(
        height=300,
        title="Environmental Stress Factors",
        template="plotly_white"
    )
    
    return fig


# =============================================================================
# REPORT GENERATION
# =============================================================================

def generate_html_report(
    bridge_config: BridgeConfig,
    vehicle_data: Dict,
    weather: Dict,
    scenario_result: Dict,
    baseline_metrics: Dict
) -> str:
    """Generate HTML report for download"""
    
    status_text, status_color = get_status(scenario_result["combined_fatigue"])
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Bridge Fatigue Assessment Report</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; line-height: 1.6; }}
            h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
            h2 {{ color: #34495e; margin-top: 30px; }}
            .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px; margin-bottom: 30px; }}
            .header h1 {{ color: white; border: none; margin: 0; }}
            .header p {{ margin: 5px 0; opacity: 0.9; }}
            .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
            .card {{ background: #f8f9fa; border-radius: 8px; padding: 20px; border-left: 4px solid #3498db; }}
            .card.warning {{ border-left-color: #f39c12; }}
            .card.danger {{ border-left-color: #e74c3c; }}
            .card.success {{ border-left-color: #27ae60; }}
            .card h3 {{ margin: 0 0 10px 0; color: #7f8c8d; font-size: 0.9em; text-transform: uppercase; }}
            .card .value {{ font-size: 2em; font-weight: bold; color: #2c3e50; }}
            .score-box {{ background: linear-gradient(135deg, #74b9ff, #0984e3); color: white; padding: 30px; border-radius: 10px; text-align: center; margin: 30px 0; }}
            .score-box .score {{ font-size: 4em; font-weight: bold; }}
            .score-box .status {{ font-size: 1.5em; margin-top: 10px; }}
            table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
            th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background: #f8f9fa; }}
            .limitations {{ background: #fff3cd; border: 1px solid #ffc107; border-radius: 8px; padding: 20px; margin-top: 30px; }}
            .limitations h3 {{ color: #856404; margin-top: 0; }}
            .footer {{ margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #7f8c8d; font-size: 0.9em; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🌉 Hybrid Digital Twin</h1>
            <p>Bridge Fatigue Assessment Report</p>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <h2>Bridge Information</h2>
        <div class="grid">
            <div class="card">
                <h3>Structure</h3>
                <div class="value">{bridge_config.name}</div>
            </div>
            <div class="card">
                <h3>Age</h3>
                <div class="value">{bridge_config.age_years} years</div>
            </div>
            <div class="card">
                <h3>Material</h3>
                <div class="value">{bridge_config.material}</div>
            </div>
            <div class="card">
                <h3>Main Span</h3>
                <div class="value">{bridge_config.main_span_m}m</div>
            </div>
        </div>
        
        <h2>Current Traffic Observation</h2>
        <div class="grid">
            <div class="card success">
                <h3>Vehicles on Bridge</h3>
                <div class="value">{vehicle_data.get('approaching', {}).get('total', 'N/A')}</div>
            </div>
            <div class="card">
                <h3>Current Load</h3>
                <div class="value">{vehicle_data.get('load_tons', 'N/A')} tons</div>
            </div>
            <div class="card">
                <h3>Traffic Density</h3>
                <div class="value">{vehicle_data.get('density', 'N/A')}</div>
            </div>
        </div>
        
        <h2>Weather Conditions</h2>
        <div class="grid">
            <div class="card">
                <h3>Temperature</h3>
                <div class="value">{weather.get('temperature', 'N/A')}°C</div>
            </div>
            <div class="card">
                <h3>Humidity</h3>
                <div class="value">{weather.get('humidity', 'N/A')}%</div>
            </div>
            <div class="card">
                <h3>Precipitation</h3>
                <div class="value">{weather.get('precipitation', 'N/A')} mm</div>
            </div>
            <div class="card warning">
                <h3>Freeze-Thaw (7 days)</h3>
                <div class="value">{weather.get('freeze_thaw_7day', 'N/A')} cycles</div>
            </div>
        </div>
        
        <div class="score-box" style="background: linear-gradient(135deg, {'#27ae60, #2ecc71' if scenario_result['combined_fatigue'] < 50 else '#e74c3c, #c0392b' if scenario_result['combined_fatigue'] > 70 else '#f39c12, #e67e22'});">
            <div class="score">{scenario_result['combined_fatigue']}/100</div>
            <div class="status">{status_text}</div>
        </div>
        
        <h2>Fatigue Breakdown</h2>
        <table>
            <tr>
                <th>Component</th>
                <th>Score</th>
                <th>Weight</th>
                <th>Contribution</th>
            </tr>
            <tr>
                <td>Traffic Stress</td>
                <td>{scenario_result['traffic_fatigue']}</td>
                <td>70%</td>
                <td>{scenario_result['traffic_fatigue'] * 0.7:.1f}</td>
            </tr>
            <tr>
                <td>Environmental Stress</td>
                <td>{scenario_result['environmental_stress']}</td>
                <td>30%</td>
                <td>{scenario_result['environmental_stress'] * 0.3:.1f}</td>
            </tr>
            <tr style="font-weight: bold;">
                <td>Combined</td>
                <td colspan="3">{scenario_result['combined_fatigue']}</td>
            </tr>
        </table>
        
        <div class="limitations">
            <h3>⚠️ Critical Limitations</h3>
            <p>This is a <strong>proof-of-concept demonstration</strong> with significant limitations:</p>
            <ul>
                <li>Material properties are estimated, not measured from actual bridge specifications</li>
                <li>Fatigue score is a proxy metric, not validated against structural sensors</li>
                <li>Environmental correlations based on engineering literature, not site-specific calibration</li>
                <li>Computer vision load estimates lack weigh-in-motion validation</li>
            </ul>
            <p>This framework demonstrates feasibility of low-cost monitoring, not operational readiness.</p>
        </div>
        
        <div class="footer">
            <p>Hybrid Digital Twin - Bridge Fatigue Monitoring System</p>
            <p>Case Study: Twin Bridges (I-87 over Mohawk River, New York)</p>
            <p>For research and demonstration purposes only</p>
        </div>
    </body>
    </html>
    """
    
    return html


def generate_session_report(
    bridge_config: BridgeConfig,
    session_log: list,
    weather: Dict
) -> str:
    """Generate HTML report for monitoring session with images"""
    
    if not session_log:
        return "<html><body><p>No data captured</p></body></html>"
    
    # Calculate session stats
    total_vehicles = sum(entry["vehicle_data"]["approaching"]["total"] for entry in session_log)
    avg_vehicles = total_vehicles / len(session_log)
    avg_load = sum(entry["vehicle_data"]["load_tons"] for entry in session_log) / len(session_log)
    max_load = max(entry["vehicle_data"]["load_tons"] for entry in session_log)
    
    start_time = session_log[0]["timestamp"]
    end_time = session_log[-1]["timestamp"]
    
    # Build capture rows
    capture_rows = ""
    for i, entry in enumerate(session_log):
        vd = entry["vehicle_data"]
        approaching = vd.get("approaching", {})
        capture_rows += f"""
        <div class="capture-card">
            <div class="capture-header">
                <span class="capture-num">#{i+1}</span>
                <span class="capture-time">{entry['timestamp'].strftime('%H:%M:%S')}</span>
            </div>
            <img src="data:image/jpeg;base64,{entry['image_b64']}" alt="Capture {i+1}" />
            <div class="capture-stats">
                <div><strong>{approaching.get('total', 0)}</strong> vehicles</div>
                <div><strong>{vd.get('load_tons', 0)}</strong> tons</div>
                <div>🚗 {approaching.get('car', 0)} 🚛 {approaching.get('truck', 0)} 🚌 {approaching.get('bus', 0)}</div>
            </div>
        </div>
        """
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Bridge Monitoring Session Report</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; line-height: 1.6; }}
            .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px; margin-bottom: 30px; }}
            .header h1 {{ margin: 0; }}
            .header p {{ margin: 5px 0; opacity: 0.9; }}
            h2 {{ color: #34495e; margin-top: 30px; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
            .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; margin: 20px 0; }}
            .stat-card {{ background: #f8f9fa; border-radius: 8px; padding: 20px; text-align: center; }}
            .stat-card .value {{ font-size: 2em; font-weight: bold; color: #2c3e50; }}
            .stat-card .label {{ color: #7f8c8d; font-size: 0.9em; }}
            .captures-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin: 20px 0; }}
            .capture-card {{ background: white; border: 1px solid #ddd; border-radius: 8px; overflow: hidden; }}
            .capture-header {{ background: #34495e; color: white; padding: 10px 15px; display: flex; justify-content: space-between; }}
            .capture-num {{ font-weight: bold; }}
            .capture-card img {{ width: 100%; height: auto; }}
            .capture-stats {{ padding: 15px; display: flex; justify-content: space-around; background: #f8f9fa; }}
            .weather-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin: 20px 0; }}
            .weather-card {{ background: #e3f2fd; border-radius: 8px; padding: 15px; text-align: center; }}
            .limitations {{ background: #fff3cd; border: 1px solid #ffc107; border-radius: 8px; padding: 20px; margin-top: 30px; }}
            .limitations h3 {{ color: #856404; margin-top: 0; }}
            .footer {{ margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #7f8c8d; font-size: 0.9em; text-align: center; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>📹 Bridge Monitoring Session Report</h1>
            <p><strong>{bridge_config.name}</strong></p>
            <p>Session: {start_time.strftime('%Y-%m-%d %H:%M:%S')} - {end_time.strftime('%H:%M:%S')}</p>
            <p>{len(session_log)} captures recorded</p>
        </div>
        
        <h2>📊 Session Summary</h2>
        <div class="stats-grid">
            <div class="stat-card">
                <div class="value">{len(session_log)}</div>
                <div class="label">Captures</div>
            </div>
            <div class="stat-card">
                <div class="value">{avg_vehicles:.1f}</div>
                <div class="label">Avg Vehicles</div>
            </div>
            <div class="stat-card">
                <div class="value">{avg_load:.1f}</div>
                <div class="label">Avg Load (tons)</div>
            </div>
            <div class="stat-card">
                <div class="value">{max_load:.1f}</div>
                <div class="label">Max Load (tons)</div>
            </div>
        </div>
        
        <h2>🌤️ Weather Conditions</h2>
        <div class="weather-grid">
            <div class="weather-card">
                <div style="font-size: 1.5em;">{weather.get('temperature', 'N/A')}°C</div>
                <div>Temperature</div>
            </div>
            <div class="weather-card">
                <div style="font-size: 1.5em;">{weather.get('humidity', 'N/A')}%</div>
                <div>Humidity</div>
            </div>
            <div class="weather-card">
                <div style="font-size: 1.5em;">{weather.get('precipitation', 'N/A')} mm</div>
                <div>Precipitation</div>
            </div>
            <div class="weather-card">
                <div style="font-size: 1.5em;">{weather.get('freeze_thaw_7day', 'N/A')}</div>
                <div>F/T Cycles (7d)</div>
            </div>
        </div>
        
        <h2>📷 Captured Frames</h2>
        <div class="captures-grid">
            {capture_rows}
        </div>
        
        <div class="limitations">
            <h3>⚠️ Limitations</h3>
            <p>This monitoring session data is for <strong>research demonstration purposes only</strong>. 
            Vehicle detection may have errors. Load estimates are approximations based on average vehicle weights.
            This system has not been validated against physical sensors.</p>
        </div>
        
        <div class="footer">
            <p>Hybrid Digital Twin - Bridge Fatigue Monitoring System</p>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
    </body>
    </html>
    """
    
    return html


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    st.set_page_config(
        page_title="Hybrid Digital Twin - Bridge Monitoring",
        page_icon="🌉",
        layout="wide"
    )
    
    # Initialize
    bridge_config = BridgeConfig()
    
    # Check dependencies
    if not YOLO_AVAILABLE:
        st.error("YOLO not available. Install: `pip install ultralytics`")
        st.stop()
    
    # =========================================================================
    # HEADER
    # =========================================================================
    st.title("🌉 Hybrid Digital Twin - Bridge Fatigue Monitoring")
    st.markdown(
        f"**{bridge_config.name}** | "
        f"Built {bridge_config.year_built} | "
        f"Age: {bridge_config.age_years} years | "
        f"*Proof-of-concept: Low-cost monitoring using existing cameras*"
    )
    
    # =========================================================================
    # SIDEBAR
    # =========================================================================
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Camera selection
        selected_camera = st.selectbox("📹 Camera", list(CAMERAS.keys()))
        camera_config = CAMERAS[selected_camera]
        
        st.markdown("---")
        st.subheader("Detection")
        lane_divider = st.slider("Lane Divider", 0.3, 0.7, 0.43, 0.01)
        confidence = st.slider("Confidence", 0.05, 0.50, 0.15, 0.05)
        
        st.markdown("---")
        st.subheader("📹 Monitoring Session")
        capture_interval = st.select_slider(
            "Capture every",
            options=[10, 15, 30, 45, 60, 120, 300],
            value=30,
            format_func=lambda x: f"{x} sec" if x < 60 else f"{x//60} min"
        )
        monitor_duration = st.select_slider(
            "Duration",
            options=[1, 2, 5, 10, 15, 30, 60],
            value=5,
            format_func=lambda x: f"{x} min"
        )
        
        st.markdown("---")
        st.subheader("Simulation")
        mc_runs = st.slider("Monte Carlo Runs", 50, 300, 100, 50)
        
        st.markdown("---")
        st.warning(
            "⚠️ **Proof-of-Concept**\n\n"
            "Fatigue scores are proxy metrics. "
            "Not validated against real sensors."
        )
    
    # =========================================================================
    # LOAD MODELS
    # =========================================================================
    if "yolo_model" not in st.session_state:
        with st.spinner("Loading YOLO model..."):
            st.session_state.yolo_model = YOLO("yolov8n.pt")
    
    if "baseline_calculated" not in st.session_state:
        st.session_state.baseline_calculated = False
        st.session_state.baseline_traffic_fatigue = 50
        st.session_state.baseline_env_stress = 30
        st.session_state.mc_data = None
        st.session_state.rf_model = None
        st.session_state.rf_metrics = {}
        st.session_state.vehicle_data = {}
        st.session_state.weather = {}
    
    # Monitoring session state
    if "monitoring_active" not in st.session_state:
        st.session_state.monitoring_active = False
        st.session_state.monitoring_start_time = None
        st.session_state.session_log = []  # List of {timestamp, vehicle_data, image_b64}
    
    # =========================================================================
    # LIVE FEED + CURRENT CONDITIONS
    # =========================================================================
    st.markdown("---")
    
    col_video, col_conditions = st.columns([2, 1])
    
    with col_video:
        st.subheader("📹 Live Camera Feed")
        video_placeholder = st.empty()
        status_placeholder = st.empty()
        
        # Auto-refresh button
        if st.button("🔄 Refresh Frame"):
            pass  # Just triggers rerun
        
        # Capture frame
        status_placeholder.info("📡 Connecting to camera...")
        frame = capture_frame(camera_config["url"])
        
        if frame is not None:
            vehicle_data, annotated_frame = detect_vehicles(
                frame,
                st.session_state.yolo_model,
                camera_config,
                lane_divider,
                confidence,
                bridge_config
            )
            st.session_state.vehicle_data = vehicle_data
            
            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame_rgb, use_container_width=True)
            status_placeholder.success(f"✅ Live | {datetime.now().strftime('%H:%M:%S')}")
        else:
            video_placeholder.warning("📷 Camera unavailable - using demo mode")
            # Demo data
            st.session_state.vehicle_data = {
                "approaching": {"car": 5, "truck": 2, "bus": 1, "motorcycle": 0, "total": 8},
                "load_tons": 42.5,
                "density": 0.034
            }
    
    with col_conditions:
        st.subheader("📊 Current Conditions")
        
        # Traffic
        vd = st.session_state.vehicle_data
        approaching = vd.get("approaching", {})
        
        st.metric("🚗 Vehicles on Bridge", approaching.get("total", 0))
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Cars", approaching.get("car", 0))
            st.metric("Trucks", approaching.get("truck", 0))
        with col_b:
            st.metric("Buses", approaching.get("bus", 0))
            st.metric("Load", f"{vd.get('load_tons', 0)} tons")
        
        st.markdown("---")
        
        # Weather
        st.subheader("🌤️ Weather")
        weather = fetch_weather(bridge_config.latitude, bridge_config.longitude)
        st.session_state.weather = weather
        
        col_w1, col_w2 = st.columns(2)
        with col_w1:
            st.metric("Temp", f"{weather['temperature']}°C")
            st.metric("Humidity", f"{weather['humidity']}%")
        with col_w2:
            st.metric("Precip", f"{weather['precipitation']} mm")
            st.metric("F/T Cycles", weather['freeze_thaw_7day'])
    
    # =========================================================================
    # MONITORING SESSION
    # =========================================================================
    st.markdown("---")
    st.subheader("📹 Monitoring Session")
    
    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([1, 1, 2])
    
    with col_ctrl1:
        if st.button("▶️ START", type="primary", disabled=st.session_state.monitoring_active, use_container_width=True):
            st.session_state.monitoring_active = True
            st.session_state.monitoring_start_time = datetime.now()
            st.session_state.session_log = []
            st.rerun()
    
    with col_ctrl2:
        if st.button("⏹️ STOP", disabled=not st.session_state.monitoring_active, use_container_width=True):
            st.session_state.monitoring_active = False
            st.rerun()
    
    with col_ctrl3:
        if st.session_state.monitoring_active:
            elapsed = (datetime.now() - st.session_state.monitoring_start_time).total_seconds()
            total_duration = monitor_duration * 60
            captures_done = len(st.session_state.session_log)
            expected_captures = int(total_duration / capture_interval)
            
            progress = min(elapsed / total_duration, 1.0)
            st.progress(progress, text=f"Capture {captures_done}/{expected_captures} | {int(elapsed)}s / {total_duration}s")
            
            # Check if session complete
            if elapsed >= total_duration:
                st.session_state.monitoring_active = False
                st.success("✅ Monitoring session complete!")
    
    # Monitoring loop
    if st.session_state.monitoring_active:
        elapsed = (datetime.now() - st.session_state.monitoring_start_time).total_seconds()
        total_duration = monitor_duration * 60
        
        if elapsed < total_duration:
            captures_done = len(st.session_state.session_log)
            next_capture_at = captures_done * capture_interval
            
            if elapsed >= next_capture_at:
                # Time to capture
                frame = capture_frame(camera_config["url"])
                
                if frame is not None:
                    vehicle_data, annotated_frame = detect_vehicles(
                        frame, st.session_state.yolo_model, camera_config,
                        lane_divider, confidence, bridge_config
                    )
                    
                    # Convert frame to base64 for HTML report
                    frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(frame_rgb)
                    buffer = io.BytesIO()
                    pil_img.save(buffer, format="JPEG", quality=80)
                    img_b64 = base64.b64encode(buffer.getvalue()).decode()
                    
                    st.session_state.session_log.append({
                        "timestamp": datetime.now(),
                        "vehicle_data": vehicle_data,
                        "image_b64": img_b64
                    })
            
            # Auto-refresh for next capture
            time.sleep(2)
            st.rerun()
    
    # Show session log
    if st.session_state.session_log:
        st.markdown("**Session Log:**")
        
        log_df = pd.DataFrame([
            {
                "Time": entry["timestamp"].strftime("%H:%M:%S"),
                "Vehicles": entry["vehicle_data"]["approaching"]["total"],
                "Cars": entry["vehicle_data"]["approaching"]["car"],
                "Trucks": entry["vehicle_data"]["approaching"]["truck"],
                "Buses": entry["vehicle_data"]["approaching"]["bus"],
                "Load (tons)": entry["vehicle_data"]["load_tons"]
            }
            for entry in st.session_state.session_log
        ])
        st.dataframe(log_df, use_container_width=True, hide_index=True)
        
        # Download session report
        if not st.session_state.monitoring_active:
            session_html = generate_session_report(
                bridge_config,
                st.session_state.session_log,
                st.session_state.weather
            )
            st.download_button(
                "📄 Download Session Report (HTML with Images)",
                session_html,
                f"session_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                "text/html",
                use_container_width=True
            )
    
    # =========================================================================
    # ANALYZE BUTTON
    # =========================================================================
    st.markdown("---")
    
    if st.button("🔬 ANALYZE BASELINE CONDITIONS", type="primary", use_container_width=True):
        
        with st.spinner("Running analysis..."):
            progress = st.progress(0)
            
            # Get current density
            density = st.session_state.vehicle_data.get("density", 0.03)
            
            # Monte Carlo
            progress.progress(20, "Running Monte Carlo simulation...")
            mc_data = run_monte_carlo(mc_runs, bridge_config.total_length_m, density)
            st.session_state.mc_data = mc_data
            
            # Train model
            progress.progress(50, "Training ML model...")
            model, metrics = train_model(mc_data)
            st.session_state.rf_model = model
            st.session_state.rf_metrics = metrics
            
            # Predict traffic fatigue
            progress.progress(70, "Calculating fatigue...")
            traffic_fatigue = predict_fatigue(model, density, mc_data["shockwave_speed"].mean())
            st.session_state.baseline_traffic_fatigue = traffic_fatigue
            
            # Environmental stress
            weather = st.session_state.weather
            env_stress = calculate_environmental_stress(
                weather["temperature"],
                weather["humidity"],
                weather["precipitation"],
                weather["freeze_thaw_7day"],
                weather.get("wind_speed", 10),
                bridge_config.age_years
            )
            st.session_state.baseline_env_stress = env_stress["combined"]
            st.session_state.env_breakdown = env_stress
            
            progress.progress(100, "Done!")
            st.session_state.baseline_calculated = True
            time.sleep(0.3)
            progress.empty()
        
        st.success("✅ Baseline analysis complete!")
    
    # =========================================================================
    # SCENARIO DASHBOARD
    # =========================================================================
    if st.session_state.baseline_calculated:
        st.markdown("---")
        st.header("🎛️ Scenario Dashboard")
        st.markdown("*Adjust parameters to see how fatigue score changes*")
        
        col_sliders, col_result = st.columns([1, 1])
        
        with col_sliders:
            st.subheader("Traffic Scenarios")
            traffic_change = st.slider(
                "Traffic Volume Change",
                min_value=-30,
                max_value=50,
                value=0,
                step=5,
                format="%+d%%"
            )
            traffic_mult = 1 + traffic_change / 100
            
            truck_pct = st.slider(
                "Truck Percentage",
                min_value=5,
                max_value=40,
                value=15,
                step=5,
                format="%d%%"
            ) / 100
            
            st.subheader("Environmental Scenarios")
            scenario_temp = st.slider(
                "Temperature (°C)",
                min_value=-20,
                max_value=40,
                value=int(st.session_state.weather.get("temperature", 10))
            )
            
            scenario_freeze = st.slider(
                "Freeze-Thaw Cycles (weekly)",
                min_value=0,
                max_value=14,
                value=st.session_state.weather.get("freeze_thaw_7day", 0)
            )
            
            scenario_precip = st.slider(
                "Precipitation (mm)",
                min_value=0,
                max_value=50,
                value=int(st.session_state.weather.get("precipitation", 0))
            )
        
        # Calculate scenario
        scenario = calculate_scenario_fatigue(
            st.session_state.baseline_traffic_fatigue,
            st.session_state.baseline_env_stress,
            traffic_mult,
            truck_pct,
            scenario_freeze,
            scenario_temp,
            scenario_precip
        )
        
        with col_result:
            st.subheader("Fatigue Prediction")
            
            status_text, status_color = get_status(scenario["combined_fatigue"])
            
            # Big score display
            st.markdown(
                f"""
                <div style="background: linear-gradient(135deg, #667eea, #764ba2); 
                            padding: 30px; border-radius: 15px; text-align: center; color: white;">
                    <div style="font-size: 4em; font-weight: bold;">{scenario['combined_fatigue']}</div>
                    <div style="font-size: 1.2em;">/ 100</div>
                    <div style="font-size: 1.5em; margin-top: 10px;">{status_text}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            st.markdown("")
            
            col_t, col_e = st.columns(2)
            with col_t:
                st.metric("Traffic Stress", f"{scenario['traffic_fatigue']:.1f}", 
                         delta=f"{scenario['traffic_fatigue'] - st.session_state.baseline_traffic_fatigue:.1f}")
            with col_e:
                st.metric("Environmental", f"{scenario['environmental_stress']:.1f}",
                         delta=f"{scenario['environmental_stress'] - st.session_state.baseline_env_stress:.1f}")
        
        # Charts
        st.markdown("---")
        st.subheader("📈 Analysis Charts")
        
        tab1, tab2, tab3 = st.tabs(["Fatigue Breakdown", "Sensitivity", "Environmental Factors"])
        
        with tab1:
            fig = create_fatigue_breakdown_chart(scenario["traffic_fatigue"], scenario["environmental_stress"])
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            fig = create_sensitivity_chart(st.session_state.baseline_traffic_fatigue, scenario["environmental_stress"])
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            fig = create_environmental_breakdown_chart(scenario["env_breakdown"])
            st.plotly_chart(fig, use_container_width=True)
        
        # =====================================================================
        # DOWNLOADS
        # =====================================================================
        st.markdown("---")
        st.subheader("📥 Download Reports")
        
        col_d1, col_d2, col_d3 = st.columns(3)
        
        with col_d1:
            html_report = generate_html_report(
                bridge_config,
                st.session_state.vehicle_data,
                st.session_state.weather,
                scenario,
                st.session_state.rf_metrics
            )
            st.download_button(
                "📄 Download HTML Report",
                html_report,
                f"bridge_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                "text/html",
                use_container_width=True
            )
        
        with col_d2:
            if st.session_state.mc_data is not None:
                csv = st.session_state.mc_data.to_csv(index=False)
                st.download_button(
                    "📊 Download MC Data (CSV)",
                    csv,
                    f"monte_carlo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv",
                    use_container_width=True
                )
        
        with col_d3:
            # Summary text
            summary = f"""
HYBRID DIGITAL TWIN - BRIDGE FATIGUE ASSESSMENT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Bridge: {bridge_config.name}
Age: {bridge_config.age_years} years

CURRENT CONDITIONS:
- Vehicles on bridge: {st.session_state.vehicle_data.get('approaching', {}).get('total', 'N/A')}
- Load: {st.session_state.vehicle_data.get('load_tons', 'N/A')} tons
- Temperature: {st.session_state.weather.get('temperature')}°C
- Freeze-thaw cycles (7d): {st.session_state.weather.get('freeze_thaw_7day')}

FATIGUE ASSESSMENT:
- Traffic stress: {scenario['traffic_fatigue']}/100
- Environmental stress: {scenario['environmental_stress']}/100
- Combined fatigue: {scenario['combined_fatigue']}/100
- Status: {status_text}

LIMITATIONS:
This is a proof-of-concept. Fatigue scores are proxy metrics,
not validated against actual structural monitoring data.
            """
            st.download_button(
                "📝 Download Summary (TXT)",
                summary,
                f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                "text/plain",
                use_container_width=True
            )


if __name__ == "__main__":
    main()
