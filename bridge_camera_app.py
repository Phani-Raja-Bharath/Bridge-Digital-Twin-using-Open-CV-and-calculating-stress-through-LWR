"""
COMPLETE HYBRID DIGITAL TWIN - FINAL VERSION
With integrated live screenshot monitoring + full analysis pipeline

"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import time
import os
import subprocess
from PIL import Image
import io
import pickle
from scipy import stats
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Constants
TWIN_BRIDGES_SPECS = {
    'name': 'Twin Bridges - I-87 Northway',
    'main_span_ft': 599.8,
    'total_length_ft': 778.9,
    'main_span_m': 182.8,
    'total_length_m': 237.4,
    'age_years': 66,
    'material': 'Steel through arch',
}

TWIN_BRIDGES_CAMERAS = {
    "5821 - North of Mohawk (Twin Bridges)": {
        "url": "https://s51.nysdot.skyvdn.com:443/rtplive/R1_003/playlist.m3u8",
        "approaching_bridge": "right",
        "camera_location": "North of bridge looking south",
        "left_lane_label": "Southbound (PAST LOAD)",
        "right_lane_label": "Northbound (CURRENT LOAD)"
    },
    "3645 - South of Mohawk (Twin Bridges)": {
        "url": "https://s51.nysdot.skyvdn.com:443/rtplive/R1_001/playlist.m3u8",
        "approaching_bridge": "left",
        "camera_location": "South of bridge looking north",
        "left_lane_label": "Northbound (CURRENT LOAD)",
        "right_lane_label": "Southbound (PAST LOAD)"
    },
}

VEHICLE_WEIGHTS = {'car': 4000, 'truck': 35000, 'bus': 25000, 'motorcycle': 500}
VEHICLE_CLASSES = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}


#############################################
# CAMERA & DETECTION FUNCTIONS
#############################################

def capture_frame_ffmpeg(stream_url, timeout=10):
    """Capture single frame from HLS stream"""
    try:
        cmd = ['ffmpeg', '-loglevel', 'error', '-i', stream_url, '-vframes', '1',
               '-f', 'image2pipe', '-vcodec', 'png', '-']
        result = subprocess.run(cmd, capture_output=True, timeout=timeout)
        if result.returncode == 0:
            image = Image.open(io.BytesIO(result.stdout))
            frame = np.array(image)
            if len(frame.shape) == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            return frame
        return None
    except:
        return None


def preprocess_frame(frame, enhance=True):
    """Enhance image for better detection"""
    if not enhance or frame is None:
        return frame
    
    try:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        return enhanced
    except:
        return frame


def define_lane_regions(frame, lane_divider_percent=0.43):
    """Define lane regions with divider"""
    height, width = frame.shape[:2]
    divider_x = int(width * lane_divider_percent)
    
    regions = {
        'left_lane': {'x1': 0, 'y1': 0, 'x2': divider_x, 'y2': height},
        'right_lane': {'x1': divider_x, 'y1': 0, 'x2': width, 'y2': height}
    }
    
    return regions, divider_x


def detect_vehicles_with_overlay(frame, model, regions, camera_config, confidence=0.15, enhance=True):
    """
    Detect vehicles and create visual overlay (like earlier simple dashboard)
    Returns: vehicle_data dict and annotated frame
    """
    if frame is None or model is None:
        return {}, frame
    
    try:
        processed = preprocess_frame(frame, enhance)
        output_frame = frame.copy()
        height, width = frame.shape[:2]
        
        approaching_side = camera_config['approaching_bridge']
        
        vehicle_data = {
            'approaching_bridge': {'car': 0, 'truck': 0, 'bus': 0, 'motorcycle': 0, 'total': 0},
            'leaving_bridge': {'car': 0, 'truck': 0, 'bus': 0, 'motorcycle': 0, 'total': 0},
        }
        
        results = model(processed, conf=confidence, iou=0.45, imgsz=640, 
                       verbose=False, classes=[2, 3, 5, 7])
        
        divider_x = regions['right_lane']['x1']
        
        # Draw lane divider
        cv2.line(output_frame, (divider_x, 0), (divider_x, height), (0, 255, 255), 4)
        
        # Lane labels
        left_label = camera_config['left_lane_label']
        right_label = camera_config['right_lane_label']
        
        cv2.rectangle(output_frame, (5, 5), (450, 65), (0, 0, 0), -1)
        cv2.putText(output_frame, left_label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 2)
        cv2.putText(output_frame, "Twin Bridges I-87 (1959)", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 1)
        
        cv2.rectangle(output_frame, (divider_x + 5, 5), (divider_x + 450, 65), (0, 0, 0), -1)
        cv2.putText(output_frame, right_label, (divider_x + 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(output_frame, "Steel Arch - 66yrs", (divider_x + 10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 1)
        
        # Detect and draw boxes
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                class_id = int(box.cls[0])
                
                if class_id in VEHICLE_CLASSES:
                    vehicle_type = VEHICLE_CLASSES[class_id]
                    
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    center_x = (x1 + x2) // 2
                    
                    # Determine lane
                    if center_x < divider_x:
                        category = 'approaching_bridge' if approaching_side == 'left' else 'leaving_bridge'
                        box_color = (0, 255, 0) if category == 'approaching_bridge' else (150, 150, 150)
                        label_prefix = "LOAD" if category == 'approaching_bridge' else "PAST"
                    else:
                        category = 'approaching_bridge' if approaching_side == 'right' else 'leaving_bridge'
                        box_color = (0, 255, 0) if category == 'approaching_bridge' else (150, 150, 150)
                        label_prefix = "LOAD" if category == 'approaching_bridge' else "PAST"
                    
                    vehicle_data[category][vehicle_type] += 1
                    vehicle_data[category]['total'] += 1
                    
                    # Draw box
                    thickness = 3 if category == 'approaching_bridge' else 1
                    cv2.rectangle(output_frame, (x1, y1), (x2, y2), box_color, thickness)
                    
                    label = f"{label_prefix}-{vehicle_type.upper()[:3]}"
                    cv2.putText(output_frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
        
        # Bottom summary
        approaching_count = vehicle_data['approaching_bridge']['total']
        leaving_count = vehicle_data['leaving_bridge']['total']
        
        summary_height = 100
        cv2.rectangle(output_frame, (0, height - summary_height), (width, height), (0, 0, 0), -1)
        
        # Calculate load
        load_lbs = sum(vehicle_data['approaching_bridge'][v] * VEHICLE_WEIGHTS.get(v, 4000) 
                      for v in ['car', 'truck', 'bus', 'motorcycle'])
        load_tons = load_lbs / 2000
        
        cv2.putText(output_frame,
                    f"CURRENT BRIDGE LOAD: {approaching_count} vehicles, {load_tons:.1f} tons "
                    f"(C:{vehicle_data['approaching_bridge']['car']}, "
                    f"T:{vehicle_data['approaching_bridge']['truck']}, "
                    f"B:{vehicle_data['approaching_bridge']['bus']})",
                    (10, height - 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.putText(output_frame,
                    f"Bridge: Steel Arch, Span: 599.8ft, Age: 66yrs",
                    (10, height - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 255), 2)
        
        cv2.putText(output_frame,
                    f"Past load (leaving): {leaving_count} vehicles (not counted)",
                    (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        # Add load to return data
        vehicle_data['load_tons'] = load_tons
        vehicle_data['load_lbs'] = load_lbs
        vehicle_data['density'] = approaching_count * 5 / TWIN_BRIDGES_SPECS['total_length_m']
        vehicle_data['timestamp'] = datetime.now()
        
        return vehicle_data, output_frame
        
    except Exception as e:
        print(f"Error: {e}")
        return {}, frame


def save_snapshot(frame, snapshot_dir, camera_name):
    """Save snapshot with timestamp"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{camera_name}_{timestamp}.jpg"
    filepath = os.path.join(snapshot_dir, filename)
    cv2.imwrite(filepath, frame)
    return filepath


#############################################
# SIMULATION FUNCTIONS
#############################################

def simulate_lwr_traffic(initial_density, road_length_m, v_max_mps, dt=1.0, total_time=300):
    """LWR traffic flow simulation"""
    dx = 10.0
    num_sections = int(road_length_m / dx)
    num_steps = int(total_time / dt)
    rho_max = 0.2
    
    rho = np.ones(num_sections) * initial_density * rho_max
    rho += np.random.normal(0, 0.01 * rho_max, num_sections)
    rho = np.clip(rho, 0, rho_max)
    
    density_history = [rho.copy()]
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
        rho_new[-1] = rho[-1]
        rho = np.clip(rho_new, 0, rho_max)
        
        # Traffic jam
        if step == num_steps // 2:
            jam_location = num_sections // 2
            jam_width = 10
            rho[jam_location:jam_location + jam_width] = rho_max * 0.9
        
        avg_density = np.mean(rho)
        stress = avg_density / rho_max * 100
        
        density_history.append(rho.copy())
        stress_history.append(stress)
    
    cumulative_stress = np.trapz(stress_history, dx=dt)
    fatigue_score = min(cumulative_stress / 100, 100)
    
    final_density = density_history[-1]
    density_gradient = np.gradient(final_density)
    shockwave_speed = np.mean(np.abs(density_gradient)) * v_max_mps
    
    return {
        'density_history': np.array(density_history),
        'stress_history': np.array(stress_history),
        'final_fatigue': fatigue_score,
        'shockwave_speed': shockwave_speed,
        'avg_density': np.mean(final_density),
        'max_density': np.max(final_density)
    }


def run_monte_carlo_simulation(num_runs=100, road_length_m=237.4, live_density=None):
    """Monte Carlo simulation"""
    results = []
    
    for run in range(num_runs):
        if live_density is not None and run % 5 == 0:
            initial_density = live_density
        else:
            initial_density = np.random.uniform(0.2, 0.6)
        
        v_max_kmh = np.random.choice([40, 60, 80, 100])
        v_max_mps = v_max_kmh / 3.6
        alpha = np.random.uniform(0.00005, 0.001)
        sensor_interval = np.random.randint(100, 300)
        
        sim_result = simulate_lwr_traffic(
            initial_density=initial_density,
            road_length_m=road_length_m,
            v_max_mps=v_max_mps,
            dt=1.0,
            total_time=300
        )
        
        adjusted_fatigue = sim_result['final_fatigue'] * (alpha / 0.0001)
        
        results.append({
            'run': run,
            'initial_density': initial_density,
            'v_max': v_max_kmh,
            'alpha': alpha,
            'sensor_interval': sensor_interval,
            'shockwave_speed': sim_result['shockwave_speed'],
            'final_fatigue': adjusted_fatigue,
            'avg_density': sim_result['avg_density'],
            'max_density': sim_result['max_density'],
            'from_live_data': live_density is not None and run % 5 == 0
        })
    
    return pd.DataFrame(results)


def train_random_forest_model(training_data):
    """Train Random Forest"""
    if not SKLEARN_AVAILABLE:
        return None, {}
    
    features = ['initial_density', 'v_max', 'alpha', 'sensor_interval', 'shockwave_speed']
    target = 'final_fatigue'
    
    X = training_data[features]
    y = training_data[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, max_depth=20, min_samples_split=5, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    metrics = {
        'r2': r2_score(y_test, y_pred),
        'mae': mean_absolute_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'feature_importance': dict(zip(features, model.feature_importances_)),
        'y_test': y_test,
        'y_pred': y_pred
    }
    
    return model, metrics


def predict_fatigue_from_live_data(model, live_traffic, avg_shockwave=0):
    """Predict fatigue from live data"""
    if model is None:
        return None
    
    estimated_density = live_traffic.get('density', live_traffic.get('total', 5) * 5 / TWIN_BRIDGES_SPECS['total_length_m'])
    
    features = {
        'initial_density': estimated_density,
        'v_max': 80,
        'alpha': 0.0005,
        'sensor_interval': 200,
        'shockwave_speed': avg_shockwave
    }
    
    X = pd.DataFrame([features])
    prediction = model.predict(X)[0]
    
    return prediction


#############################################
# VISUALIZATION FUNCTIONS
#############################################

def plot_fatigue_distribution(mc_data, live_fatigue=None, predicted_fatigue=None):
    """Fatigue distribution plot"""
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=mc_data['final_fatigue'],
        name='Monte Carlo Simulations',
        nbinsx=30,
        marker_color='lightblue',
        opacity=0.7
    ))
    
    mu, sigma = mc_data['final_fatigue'].mean(), mc_data['final_fatigue'].std()
    x_range = np.linspace(mc_data['final_fatigue'].min(), mc_data['final_fatigue'].max(), 100)
    y_norm = stats.norm.pdf(x_range, mu, sigma) * len(mc_data) * (mc_data['final_fatigue'].max() - mc_data['final_fatigue'].min()) / 30
    
    fig.add_trace(go.Scatter(
        x=x_range,
        y=y_norm,
        name='Normal Fit',
        line=dict(color='red', width=2, dash='dash')
    ))
    
    if live_fatigue:
        fig.add_vline(x=live_fatigue, line_dash="dash", line_color="green",
                     annotation_text=f"Live: {live_fatigue:.1f}")
    
    if predicted_fatigue:
        fig.add_vline(x=predicted_fatigue, line_dash="dash", line_color="orange",
                     annotation_text=f"Predicted: {predicted_fatigue:.1f}")
    
    fig.update_layout(
        title="Distribution of Fatigue Scores (Monte Carlo Simulation)",
        xaxis_title="Fatigue Score",
        yaxis_title="Frequency",
        template="plotly_white",
        height=500
    )
    
    return fig


def plot_ml_performance(metrics):
    """ML performance visualization"""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Predicted vs Actual", "Feature Importance")
    )
    
    fig.add_trace(
        go.Scatter(
            x=metrics['y_test'],
            y=metrics['y_pred'],
            mode='markers',
            marker=dict(color='blue', size=8, opacity=0.6)
        ),
        row=1, col=1
    )
    
    min_val = min(metrics['y_test'].min(), metrics['y_pred'].min())
    max_val = max(metrics['y_test'].max(), metrics['y_pred'].max())
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            line=dict(color='red', dash='dash')
        ),
        row=1, col=1
    )
    
    features = list(metrics['feature_importance'].keys())
    importance = list(metrics['feature_importance'].values())
    
    fig.add_trace(
        go.Bar(
            x=importance,
            y=features,
            orientation='h',
            marker_color='green'
        ),
        row=1, col=2
    )
    
    fig.update_xaxes(title_text="Actual", row=1, col=1)
    fig.update_yaxes(title_text="Predicted", row=1, col=1)
    fig.update_xaxes(title_text="Importance", row=1, col=2)
    
    fig.update_layout(
        showlegend=False,
        template="plotly_white",
        height=400,
        title_text=f"Random Forest Performance (R² = {metrics['r2']:.3f})"
    )
    
    return fig


def generate_summary_report(live_data, mc_data, metrics, prediction):
    """Generate text report"""
    
    report = f"""
HYBRID DIGITAL TWIN - BRIDGE FATIGUE ASSESSMENT REPORT
Twin Bridges (I-87 over Mohawk River)
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{'='*60}
BRIDGE SPECIFICATIONS
{'='*60}
Name: {TWIN_BRIDGES_SPECS['name']}
Design: {TWIN_BRIDGES_SPECS['material']}
Main Span: {TWIN_BRIDGES_SPECS['main_span_ft']} ft
Age: {TWIN_BRIDGES_SPECS['age_years']} years

{'='*60}
LIVE TRAFFIC OBSERVATION
{'='*60}
"""
    
    if live_data:
        report += f"""Total Vehicles on Bridge: {live_data.get('approaching_bridge', {}).get('total', 'N/A')}
  - Cars: {live_data.get('approaching_bridge', {}).get('car', 'N/A')}
  - Trucks: {live_data.get('approaching_bridge', {}).get('truck', 'N/A')}
  - Buses: {live_data.get('approaching_bridge', {}).get('bus', 'N/A')}
Current Bridge Load: {live_data.get('load_tons', 'N/A'):.2f} tons
Traffic Density: {live_data.get('density', 'N/A'):.3f}
"""
    
    report += f"""
{'='*60}
MONTE CARLO SIMULATION RESULTS
{'='*60}
Number of Simulations: {len(mc_data)}
Mean Fatigue: {mc_data['final_fatigue'].mean():.2f}
Std Dev: {mc_data['final_fatigue'].std():.2f}
Range: {mc_data['final_fatigue'].min():.2f} - {mc_data['final_fatigue'].max():.2f}

{'='*60}
MACHINE LEARNING PREDICTION
{'='*60}
R² Score: {metrics['r2']:.4f}
MAE: {metrics['mae']:.4f}
RMSE: {metrics['rmse']:.4f}

{'='*60}
PREDICTED FATIGUE SCORE
{'='*60}
Score: {prediction:.2f} / 100
"""
    
    if prediction < 60:
        report += "Status: ✅ SAFE OPERATION\n"
    elif prediction < 80:
        report += "Status: ⚠️  MONITOR CLOSELY\n"
    else:
        report += "Status: 🔴 IMMEDIATE INSPECTION REQUIRED\n"
    
    return report


def generate_html_report(live_data, mc_data, metrics, prediction, fig1=None, fig2=None):
    """Generate an HTML report string with optional embedded Plotly figures."""
    def fmt(v, fmt_str="{:.2f}"):
        try:
            return fmt_str.format(v)
        except Exception:
            return str(v)

    # Basic styles
    styles = """
    <style>
      body { font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 24px; }
      h1 { margin-bottom: 0; }
      h2 { margin-top: 28px; }
      .muted { color: #666; }
      .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; }
      .card { border: 1px solid #eee; border-radius: 8px; padding: 12px; }
      table { border-collapse: collapse; width: 100%; }
      th, td { border: 1px solid #eee; padding: 8px; text-align: left; }
      .section { margin-top: 16px; }
    </style>
    """

    # Header
    header = f"""
    <h1>Hybrid Digital Twin - Bridge Fatigue Assessment</h1>
    <div class="muted">Twin Bridges (I-87 over Mohawk River) · Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
    """

    # Summary cards
    live_total = (live_data or {}).get('approaching_bridge', {}).get('total', 'N/A')
    load_tons = (live_data or {}).get('load_tons', 'N/A')
    density = (live_data or {}).get('density', 'N/A')
    r2 = (metrics or {}).get('r2', None)
    mae = (metrics or {}).get('mae', None)
    rmse = (metrics or {}).get('rmse', None)

    cards = f"""
    <div class="grid">
      <div class="card"><div class="muted">Predicted Fatigue</div><div><strong>{fmt(prediction) if prediction is not None else 'N/A'}</strong></div></div>
      <div class="card"><div class="muted">Live Vehicles</div><div><strong>{live_total}</strong></div></div>
      <div class="card"><div class="muted">Current Load (tons)</div><div><strong>{fmt(load_tons) if isinstance(load_tons,(int,float)) else load_tons}</strong></div></div>
      <div class="card"><div class="muted">Traffic Density</div><div><strong>{fmt(density, '{:.3f}') if isinstance(density,(int,float)) else density}</strong></div></div>
      <div class="card"><div class="muted">R²</div><div><strong>{fmt(r2, '{:.3f}') if r2 is not None else 'N/A'}</strong></div></div>
      <div class="card"><div class="muted">MAE</div><div><strong>{fmt(mae, '{:.3f}') if mae is not None else 'N/A'}</strong></div></div>
      <div class="card"><div class="muted">RMSE</div><div><strong>{fmt(rmse, '{:.3f}') if rmse is not None else 'N/A'}</strong></div></div>
    </div>
    """

    # MC stats table
    mc_rows = ""
    if mc_data is not None and hasattr(mc_data, 'mean'):
        mc_rows = f"""
        <tr><th>Simulations</th><td>{len(mc_data)}</td></tr>
        <tr><th>Mean Fatigue</th><td>{fmt(mc_data['final_fatigue'].mean())}</td></tr>
        <tr><th>Std Dev</th><td>{fmt(mc_data['final_fatigue'].std())}</td></tr>
        <tr><th>Range</th><td>{fmt(mc_data['final_fatigue'].min())} – {fmt(mc_data['final_fatigue'].max())}</td></tr>
        """

    mc_section = f"""
    <div class="section">
      <h2>Monte Carlo Summary</h2>
      <table>
        <tbody>
          {mc_rows}
        </tbody>
      </table>
    </div>
    """

    # Embed figures if provided (embed JS for offline viewing)
    figs_html = ""
    try:
        if fig1 is not None:
            figs_html += fig1.to_html(full_html=False, include_plotlyjs=True)
        if fig2 is not None:
            figs_html += fig2.to_html(full_html=False, include_plotlyjs=False)
    except Exception:
        pass

    html = f"""
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1" />
      <title>Bridge Fatigue Assessment Report</title>
      {styles}
    </head>
    <body>
      {header}
      {cards}
      {mc_section}
      <div class="section">
        <h2>Visualizations</h2>
        {figs_html}
      </div>
    </body>
    </html>
    """

    return html


#############################################
# MAIN APPLICATION
#############################################

def main():
    st.set_page_config(page_title="Complete Hybrid Digital Twin", layout="wide")
    
    st.title("🌉 Complete Hybrid Digital Twin - Twin Bridges (I-87)")
    st.markdown("**Live Monitoring + LWR Simulation + Monte Carlo + ML Prediction**")
    
    if not YOLO_AVAILABLE or not SKLEARN_AVAILABLE or not PLOTLY_AVAILABLE:
        st.error("Missing deps. Install: `pip install ultralytics scikit-learn plotly scipy`")
        st.stop()
    
    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    
    mode = st.sidebar.radio(
        "Select Mode:",
        ["🎥 Live Monitoring Dashboard", "📊 Full Analysis Pipeline", "📈 Visualizations & Export"]
    )
    
    # Session state
    if 'monte_carlo_data' not in st.session_state:
        st.session_state.monte_carlo_data = None
    if 'trained_model' not in st.session_state:
        st.session_state.trained_model = None
    if 'model_metrics' not in st.session_state:
        st.session_state.model_metrics = None
    if 'live_traffic_log' not in st.session_state:
        st.session_state.live_traffic_log = []
    if 'latest_prediction' not in st.session_state:
        st.session_state.latest_prediction = None
    if 'monitoring_active' not in st.session_state:
        st.session_state.monitoring_active = False
    if 'current_frame' not in st.session_state:
        st.session_state.current_frame = None
    if 'last_snapshot_time' not in st.session_state:
        st.session_state.last_snapshot_time = 0
    
    #===========================================
    # MODE 1: LIVE MONITORING DASHBOARD
    #===========================================
    if mode == "🎥 Live Monitoring Dashboard":
        st.header("Live Camera Monitoring Dashboard")
        st.markdown("*Real-time vehicle detection with visual overlay (like the simple dashboard)*")
        
        # Camera selection
        selected_camera_name = st.sidebar.selectbox(
            "Select Camera",
            list(TWIN_BRIDGES_CAMERAS.keys())
        )
        
        camera_config = TWIN_BRIDGES_CAMERAS[selected_camera_name]
        
        st.sidebar.subheader("📸 Monitoring Settings")
        snapshot_interval = st.sidebar.number_input("Auto-capture interval (sec)", 5, 300, 30, 5)
        auto_capture = st.sidebar.checkbox("Enable Auto-Capture", value=False)
        
        st.sidebar.subheader("🛣️ Lane Settings")
        lane_divider = st.sidebar.slider("Lane Divider Position", 0.3, 0.7, 0.43, 0.01)
        
        st.sidebar.subheader("🚗 Detection Settings")
        confidence = st.sidebar.slider("Confidence Threshold", 0.05, 0.50, 0.15, 0.05)
        enhance = st.sidebar.checkbox("Enhance Image", value=True)
        
        # Snapshot directory
        snapshot_dir = "bridge_snapshots"
        os.makedirs(snapshot_dir, exist_ok=True)
        
        # Load YOLO
        if 'yolo_model' not in st.session_state:
            with st.spinner("Loading YOLO..."):
                st.session_state.yolo_model = YOLO('yolov8n.pt')
        
        # Controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("▶️ Start", disabled=st.session_state.monitoring_active):
                st.session_state.monitoring_active = True
                st.session_state.last_snapshot_time = time.time()
                st.rerun()
        
        with col2:
            if st.button("⏹️ Stop", disabled=not st.session_state.monitoring_active):
                st.session_state.monitoring_active = False
                st.rerun()
        
        with col3:
            manual_snapshot = st.button("📸 Manual Snapshot", disabled=not st.session_state.monitoring_active)
        
        # Display area
        col_video, col_stats = st.columns([2, 1])
        
        with col_video:
            st.subheader("📹 Live Feed")
            video_placeholder = st.empty()
            status_placeholder = st.empty()
        
        with col_stats:
            st.subheader("⚖️ Current Bridge Load")
            load_container = st.container()
            
            st.subheader("📊 Statistics")
            stats_container = st.container()
        
        # Data table
        st.subheader("📈 Monitoring Log")
        data_placeholder = st.empty()
        
        # Main monitoring loop
        if st.session_state.monitoring_active:
            current_time = time.time()
            time_since = current_time - st.session_state.last_snapshot_time
            should_capture = auto_capture and (time_since >= snapshot_interval)
            
            if should_capture or manual_snapshot or st.session_state.current_frame is None:
                status_placeholder.info("🔄 Capturing & analyzing...")
                
                # Capture frame
                stream_url = camera_config['url']
                frame = capture_frame_ffmpeg(stream_url, timeout=15)
                
                if frame is not None:
                    st.session_state.current_frame = frame
                    
                    # Detect vehicles
                    regions, _ = define_lane_regions(frame, lane_divider)
                    
                    vehicle_data, annotated_frame = detect_vehicles_with_overlay(
                        frame,
                        st.session_state.yolo_model,
                        regions,
                        camera_config,
                        confidence,
                        enhance
                    )
                    
                    # Display annotated frame
                    frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    video_placeholder.image(frame_rgb, channels="RGB", width='stretch')
                    
                    # Save snapshot
                    if should_capture or manual_snapshot:
                        camera_short = selected_camera_name.split()[0]
                        snapshot_path = save_snapshot(annotated_frame, snapshot_dir, camera_short)
                        
                        # Log data
                        st.session_state.live_traffic_log.append(vehicle_data)
                        
                        if should_capture:
                            st.session_state.last_snapshot_time = current_time
                        
                        # Update stats
                        with load_container:
                            st.metric("Vehicles on Bridge", 
                                     vehicle_data['approaching_bridge']['total'])
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.metric("Cars", vehicle_data['approaching_bridge']['car'])
                                st.metric("Buses", vehicle_data['approaching_bridge']['bus'])
                            with col_b:
                                st.metric("Trucks", vehicle_data['approaching_bridge']['truck'])
                                st.metric("Motorcycles", vehicle_data['approaching_bridge']['motorcycle'])
                            
                            st.metric("**Bridge Load**", f"**{vehicle_data['load_tons']:.1f} tons**")
                        
                        with stats_container:
                            if st.session_state.live_traffic_log:
                                df_log = pd.DataFrame(st.session_state.live_traffic_log)
                                st.metric("Total Snapshots", len(st.session_state.live_traffic_log))
                                st.metric("Avg Load", f"{df_log['load_tons'].mean():.1f} tons")
                        
                        status_placeholder.success(f"✅ Snapshot saved")
                    else:
                        status_placeholder.success(f"✅ Live - {datetime.now().strftime('%H:%M:%S')}")
                else:
                    status_placeholder.error("❌ Capture failed")
            
            # Show log
            if st.session_state.live_traffic_log:
                df_log = pd.DataFrame(st.session_state.live_traffic_log)
                # Flatten nested dicts for display
                display_df = pd.DataFrame({
                    'Timestamp': [d['timestamp'].strftime('%H:%M:%S') for d in st.session_state.live_traffic_log],
                    'Vehicles': [d['approaching_bridge']['total'] for d in st.session_state.live_traffic_log],
                    'Cars': [d['approaching_bridge']['car'] for d in st.session_state.live_traffic_log],
                    'Trucks': [d['approaching_bridge']['truck'] for d in st.session_state.live_traffic_log],
                    'Load_Tons': [d['load_tons'] for d in st.session_state.live_traffic_log],
                })
                data_placeholder.dataframe(display_df, width='stretch', height=300)
            
            time.sleep(2)
            st.rerun()
        
        else:
            status_placeholder.info("⏸️ Monitoring stopped")
            
            if st.session_state.current_frame is not None:
                frame_rgb = cv2.cvtColor(st.session_state.current_frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(frame_rgb, channels="RGB", width='stretch')
            
            if st.session_state.live_traffic_log:
                display_df = pd.DataFrame({
                    'Timestamp': [d['timestamp'].strftime('%H:%M:%S') for d in st.session_state.live_traffic_log],
                    'Vehicles': [d['approaching_bridge']['total'] for d in st.session_state.live_traffic_log],
                    'Load_Tons': [d['load_tons'] for d in st.session_state.live_traffic_log],
                })
                data_placeholder.dataframe(display_df, width='stretch', height=300)
        
        # Export button
        if st.session_state.live_traffic_log:
            st.markdown("---")
            display_df = pd.DataFrame({
                'Timestamp': [d['timestamp'] for d in st.session_state.live_traffic_log],
                'Vehicles': [d['approaching_bridge']['total'] for d in st.session_state.live_traffic_log],
                'Load_Tons': [d['load_tons'] for d in st.session_state.live_traffic_log],
                'Density': [d['density'] for d in st.session_state.live_traffic_log],
            })
            csv = display_df.to_csv(index=False)
            st.download_button(
                "📥 Download Monitoring Log (CSV)",
                csv,
                f"monitoring_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv"
            )
    
    #===========================================
    # MODE 2: FULL ANALYSIS PIPELINE
    #===========================================
    elif mode == "📊 Full Analysis Pipeline":
        st.header("Complete Analysis Pipeline")
        
        st.markdown("""
        **Integrated Workflow:**
        1. Capture live traffic (with visual overlay)
        2. Run Monte Carlo simulations (LWR model)
        3. Train Random Forest ML model
        4. Predict fatigue score
        5. Generate visualizations & reports
        """)
        
        num_runs = st.sidebar.slider("Monte Carlo Runs", 50, 500, 100)
        use_live = st.sidebar.checkbox("Use Live Traffic Data", value=True)
        
        if st.button("🚀 Run Complete Analysis", type="primary"):
            
            progress = st.progress(0)
            status = st.empty()
            
            # Step 1: Capture
            status.text("Step 1/5: Capturing live traffic...")
            progress.progress(10)
            
            if 'yolo_model' not in st.session_state:
                st.session_state.yolo_model = YOLO('yolov8n.pt')
            
            live_data = None
            
            if use_live:
                selected_camera = list(TWIN_BRIDGES_CAMERAS.keys())[0]
                camera_config = TWIN_BRIDGES_CAMERAS[selected_camera]
                stream_url = camera_config['url']
                frame = capture_frame_ffmpeg(stream_url)
                
                if frame is not None and getattr(frame, 'size', 0) > 0:
                    regions, _ = define_lane_regions(frame)
                    live_data, annotated = detect_vehicles_with_overlay(
                        frame, st.session_state.yolo_model, regions, camera_config
                    )
                    
                    st.success(f"✅ Captured: {live_data['approaching_bridge']['total']} vehicles, "
                              f"{live_data['load_tons']:.1f} tons")
                    
                    # Show image
                    frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    st.image(frame_rgb, caption="Live Traffic Snapshot", width='stretch')
            
            progress.progress(25)
            
            # Step 2: Monte Carlo
            status.text(f"Step 2/5: Running {num_runs} Monte Carlo simulations...")
            
            live_density = live_data['density'] if live_data else None
            mc_data = run_monte_carlo_simulation(num_runs, TWIN_BRIDGES_SPECS['total_length_m'], live_density)
            st.session_state.monte_carlo_data = mc_data
            
            st.success(f"✅ MC complete. Mean fatigue: {mc_data['final_fatigue'].mean():.2f}")
            progress.progress(50)
            
            # Step 3: Train ML
            status.text("Step 3/5: Training Random Forest...")
            
            model, metrics = train_random_forest_model(mc_data)
            st.session_state.trained_model = model
            st.session_state.model_metrics = metrics
            
            st.success(f"✅ Model trained. R² = {metrics['r2']:.3f}")
            progress.progress(75)
            
            # Step 4: Predict
            status.text("Step 4/5: Predicting fatigue...")
            
            if live_data:
                prediction = predict_fatigue_from_live_data(model, live_data, mc_data['shockwave_speed'].mean())
            else:
                prediction = mc_data['final_fatigue'].mean()
            
            st.session_state.latest_prediction = prediction
            progress.progress(90)
            
            # Step 5: Visualize
            status.text("Step 5/5: Generating visualizations...")
            progress.progress(100)
            
            time.sleep(0.5)
            progress.empty()
            status.empty()
            
            # Results
            st.markdown("---")
            st.header("📊 Analysis Results")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Live Vehicles", live_data['approaching_bridge']['total'] if live_data else "N/A")
            with col2:
                st.metric("MC Mean", f"{mc_data['final_fatigue'].mean():.2f}")
            with col3:
                st.metric("ML Prediction", f"{prediction:.2f}")
            with col4:
                if prediction < 60:
                    st.success("✅ Safe")
                elif prediction < 80:
                    st.warning("⚠️ Monitor")
                else:
                    st.error("🔴 Critical")
            
            # Visualizations
            st.subheader("📈 Visualizations")
            
            tab1, tab2 = st.tabs(["Fatigue Distribution", "ML Performance"])
            
            with tab1:
                fig1 = plot_fatigue_distribution(mc_data, None, prediction)
                st.plotly_chart(fig1)
            
            with tab2:
                fig2 = plot_ml_performance(metrics)
                st.plotly_chart(fig2)
            
            # Export
            st.markdown("---")
            st.subheader("💾 Export Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                csv_mc = mc_data.to_csv(index=False)
                st.download_button(
                    "📥 Monte Carlo Data (CSV)",
                    csv_mc,
                    f"monte_carlo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv"
                )
            
            with col2:
                report = generate_summary_report(live_data, mc_data, metrics, prediction)
                st.download_button(
                    "📄 Summary Report (TXT)",
                    report,
                    f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    "text/plain"
                )
            
            pass  # removed celebratory balloons
    
    #===========================================
    # MODE 3: VISUALIZATIONS & EXPORT
    #===========================================
    else:
        st.header("Visualizations & Export")
        
        if st.session_state.monte_carlo_data is None:
            st.warning("⚠️ Run Full Analysis first")
        else:
            mc_data = st.session_state.monte_carlo_data
            metrics = st.session_state.model_metrics
            prediction = st.session_state.latest_prediction
            
            # Show all visualizations
            st.subheader("1. Fatigue Distribution")
            fig1 = plot_fatigue_distribution(mc_data, None, prediction)
            st.plotly_chart(fig1)
            
            if metrics:
                st.subheader("2. ML Performance")
                fig2 = plot_ml_performance(metrics)
                st.plotly_chart(fig2)
            
            # Data tables
            st.subheader("3. Data Tables")
            
            tab1, tab2 = st.tabs(["Monte Carlo Results", "Monitoring Log"])
            
            with tab1:
                st.dataframe(mc_data, width='stretch')
            
            with tab2:
                if st.session_state.live_traffic_log:
                    display_df = pd.DataFrame({
                        'Timestamp': [d['timestamp'] for d in st.session_state.live_traffic_log],
                        'Vehicles': [d['approaching_bridge']['total'] for d in st.session_state.live_traffic_log],
                        'Load_Tons': [d['load_tons'] for d in st.session_state.live_traffic_log],
                    })
                    st.dataframe(display_df, width='stretch')
                else:
                    st.info("No monitoring data available")


if __name__ == "__main__":
    main()
