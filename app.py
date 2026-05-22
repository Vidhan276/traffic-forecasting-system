"""
app.py — Traffic Forecast Dashboard (Streamlit Thin Client)
===========================================================
Consumes the FastAPI backend at http://127.0.0.1:8000.
All heavy computation (GNN inference, routing) is done by the API.

Run with:
    py -3.13 -m streamlit run app.py

The FastAPI backend must be running first:
    py -3.13 -m uvicorn api.main:app --port 8000
"""

from __future__ import annotations

import os
import sys
import pickle
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Pune Traffic Operations",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE = os.getenv("TRAFFIC_API_URL", "http://127.0.0.1:8000")

# ═════════════════════════════════════════════════════════════════════════════
# PREMIUM CSS — Dark Glassmorphism Theme
# ═════════════════════════════════════════════════════════════════════════════
PREMIUM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Reset & base ─────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', sans-serif !important;
    background-color: #0A0F1C !important;
    color: #E8EAED !important;
}

/* ── Hide Streamlit chrome ────────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Main container ───────────────────────────────────── */
.block-container {
    padding-top: 1rem !important;
    padding-bottom: 1rem !important;
    max-width: 1440px !important;
}

/* ── Sidebar ──────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D1321 0%, #0A0F1C 100%) !important;
    border-right: 1px solid rgba(255,255,255,0.08) !important;
}
section[data-testid="stSidebar"] .block-container {
    padding-top: 1.5rem !important;
}

/* ── Tab bar ──────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0px;
    background: rgba(255,255,255,0.03);
    border-radius: 12px 12px 0 0;
    border: 1px solid rgba(255,255,255,0.08);
    border-bottom: none;
    padding: 4px 4px 0;
}
.stTabs [data-baseweb="tab"] {
    font-weight: 500;
    font-size: 0.93rem;
    padding: 12px 22px;
    color: #9AA0A6;
    background: transparent;
    border-radius: 10px 10px 0 0;
    border-bottom: 3px solid transparent;
    transition: all 0.2s ease;
}
.stTabs [aria-selected="true"] {
    color: #1A73E8 !important;
    background: rgba(26,115,232,0.08) !important;
    border-bottom: 3px solid #1A73E8 !important;
    font-weight: 600;
}
.stTabs [data-baseweb="tab"]:hover { color: #BDC1C6; background: rgba(255,255,255,0.04); }

/* ── Glass card ───────────────────────────────────────── */
.glass-card {
    background: rgba(255, 255, 255, 0.04);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.09);
    border-radius: 16px;
    padding: 24px 22px;
    margin-bottom: 16px;
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
}
.glass-card:hover {
    border-color: rgba(26,115,232,0.4);
    box-shadow: 0 0 20px rgba(26,115,232,0.12);
}

/* ── Route option cards ───────────────────────────────── */
.route-card {
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 18px 20px;
    margin-bottom: 12px;
    cursor: pointer;
    transition: all 0.2s ease;
    position: relative;
}
.route-card:hover {
    background: rgba(255,255,255,0.07);
    transform: translateX(3px);
}
.route-card.active {
    border-color: #1A73E8;
    box-shadow: 0 0 0 1px #1A73E8, 0 4px 20px rgba(26,115,232,0.2);
}
.route-card .route-accent {
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 4px;
    border-radius: 14px 0 0 14px;
}
.route-card .route-label {
    font-size: 0.95rem;
    font-weight: 700;
    color: #E8EAED;
    margin-bottom: 10px;
    padding-left: 8px;
}
.route-card .eta-row {
    display: flex;
    gap: 10px;
    align-items: center;
    margin-bottom: 10px;
    padding-left: 8px;
}
.eta-badge {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 8px;
    padding: 6px 14px;
    font-size: 0.82rem;
    color: #BDC1C6;
}
.eta-badge .eta-val {
    font-size: 1.3rem;
    font-weight: 700;
    color: #E8EAED;
    display: block;
}
.eta-badge .eta-lbl {
    font-size: 0.72rem;
    color: #9AA0A6;
}

/* ── Congestion bar ───────────────────────────────────── */
.cong-bar-bg {
    height: 6px;
    background: rgba(255,255,255,0.08);
    border-radius: 3px;
    overflow: hidden;
    margin: 6px 8px 2px;
}
.cong-bar-fill {
    height: 100%;
    border-radius: 3px;
    transition: width 0.4s ease;
}

/* ── Info rows ────────────────────────────────────────── */
.info-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 6px 0;
    font-size: 0.85rem;
    border-bottom: 1px solid rgba(255,255,255,0.05);
}
.info-row:last-child { border-bottom: none; }
.info-key { color: #9AA0A6; }
.info-val { color: #E8EAED; font-weight: 600; }

/* ── Metric cards ─────────────────────────────────────── */
.metric-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 22px 18px;
    text-align: center;
}
.metric-card .metric-icon { font-size: 2rem; margin-bottom: 4px; }
.metric-card .metric-label {
    font-size: 0.75rem;
    font-weight: 600;
    color: #9AA0A6;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 8px;
}
.metric-card .metric-value {
    font-size: 2rem;
    font-weight: 800;
    color: #1A73E8;
    line-height: 1.1;
}
.metric-card .metric-desc {
    font-size: 0.74rem;
    color: #5F6368;
    margin-top: 6px;
}

/* ── Title bar ────────────────────────────────────────── */
.title-bar {
    background: linear-gradient(135deg, #1A1F35 0%, #0D1321 100%);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 20px 28px;
    margin-bottom: 20px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.title-bar h1 {
    margin: 0;
    font-size: 1.5rem;
    font-weight: 700;
    color: #E8EAED;
    background: linear-gradient(90deg, #1A73E8, #34A853);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.title-bar .subtitle { font-size: 0.85rem; color: #9AA0A6; margin-top: 3px; }
.title-bar .time-badge {
    background: rgba(26,115,232,0.12);
    border: 1px solid rgba(26,115,232,0.3);
    border-radius: 10px;
    padding: 8px 16px;
    text-align: right;
}
.title-bar .time-badge .time-main { font-size: 1.4rem; font-weight: 700; color: #1A73E8; }
.title-bar .time-badge .time-sub { font-size: 0.78rem; color: #9AA0A6; }

/* ── Section title ────────────────────────────────────── */
.section-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: #E8EAED;
    margin-bottom: 4px;
}
.section-sub { font-size: 0.85rem; color: #9AA0A6; margin-bottom: 18px; }

/* ── Traffic legend ───────────────────────────────────── */
.legend-strip {
    display: flex;
    border-radius: 10px;
    overflow: hidden;
    margin: 14px 0;
    height: 32px;
}
.legend-part {
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.72rem;
    font-weight: 600;
    color: #fff;
}

/* ── API status dot ───────────────────────────────────── */
.api-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    display: inline-block;
    margin-right: 6px;
}
.api-ok { background: #34A853; box-shadow: 0 0 6px #34A853; }
.api-err { background: #EA4335; box-shadow: 0 0 6px #EA4335; }

/* ── Divider ──────────────────────────────────────────── */
.divider { border: none; border-top: 1px solid rgba(255,255,255,0.06); margin: 20px 0; }

/* ── Map container ────────────────────────────────────── */
.map-wrapper {
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    overflow: hidden;
}

/* ── Buttons ──────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #1A73E8 0%, #1557B0 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 12px 28px !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    width: 100% !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #1765CC 0%, #1050A0 100%) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(26,115,232,0.35) !important;
}

/* ── Selectbox ────────────────────────────────────────── */
.stSelectbox > div > div {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 10px !important;
    color: #E8EAED !important;
}

/* ── Warning / error boxes ─────────────────────────────── */
.api-warning {
    background: rgba(234,67,53,0.1);
    border: 1px solid rgba(234,67,53,0.3);
    border-radius: 10px;
    padding: 14px 18px;
    color: #EA4335;
    font-size: 0.88rem;
}

/* ── Stitch-inspired route map legend ─────────────────── */
.map-legend {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px;
    padding: 12px 16px;
    margin-top: 8px;
    display: flex;
    gap: 16px;
    flex-wrap: wrap;
    font-size: 0.8rem;
}
.legend-item { display: flex; align-items: center; gap: 6px; color: #BDC1C6; }
.legend-dot { width:12px; height:12px; border-radius:50%; }

[data-testid="stMetricValue"] { color: #1A73E8 !important; font-weight: 700 !important; }
[data-testid="stMetricLabel"] { color: #9AA0A6 !important; }
</style>
"""

st.markdown(PREMIUM_CSS, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=5, show_spinner=False)
def _api_get(path: str, timeout: float = 5.0) -> dict | None:
    try:
        r = httpx.get(f"{API_BASE}{path}", timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def _api_post(path: str, body: dict, timeout: float = 30.0) -> dict | None:
    try:
        r = httpx.post(f"{API_BASE}{path}", json=body, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        st.error(f"API error {e.response.status_code}: {e.response.text}")
        return None
    except Exception as e:
        st.error(f"Could not reach the backend API: {e}")
        return None


@st.cache_data(show_spinner=False)
def _load_pickle(path: Path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def _load_npy(path: Path):
    try:
        return np.load(str(path))
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def _read_html(path: Path) -> str | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return None


def _congestion_color(score: float) -> str:
    if score < 0.25:   return "#34A853"
    elif score < 0.50: return "#FBBC04"
    elif score < 0.75: return "#F9AB00"
    else:              return "#EA4335"


def _congestion_label(score: float) -> str:
    if score < 0.25:   return "Free Flow 🟢"
    elif score < 0.50: return "Light 🟡"
    elif score < 0.75: return "Moderate 🟠"
    else:              return "Heavy 🔴"


@st.cache_data(ttl=300, show_spinner=False)
def _build_route_map_html(route_json: str, selected_label: str | None) -> str:
    import folium

    res = json.loads(route_json)
    routes = res.get("routes", [])

    m = folium.Map(
        location=[18.52, 73.855],
        zoom_start=13,
        tiles="cartodbdark_matter",
        prefer_canvas=True,
    )

    title_html = f"""
    <div style="position:fixed; top:10px; left:50%; transform:translateX(-50%);
                z-index:9999; background:rgba(10,15,28,0.85);
                backdrop-filter:blur(8px); padding:8px 18px;
                border:1px solid rgba(255,255,255,0.15); border-radius:8px;
                color:#E8EAED; font-family:Inter,sans-serif; font-size:13px; font-weight:600;">
        {res['origin']} to {res['destination']}
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))

    o_loc = None
    d_loc = None
    for r in routes:
        is_selected = r["label"] == selected_label
        opacity = 0.95 if is_selected else 0.45
        weight = 7 if is_selected else 4

        for seg in r["segments"]:
            coords = seg.get("coords") or []
            if coords:
                folium.PolyLine(
                    locations=coords,
                    color=r["color"] if is_selected else seg["color"],
                    weight=weight,
                    opacity=opacity,
                    tooltip=f"{r['label']}: {seg['road_name']} ({int(seg['congestion'] * 100)}% congestion)",
                ).add_to(m)

        if r["label"] == (routes[0]["label"] if routes else None):
            all_coords = [c for seg in r["segments"] for c in seg.get("coords", [])]
            if all_coords:
                o_loc = all_coords[0]
                d_loc = all_coords[-1]

    if o_loc:
        folium.Marker(o_loc, tooltip=f"Origin: {res['origin']}",
                      icon=folium.Icon(color="blue", icon="play", prefix="fa")).add_to(m)
    if d_loc:
        folium.Marker(d_loc, tooltip=f"Destination: {res['destination']}",
                      icon=folium.Icon(color="red", icon="flag", prefix="fa")).add_to(m)

    legend_html = """
    <div style="position:fixed; bottom:20px; left:20px; z-index:9999;
                background:rgba(10,15,28,0.85); backdrop-filter:blur(8px);
                border:1px solid rgba(255,255,255,0.12); border-radius:10px;
                padding:12px 16px; font-family:Inter,sans-serif; color:#E8EAED;">
      <div style="font-size:11px; font-weight:600; color:#9AA0A6;
                  text-transform:uppercase; letter-spacing:0.8px; margin-bottom:8px;">Routes</div>
      <div style="display:flex; align-items:center; gap:8px; margin-bottom:6px; font-size:12px;">
        <span style="width:18px;height:4px;background:#1A73E8;border-radius:2px;display:inline-block;"></span>Fastest Now
      </div>
      <div style="display:flex; align-items:center; gap:8px; margin-bottom:6px; font-size:12px;">
        <span style="width:18px;height:4px;background:#34A853;border-radius:2px;display:inline-block;"></span>Fastest 15-min
      </div>
      <div style="display:flex; align-items:center; gap:8px; font-size:12px;">
        <span style="width:18px;height:4px;background:#F9AB00;border-radius:2px;display:inline-block;"></span>Balanced
      </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    return m._repr_html_()


# ═════════════════════════════════════════════════════════════════════════════
# PATHS
# ═════════════════════════════════════════════════════════════════════════════
CURRENT_MAP   = BASE_DIR / "pune_current_traffic.html"
PREDICTED_MAP = BASE_DIR / "pune_predicted_traffic.html"
FALLBACK_MAP  = BASE_DIR / "kothrud_traffic_map.html"
FORECAST_IMG  = BASE_DIR / "visualization" / "forecast_comparison.png"
HISTORY_PATH  = BASE_DIR / "ml" / "training_history.pkl"
TEST_METRICS  = BASE_DIR / "ml" / "test_metrics.pkl"
TRAFFIC_DATA  = BASE_DIR / "data" / "traffic_data.npy"
ABLATION_PATH = BASE_DIR / "ml" / "ablation_results.json"

PUNE_LOCATIONS = [
    "Shivajinagar", "Viman Nagar", "Hinjewadi", "Katraj",
    "Koregaon Park", "Kothrud", "Hadapsar", "Baner", "Swargate", "Nal Stop",
]


# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    # Logo
    st.markdown("""
    <div style="text-align:center; padding:16px 0 20px;">
        <div style="font-size:1.25rem; font-weight:800; color:#E8EAED; margin-top:4px;">
            Pune Traffic Operations
        </div>
        <div style="font-size:0.8rem; color:#9AA0A6;">Live road conditions and route planning</div>
    </div>
    """, unsafe_allow_html=True)

    # API status
    health = _api_get("/health", timeout=3.0)
    if health:
        st.markdown(f"""
        <div class="glass-card" style="padding:14px 18px; margin-bottom:12px;">
            <div style="font-size:0.72rem; font-weight:600; color:#9AA0A6; text-transform:uppercase; letter-spacing:0.8px; margin-bottom:10px;">API Status</div>
            <div style="display:flex; align-items:center; font-size:0.85rem;">
                <span class="api-dot api-ok"></span>
                <span style="color:#34A853; font-weight:600;">Connected</span>
            </div>
            <div class="info-row" style="margin-top:8px;">
                <span class="info-key">Forecast</span>
                <span class="info-val">Available</span>
            </div>
            <div class="info-row">
                <span class="info-key">Road points</span>
                <span class="info-val">{health.get('num_nodes', '—'):,}</span>
            </div>
            <div class="info-row">
                <span class="info-key">Service</span>
                <span class="info-val">Online</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="api-warning">
            <span class="api-dot api-err"></span>
            <strong>API offline.</strong><br>
            Start the backend:<br>
            <code>venv\\Scripts\\python.exe -m uvicorn api.main:app --port 8000</code>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card" style="padding:14px 18px;">
        <div style="font-size:0.72rem; font-weight:600; color:#9AA0A6; text-transform:uppercase; letter-spacing:0.8px; margin-bottom:10px;">Planning Window</div>
        <div class="info-row"><span class="info-key">Current view</span><span class="info-val">Now</span></div>
        <div class="info-row"><span class="info-key">Forecast view</span><span class="info-val">15 min ahead</span></div>
        <div class="info-row"><span class="info-key">Coverage</span><span class="info-val">Pune road network</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ═════════════════════════════════════════════════════════════════════════════
# Indian Standard Time (IST) offset is +5:30
IST = timezone(timedelta(hours=5, minutes=30))
now = datetime.now(IST)
pred_t = now + timedelta(minutes=15)

st.markdown(f"""
<div class="title-bar">
    <div>
        <h1>Pune Traffic Operations</h1>
        <div class="subtitle">Current road conditions and 15-minute route outlook</div>
    </div>
    <div class="time-badge">
        <div class="time-main" id="live-clock">{now.strftime('%I:%M %p')}</div>
        <div class="time-sub" id="live-date">{now.strftime('%a, %d %b %Y')}</div>
    </div>
</div>
<img src="x" onerror="if(!window.clockIntervalSet){{window.clockIntervalSet=true;const updateClock=()=>{{const clock=document.getElementById('live-clock');const dateEl=document.getElementById('live-date');if(clock&&dateEl){{const now=new Date();const formatter=new Intl.DateTimeFormat('en-US',{{timeZone:'Asia/Kolkata',weekday:'short',day:'numeric',month:'short',year:'numeric',hour:'2-digit',minute:'2-digit',hour12:true}});const parts=formatter.formatToParts(now);const partMap={{}};parts.forEach(p=>partMap[p.type]=p.value);let hour=partMap.hour;if(hour.length===1){{hour='0'+hour;}}const timeStr=hour+':'+partMap.minute+' '+partMap.dayPeriod;const dateStr=partMap.weekday+', '+partMap.day+' '+partMap.month+' '+partMap.year;clock.textContent=timeStr;dateEl.textContent=dateStr;}}}};setInterval(updateClock,1000);updateClock();}}" style="display:none;">
""", unsafe_allow_html=True)

tab_map, tab_route, tab_perf, tab_forecast = st.tabs([
    "Traffic Map",
    "Route Planner",
    "System Insights",
    "Traffic Trends",
])


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1: TRAFFIC MAP
# ═════════════════════════════════════════════════════════════════════════════
with tab_map:
    st.markdown("""
    <div class="section-title">Live Traffic Overview</div>
    <div class="section-sub">Current conditions compared with the 15-minute outlook</div>
    """, unsafe_allow_html=True)

    map_choice = st.radio(
        "Map view",
        ["Current traffic", f"Expected at {pred_t.strftime('%I:%M %p')}"],
        horizontal=True,
        label_visibility="collapsed",
    )
    load_map = st.button("Load Traffic Map", key="load_traffic_map")

    if load_map or st.session_state.get("load_traffic_map"):
        map_path = "/maps/current" if map_choice == "Current traffic" else "/maps/predicted"
        map_url = f"{API_BASE}{map_path}"
        
        st.markdown('<div class="map-wrapper">', unsafe_allow_html=True)
        components.iframe(src=map_url, height=540, scrolling=False)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("Full city maps are loaded on demand to keep the dashboard responsive.")

    # Legend
    st.markdown("""
    <div class="legend-strip">
        <div class="legend-part" style="background:#34A853;">● Free Flow</div>
        <div class="legend-part" style="background:#FBBC04; color:#222;">● Light</div>
        <div class="legend-part" style="background:#F9AB00; color:#222;">● Moderate</div>
        <div class="legend-part" style="background:#EA4335;">● Heavy</div>
        <div class="legend-part" style="background:#C5221F;">● Severe</div>
    </div>
    <div style="text-align:center; font-size:0.8rem; color:#9AA0A6; margin-top:4px;">
        📍 Current: <strong>{now}</strong> &nbsp;|&nbsp; 🔮 Forecast: <strong>{pred}</strong> (+15 min)
    </div>
    """.format(now=now.strftime("%I:%M %p"), pred=pred_t.strftime("%I:%M %p")),
        unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2: ROUTE PLANNER
# ═════════════════════════════════════════════════════════════════════════════
with tab_route:
    st.markdown("""
    <div class="section-title">Route Planner</div>
    <div class="section-sub">
        Compare travel time if you leave now with the estimated time if you leave in 15 minutes.
    </div>
    """, unsafe_allow_html=True)

    left_col, right_col = st.columns([4, 7])

    with left_col:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("**Origin**")
        origin = st.selectbox("Origin", PUNE_LOCATIONS, index=5, key="origin",
                              label_visibility="collapsed")
        st.markdown("**Destination**")
        dest   = st.selectbox("Destination", PUNE_LOCATIONS, index=2, key="dest",
                              label_visibility="collapsed")

        if origin == dest:
            st.warning("Origin and destination must be different.")

        find_btn = st.button("Find Routes", disabled=(origin == dest or health is None))
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Results ──
        if find_btn and origin != dest and health:
            with st.spinner("Checking current and expected traffic..."):
                result = _api_post("/routes", {
                    "origin": origin,
                    "destination": dest,
                    "horizon_minutes": 15,
                }, timeout=60)

            if result:
                st.session_state["route_result"] = result
                st.session_state["route_map_html"] = None  # will build below

        if "route_result" in st.session_state:
            res = st.session_state["route_result"]
            routes = res.get("routes", [])

            latency = res.get("latency_ms", 0)
            st.markdown(f"""
            <div style="font-size:0.75rem; color:#9AA0A6; margin-bottom:10px;">
                {len(routes)} routes found &nbsp;·&nbsp; {latency:.0f} ms
            </div>
            """, unsafe_allow_html=True)

            selected_label = st.session_state.get("selected_route", None)

            for r in routes:
                label   = r["label"]
                color   = r["color"]
                cong    = r["congestion_score"]
                cong_pct = int(cong * 100)
                cong_col = _congestion_color(cong)
                cong_lbl = _congestion_label(cong)
                eta_now = float(r["eta_now_min"])
                eta_future = float(r["eta_future_min"])
                eta_delta = eta_future - eta_now
                if eta_delta > 0.05:
                    eta_note = f"+{eta_delta:.1f} min"
                    eta_color = "#EA4335"
                elif eta_delta < -0.05:
                    eta_note = f"{eta_delta:.1f} min"
                    eta_color = "#34A853"
                else:
                    eta_note = "No change"
                    eta_color = "#BDC1C6"
                is_selected = (label == selected_label)
                card_class = "route-card active" if is_selected else "route-card"

                st.markdown(f"""
                <div class="{card_class}">
                    <div class="route-accent" style="background:{color};"></div>
                    <div class="route-label" style="color:{color};">{label}</div>
                    <div class="eta-row">
                        <div class="eta-badge">
                            <span class="eta-val">{eta_now:.1f} min</span>
                            <span class="eta-lbl">Leave now</span>
                        </div>
                        <div style="color:#9AA0A6; font-size:1rem;">→</div>
                        <div class="eta-badge" style="border-color:{eta_color};">
                            <span class="eta-val" style="color:{eta_color};">{eta_future:.1f} min</span>
                            <span class="eta-lbl">Leave in 15 min</span>
                        </div>
                    </div>
                    <div class="cong-bar-bg">
                        <div class="cong-bar-fill" style="width:{cong_pct}%; background:{cong_col};"></div>
                    </div>
                    <div style="display:flex; justify-content:space-between; padding:2px 8px; font-size:0.76rem; color:#9AA0A6;">
                        <span>{cong_lbl}</span>
                        <span style="color:{eta_color};">{eta_note}</span>
                        <span>{float(r['distance_km']):.2f} km</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Invisible button to select route
                if st.button(f"Select {label}", key=f"sel_{label}",
                             help=f"View {label} on map"):
                    st.session_state["selected_route"] = label
                    st.rerun()

    # ── Right: Map ──────────────────────────────────────────────────────────
    with right_col:
        if "route_result" in st.session_state:
            res    = st.session_state["route_result"]
            routes = res.get("routes", [])
            sel    = st.session_state.get("selected_route", routes[0]["label"] if routes else None)
            route_json = json.dumps(res, sort_keys=True, separators=(",", ":"))
            map_html = _build_route_map_html(route_json, sel)
            st.markdown('<div class="map-wrapper">', unsafe_allow_html=True)
            components.html(map_html, height=580, scrolling=False)
            st.markdown('</div>', unsafe_allow_html=True)

        else:
            st.markdown("""
            <div style="height:580px; display:flex; align-items:center; justify-content:center;
                        background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.06);
                        border-radius:14px; color:#5F6368; text-align:center;">
                <div>
                    <div style="font-size:3rem; margin-bottom:12px;">🗺️</div>
                    <div style="font-size:1rem; font-weight:600; color:#9AA0A6;">Select origin and destination</div>
                    <div style="font-size:0.85rem; color:#5F6368; margin-top:6px;">
                        Click "Find Routes" to compare route options
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3: SYSTEM INSIGHTS
# ═════════════════════════════════════════════════════════════════════════════
with tab_perf:
    import matplotlib.pyplot as plt

    st.markdown("""
    <div class="section-title">System Insights</div>
    <div class="section-sub">Operational accuracy indicators and recent validation results</div>
    """, unsafe_allow_html=True)

    # ── Test metrics ──
    metrics = _load_pickle(TEST_METRICS)
    if metrics:
        c1, c2, c3, c4 = st.columns(4)
        defs = [
            (c1, "", "Average Error", "mae",          "Lower is better"),
            (c2, "", "Peak Error",    "rmse",         "Lower is better"),
            (c3, "", "Percent Error", "robust_mape",  "Lower is better"),
            (c4, "", "Fit Score",     "r2",           "Higher is better"),
        ]
        for col, icon, label, key, desc in defs:
            val = metrics.get(key, metrics.get("mape", 0))
            fmt = f"{val:.2f}%" if key == "robust_mape" else f"{val:.4f}"
            col.markdown(f"""
            <div class="metric-card">
                <div class="metric-icon">{icon}</div>
                <div class="metric-label">{label}</div>
                <div class="metric-value">{fmt}</div>
                <div class="metric-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Validation summary is not available yet.")

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # ── Training curves ──
    history = _load_pickle(HISTORY_PATH)
    if history:
        col_a, col_b = st.columns(2)

        def _plot(ax, data, label, color):
            epochs = range(1, len(data) + 1)
            ax.plot(epochs, data, color=color, lw=2, label=label)
            ax.fill_between(epochs, data, alpha=0.07, color=color)
            ax.set_facecolor("#0D1321")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            for sp in ["left", "bottom"]:
                ax.spines[sp].set_color((1.0, 1.0, 1.0, 0.1))
            ax.tick_params(colors="#9AA0A6", labelsize=9)
            ax.yaxis.label.set_color("#9AA0A6")
            ax.xaxis.label.set_color("#9AA0A6")
            ax.grid(axis="y", color=(1.0, 1.0, 1.0, 0.05), lw=0.7)
            ax.set_axisbelow(True)

        with col_a:
            fig, ax = plt.subplots(figsize=(6, 3.5))
            fig.patch.set_facecolor("#0D1321")
            _plot(ax, history["train_loss"], "Training", "#1A73E8")
            if "val_loss" in history:
                _plot(ax, history["val_loss"], "Validation", "#34A853")
            ax.set_title("Error Trend", color="#E8EAED", fontsize=11, fontweight=600)
            ax.set_xlabel("Run")
            ax.set_ylabel("Error")
            ax.legend(fontsize=9, facecolor="#0D1321", labelcolor="#BDC1C6", edgecolor="#333")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with col_b:
            if "val_mae" in history:
                fig, ax = plt.subplots(figsize=(6, 3.5))
                fig.patch.set_facecolor("#0D1321")
                _plot(ax, history["val_mae"],  "Average Error",  "#34A853")
                if "val_rmse" in history:
                    _plot(ax, history["val_rmse"], "Peak Error", "#FBBC04")
                ax.set_title("Validation Trend", color="#E8EAED", fontsize=11, fontweight=600)
                ax.set_xlabel("Run")
                ax.legend(fontsize=9, facecolor="#0D1321", labelcolor="#BDC1C6", edgecolor="#333")
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
    else:
        st.info("Trend history is not available yet.")

    # ── Ablation table ──
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("**Configuration Comparison**")

    import json
    abl = None
    if ABLATION_PATH.exists():
        with open(ABLATION_PATH) as f:
            abl = json.load(f)

    if abl:
        rows = []
        for name, m in abl.items():
            rows.append({
                "Configuration": name,
                "Average Error":  f"{m.get('mae',0):.4f}",
                "Peak Error": f"{m.get('rmse',0):.4f}",
                "Percent Error": f"{m.get('robust_mape',0):.2f}%",
                "Fit Score":   f"{m.get('r2',0):.4f}",
            })
        import pandas as pd
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("Configuration comparison is not available yet.")

    # ── Forecast comparison image ──
    if FORECAST_IMG.exists():
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        st.markdown("**Expected vs Actual Traffic**")
        st.image(str(FORECAST_IMG), use_container_width=True,
                 caption="Expected traffic compared with observed traffic")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4: TRAFFIC TRENDS
# ═════════════════════════════════════════════════════════════════════════════
with tab_forecast:
    import matplotlib.pyplot as plt

    st.markdown("""
    <div class="section-title">Traffic Trends</div>
    <div class="section-sub">Review recent and expected movement at selected intersections</div>
    """, unsafe_allow_html=True)

    traffic_data = _load_npy(TRAFFIC_DATA)

    if traffic_data is not None:
        T, N = traffic_data.shape

        named = {
            0: "Intersection #0 (Nal Stop area)",
            1: "Intersection #1 (Karve Rd junction)",
            5: "Intersection #5 (Paud Rd start)",
            10: "Intersection #10 (MIT College gate)",
            20: "Intersection #20 (Dahanukar Colony)",
            50: "Intersection #50 (Kothrud Depot)",
            100: "Intersection #100 (Vanaz Corner)",
            200: "Intersection #200 (Warje bridge)",
        }
        available = {k: v for k, v in named.items() if k < N}

        sel_col, info_col = st.columns([2, 1])
        with sel_col:
            mode = st.radio("Select intersection by:", ["Named locations", "Node index"], horizontal=True)
        node_idx = 0
        if mode == "Named locations" and available:
            node_idx = st.selectbox("Choose intersection:", list(available.keys()),
                                    format_func=lambda x: available[x])
        else:
            node_idx = st.slider("Node index:", 0, N - 1, 0)

        with info_col:
            st.markdown(f"""
            <div class="glass-card" style="margin-top:8px; padding:14px 16px;">
                <div style="font-size:0.72rem; font-weight:600; color:#9AA0A6; text-transform:uppercase; letter-spacing:0.8px; margin-bottom:8px;">Selected Node</div>
                <div class="info-row"><span class="info-key">Index</span><span class="info-val">#{node_idx}</span></div>
                <div class="info-row"><span class="info-key">Timesteps</span><span class="info-val">{T:,}</span></div>
                <div class="info-row"><span class="info-key">Total Nodes</span><span class="info-val">{N:,}</span></div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)

        series    = traffic_data[:, node_idx]
        disp_len  = min(288, len(series))
        disp_data = series[-disp_len:]
        time_lbls = [
            (now - timedelta(minutes=5 * (disp_len - i))).strftime("%H:%M")
            for i in range(disp_len)
        ]

        fig, ax = plt.subplots(figsize=(12, 4))
        fig.patch.set_facecolor("#0D1321")
        ax.set_facecolor("#0A0F1C")

        hist = disp_data[:-3] if len(disp_data) > 3 else disp_data
        ax.plot(range(len(hist)), hist, color="#1A73E8", lw=2, label="Historical", zorder=3)
        ax.fill_between(range(len(hist)), hist, alpha=0.08, color="#1A73E8")

        if len(disp_data) > 3:
            p_start = len(hist) - 1
            ax.plot(range(p_start, len(disp_data)), disp_data[p_start:],
                    color="#EA4335", lw=2.5, ls="--", label="Forecast (15 min)", zorder=4)
            ax.axvspan(p_start, len(disp_data) - 1, alpha=0.05, color="#EA4335", zorder=1)
            ax.axvline(p_start, color=(1.0, 1.0, 1.0, 0.15), lw=1, ls=":", zorder=2)

        ax.set_xlabel("Time", color="#9AA0A6", fontsize=10)
        ax.set_ylabel("Traffic (normalised)", color="#9AA0A6", fontsize=10)
        ax.set_title(f"Traffic Pattern — Node #{node_idx}", color="#E8EAED",
                     fontsize=12, fontweight=600, pad=12)
        ax.legend(fontsize=9, facecolor="#0D1321", labelcolor="#BDC1C6", edgecolor="#333")

        step = max(1, disp_len // 12)
        ax.set_xticks(range(0, disp_len, step))
        ax.set_xticklabels([time_lbls[i] for i in range(0, disp_len, step)],
                            rotation=45, ha="right", fontsize=8, color="#9AA0A6")
        ax.tick_params(axis="y", colors="#9AA0A6")
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        for sp in ["left", "bottom"]:
            ax.spines[sp].set_color((1.0, 1.0, 1.0, 0.08))
        ax.grid(axis="y", color=(1.0, 1.0, 1.0, 0.04), lw=0.7)

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # Quick stats
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        st.markdown("**Node Statistics (last 24 hours)**")

        s1, s2, s3, s4 = st.columns(4)
        stats = [
            (s1, "Mean", f"{disp_data.mean():.3f}", "#1A73E8"),
            (s2, "Peak",  f"{disp_data.max():.3f}",  "#EA4335"),
            (s3, "Min",   f"{disp_data.min():.3f}",  "#34A853"),
            (s4, "Std Dev", f"{disp_data.std():.3f}", "#F9AB00"),
        ]
        for col, lbl, val, color in stats:
            col.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{lbl}</div>
                <div class="metric-value" style="font-size:1.6rem; color:{color};">{val}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="api-warning">
            Traffic data not found. Run:<br>
            <code>venv\\Scripts\\python.exe data/generate_data.py</code>
        </div>
        """, unsafe_allow_html=True)


# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<hr class='divider'>
<div style="text-align:center; font-size:0.76rem; color:#3C4043; padding:6px 0 12px;">
    Pune Traffic Operations
</div>
""", unsafe_allow_html=True)
