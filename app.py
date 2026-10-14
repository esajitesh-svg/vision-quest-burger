"""Voice-upgraded cloud-hosted Smriti AI / VisionQuest demo.

This edition uses browser camera capture (HTTPS) or image upload. The vision model
runs on the hosting server via Transformers; it does not require the user's Ollama.
The app is password-gated and uses a shared SQLite demo vault. Do not store real
medical or highly sensitive personal data in this prototype.
"""
from __future__ import annotations

import csv
import hashlib
from html import escape
import hmac
import io
import os
from datetime import datetime, timezone
from pathlib import Path
from threading import Thread

import av
import cv2
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError
from streamlit_webrtc import VideoHTMLAttributes, WebRtcMode, webrtc_streamer

from core.cloud_db import init_db, list_people, list_memories, save_person, save_memory, search_memories, export_memories_csv
from core.cloud_vision import MODE_LABELS, analyze_image, model_status, prepare_image
from core.realtime_detector import REALTIME_DETECTOR_LABEL, RealtimeYOLOXDetector
from core.live_report import build_live_report
from core.cloud_cv import process_image_opencv
from core.cloud_live import LiveMonitorState, process_live_frame, draw_detection_overlay
from core.voice import SUPPORTED_LANGUAGES, build_speech_html

ROOT = Path(__file__).resolve().parent


def render_cv_view(
    frame_bgr,
    mode: str = "COLOR + HUD",
    canny_low: int = 55,
    canny_high: int = 140,
    frame_count: int = 0,
):
    """Render a selectable OpenCV diagnostic canvas without changing AI input."""
    import numpy as np

    if not isinstance(frame_bgr, np.ndarray) or frame_bgr.size == 0:
        raise ValueError("Cannot render an empty OpenCV frame.")
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError("OpenCV display input must have three color channels.")

    source = frame_bgr.copy()
    gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    low = max(0, min(254, int(canny_low)))
    high = max(low + 1, min(255, int(canny_high)))

    if mode == "GRAYSCALE":
        display = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    elif mode == "CANNY EDGE MAP":
        display = cv2.cvtColor(cv2.Canny(gray, low, high), cv2.COLOR_GRAY2BGR)
    elif mode == "CONTOUR MAP":
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        display = source.copy()
        visible = [contour for contour in contours if cv2.contourArea(contour) >= 120]
        cv2.drawContours(display, visible, -1, (70, 255, 115), 2)
        for contour in visible[:30]:
            x, y, width, height = cv2.boundingRect(contour)
            if width > 24 and height > 20:
                cv2.rectangle(display, (x, y), (x + width, y + height), (255, 210, 60), 1)
    elif mode == "ADAPTIVE THRESHOLD":
        mask = cv2.adaptiveThreshold(
            cv2.GaussianBlur(gray, (5, 5), 0),
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            21,
            5,
        )
        display = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    elif mode == "HSV SATURATION":
        saturation = cv2.cvtColor(source, cv2.COLOR_BGR2HSV)[:, :, 1]
        display = cv2.applyColorMap(saturation, cv2.COLORMAP_TURBO)
    else:
        display = source.copy()
        height, width = display.shape[:2]
        cx, cy = width // 2, height // 2
        hud_color = (255, 235, 70)
        cv2.line(display, (cx - 22, cy), (cx - 7, cy), hud_color, 1, cv2.LINE_AA)
        cv2.line(display, (cx + 7, cy), (cx + 22, cy), hud_color, 1, cv2.LINE_AA)
        cv2.line(display, (cx, cy - 22), (cx, cy - 7), hud_color, 1, cv2.LINE_AA)
        cv2.line(display, (cx, cy + 7), (cx, cy + 22), hud_color, 1, cv2.LINE_AA)
        cv2.circle(display, (cx, cy), 4, hud_color, 1, cv2.LINE_AA)
        arm, margin = max(12, min(30, width // 18, height // 12)), 10
        for x, y, sx, sy in [
            (margin, margin, 1, 1),
            (width - margin, margin, -1, 1),
            (margin, height - margin, 1, -1),
            (width - margin, height - margin, -1, -1),
        ]:
            cv2.line(display, (x, y), (x + sx * arm, y), hud_color, 2, cv2.LINE_AA)
            cv2.line(display, (x, y), (x, y + sy * arm), hud_color, 2, cv2.LINE_AA)

    height, width = display.shape[:2]
    cv2.rectangle(display, (0, 0), (width, min(32, height)), (4, 14, 20), thickness=-1)
    cv2.rectangle(display, (0, max(0, height - 25)), (width, height), (4, 14, 20), thickness=-1)
    cv2.putText(
        display,
        f"VISIONQUEST // {mode}"[:64],
        (10, min(21, max(15, height - 2))),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.47,
        (90, 240, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        display,
        f"FRAME {max(0, int(frame_count)):06d}  |  {width}x{height}  |  OPENCV PIPELINE",
        (10, max(15, height - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.39,
        (165, 222, 230),
        1,
        cv2.LINE_AA,
    )
    return display


def run_local_detector_worker(live_state, detector, frame_bgr, confidence_threshold: float) -> None:
    """Run local detection in the background so inference never blocks video delivery."""
    try:
        result = detector.detect(
            frame_bgr,
            confidence_threshold=confidence_threshold,
            max_detections=30,
        )
        live_state.finish_detection(
            detections=result.detections,
            model=REALTIME_DETECTOR_LABEL,
            elapsed_seconds=result.elapsed_seconds,
        )
    except Exception as exc:
        live_state.finish_detection(error=f"{type(exc).__name__}: {exc}")


def run_live_vision_worker(live_state, sample_bytes: bytes, mode: str, question: str, max_side: int) -> None:
    """Run remote scene understanding off the Streamlit/UI thread."""
    try:
        result = analyze_image(sample_bytes, mode=mode, focus=question, max_side=max_side)
        observation = {
            "text": result.text,
            "mode": mode,
            "model": result.model,
            "elapsed": result.elapsed_seconds,
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "sequence": int(datetime.now(timezone.utc).timestamp() * 1000),
            "source_hash": hashlib.sha256(sample_bytes).hexdigest(),
        }
        live_state.finish_analysis(observation=observation)
    except Exception as exc:
        live_state.finish_analysis(error=f"{type(exc).__name__}: {exc}")


st.set_page_config(
    page_title="Smriti AI · VisionQuest Cloud",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_resource(show_spinner=False)
def get_realtime_detector() -> RealtimeYOLOXDetector:
    """Load the local OpenCV DNN model once per app process."""
    return RealtimeYOLOXDetector()


CSS = r"""
<style>
:root { color-scheme: dark; }
html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] { background:#05090e !important; color:#c5d9e3 !important; }
[data-testid="stHeader"] { background:rgba(5,9,14,.94) !important; }
[data-testid="stMainBlockContainer"] { max-width:none; width:100%; padding:.3rem .55rem .65rem; }
[data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li, [data-testid="stCaptionContainer"], [data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] *, label, .stRadio label, .stCheckbox label { color:#b9ccd7 !important; }
h1,h2,h3,h4 { color:#e3f7ff !important; letter-spacing:-.02em; }
a { color:#46eaff !important; }
.hero { padding:1.8rem 2rem; border-radius:18px; margin:.15rem 0 1.2rem; color:#e6fbff !important; background:radial-gradient(circle at 90% 0%,rgba(0,220,255,.13),transparent 30%),linear-gradient(115deg,#071017 0%,#0b1b25 55%,#102a37 100%); border:1px solid rgba(64,225,255,.32); box-shadow:0 0 28px rgba(0,208,240,.08), inset 0 0 28px rgba(0,190,225,.035); }
.hero * { color:#e6fbff !important; }
.hero .kicker { text-transform:uppercase; letter-spacing:.16em; font-size:.72rem; font-weight:800; color:#52edff !important; }
.hero h1 { font-size:clamp(2rem,3vw,2.7rem); margin:.55rem 0 .35rem; line-height:1.1; text-transform:uppercase; }
.hero p { color:#a8c5d3 !important; max-width:900px; line-height:1.65; margin:0; }
.pill { display:inline-block; padding:.28rem .6rem; margin:.75rem .3rem 0 0; border:1px solid rgba(73,228,255,.28); background:rgba(0,207,238,.06); border-radius:3px; font-size:.76rem; letter-spacing:.04em; }
[data-testid="stMetric"] { background:linear-gradient(135deg,#09151d,#0b1b24) !important; border:1px solid #21404d !important; border-radius:10px; padding:.8rem !important; }
[data-testid="stMetric"] * { color:#d5f6ff !important; }
[data-testid="stMetricLabel"] { color:#69dbe9 !important; }
[data-baseweb="input"] input, [data-baseweb="textarea"] textarea, [data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea { background:#09151c !important; color:#d9f4fa !important; border-color:#254653 !important; }
[data-baseweb="select"] > div { background:#09151c !important; border-color:#254653 !important; }
[data-baseweb="select"] * { color:#d9f4fa !important; }
[data-testid="stFileUploader"] { background:#08141b !important; border:1px dashed #2e6170 !important; border-radius:8px; padding:.3rem; }
[data-testid="stFileUploader"] * { color:#c5d9e3 !important; }
.stButton button, [data-testid="stDownloadButton"] button { border-radius:5px !important; font-weight:750 !important; min-height:2.6rem; letter-spacing:.035em; transition:all .16s ease; }
.stButton button[kind="primary"], [data-testid="stDownloadButton"] button[kind="primary"] { background:#00cfe8 !important; color:#031015 !important; border:1px solid #31efff !important; box-shadow:0 0 15px rgba(0,221,247,.16); }
.stButton button[kind="primary"] *, [data-testid="stDownloadButton"] button[kind="primary"] * { color:#031015 !important; }
.stButton button:hover { border-color:#48efff !important; box-shadow:0 0 14px rgba(0,221,247,.14); }
[data-testid="stVerticalBlockBorderWrapper"] { background:#08141b !important; border-color:#1e3d49 !important; border-radius:10px !important; }
[data-testid="stAlert"] { border-radius:7px; }
[data-testid="stTabs"] [data-baseweb="tab-list"] { gap:.25rem; border-bottom:1px solid #1d414e; }
[data-testid="stTabs"] [data-baseweb="tab"] { background:transparent; padding:.72rem .8rem; }
[data-testid="stTabs"] [aria-selected="true"] { color:#5eecff !important; border-bottom-color:#00d8f2 !important; }
.result { padding:1rem 1.15rem; border:1px solid #214d5a; border-left:3px solid #00d7ef; background:#07151c; border-radius:6px; color:#d1eef5; line-height:1.65; white-space:pre-wrap; box-shadow:inset 0 0 20px rgba(0,190,225,.035); }
.small-note { font-size:.86rem; color:#86a7b6 !important; }
.workspace-masthead { display:flex; flex-wrap:wrap; align-items:center; justify-content:space-between; gap:1rem; padding:.55rem 1rem; margin:.08rem 0 .3rem; background:linear-gradient(110deg,#03080c,#071820 65%,#071019); border:1px solid #1c5665; border-left:3px solid #32eaff; box-shadow:inset 0 0 24px rgba(0,216,239,.035); }
.workspace-brand { display:flex; gap:.9rem; align-items:center; min-width:250px; }
.workspace-glyph { width:48px; height:48px; display:flex; align-items:center; justify-content:center; border:1px solid #2c93a5; color:#5ef3ff; font-size:1.8rem; background:radial-gradient(circle,#12313b,#061116); box-shadow:0 0 20px rgba(0,223,247,.1); }
.workspace-kicker { color:#50e7f8 !important; font:700 .66rem/1.4 monospace; letter-spacing:.2em; }
.workspace-title { font:800 clamp(1.15rem,2.3vw,1.9rem)/1.15 monospace; color:#e1fbff !important; letter-spacing:.05em; margin:.15rem 0 .3rem; }
.workspace-subtitle { color:#7799a8 !important; font:.75rem/1.5 monospace; letter-spacing:.06em; margin:0; }
.workspace-node { display:flex; flex-wrap:wrap; gap:.45rem; }
.workspace-node span { border:1px solid #204450; background:#07131a; padding:.55rem .7rem; font:700 .65rem monospace; letter-spacing:.06em; color:#79bdc9; }
.workspace-node b { color:#53f2bd; font-weight:800; }
.workspace-readouts { display:flex; flex-wrap:wrap; gap:1.2rem; padding:.25rem .75rem; margin-bottom:.35rem; border-bottom:1px solid #153a45; background:rgba(6,19,26,.7); font: .7rem monospace; letter-spacing:.08em; color:#5da2b1; }
.workspace-readouts b { color:#d5faff; margin-left:.35rem; }
.cv-workbench-label { padding:.45rem .6rem; margin:.3rem 0 .65rem; border-left:2px solid #39e6f8; background:linear-gradient(90deg,rgba(0,196,220,.09),transparent); color:#69e9f8 !important; font:700 .7rem monospace; letter-spacing:.15em; }
.cv-feed-heading { color:#67edf8 !important; font:700 .76rem monospace; letter-spacing:.16em; border-bottom:1px solid #174552; padding-bottom:.55rem; margin:.8rem 0 .45rem; }
div[data-testid="stMetric"] { box-shadow:inset 0 0 18px rgba(0,220,240,.025); }
[data-testid="stTabs"] [data-baseweb="tab"] { font-family:monospace; font-size:.72rem; letter-spacing:.04em; }
[data-testid="stTabs"] [aria-selected="true"] { background:linear-gradient(180deg,rgba(0,216,239,.1),transparent); }
[data-testid="stVerticalBlockBorderWrapper"] { border-radius:3px !important; }
.hud-label { color:#59eaff !important; font-size:.74rem; font-weight:800; letter-spacing:.15em; text-transform:uppercase; }
.hud-rule { height:1px; width:100%; background:linear-gradient(90deg,#00d7ef,rgba(0,215,239,.1),transparent); margin:.55rem 0 1rem; }
@keyframes vqSpin { to { transform:rotate(360deg); } }
@keyframes vqPulse { 0%,100% { box-shadow:0 0 15px rgba(0,225,255,.22), inset 0 0 18px rgba(0,225,255,.08); } 50% { box-shadow:0 0 36px rgba(0,225,255,.48), inset 0 0 25px rgba(0,225,255,.17); } }
@keyframes vqSweep { 0% { transform:rotate(0deg); } 100% { transform:rotate(360deg); } }
.vq-splash { position:fixed; inset:0; width:100vw; height:100vh; z-index:999999; display:flex; flex-direction:column; align-items:center; justify-content:center; gap:1.2rem; background:radial-gradient(circle at 50% 45%,#102a36 0%,#07121a 32%,#020507 74%); color:#bdf8ff; overflow:hidden; text-align:center; }
.vq-splash:before, .vq-splash:after { content:""; position:absolute; width:82vw; height:82vw; max-width:900px; max-height:900px; border:1px solid rgba(48,216,244,.08); border-radius:50%; }
.vq-splash:after { width:65vw; height:65vw; border-color:rgba(48,216,244,.10); }
.vq-splash-core { position:relative; width:210px; height:210px; border-radius:50%; display:flex; align-items:center; justify-content:center; flex-direction:column; border:1px solid rgba(86,238,255,.8); background:radial-gradient(circle,#13333e 0%,#06151d 63%,#02070a 100%); animation:vqPulse 2.1s ease-in-out infinite; z-index:2; }
.vq-ring { position:absolute; inset:-22px; border:2px solid transparent; border-top-color:#51efff; border-right-color:rgba(61,225,255,.26); border-radius:50%; animation:vqSpin 2.2s linear infinite; }
.vq-ring.inner { inset:-9px; border-width:1px; border-top-color:#d2fcff; border-left-color:rgba(50,217,242,.38); animation-duration:3.4s; animation-direction:reverse; }
.vq-orbit { position:absolute; width:245px; height:245px; border:1px dashed rgba(72,218,246,.33); border-radius:50%; animation:vqSweep 11s linear infinite; }
.vq-orbit:before { content:""; position:absolute; top:27px; left:27px; width:7px; height:7px; border-radius:50%; background:#a6fbff; box-shadow:0 0 12px #42eaff; }
.vq-splash-title { font-size:1.15rem; letter-spacing:.13em; font-weight:900; color:#e1fcff; text-shadow:0 0 18px rgba(0,221,255,.6); }
.vq-splash-sub { position:relative; z-index:3; color:#68d9e9; font-size:.7rem; letter-spacing:.2em; text-transform:uppercase; }
.vq-splash-line { width:210px; height:2px; background:#14333d; overflow:hidden; position:relative; z-index:3; }
.vq-splash-line:after { content:""; display:block; height:100%; width:35%; background:#6af1ff; box-shadow:0 0 9px #45ecff; animation:vqSweep 1.1s linear infinite; }
@media(max-width:720px) { .hero { padding:1.25rem; border-radius:10px; } .hero h1 { font-size:1.9rem; } .vq-splash-core { width:175px; height:175px; } .vq-orbit { width:208px; height:208px; } }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# One-time cinematic boot splash per browser session.
if not st.session_state.get("vq_intro_shown", False):
    import time as _intro_time
    splash = st.empty()
    splash.markdown(
        """
        <div class="vq-splash">
          <div class="vq-orbit"></div>
          <div class="vq-splash-core">
            <div class="vq-ring"></div><div class="vq-ring inner"></div>
            <div style="font-size:2rem;line-height:1;margin-bottom:.4rem;">◉</div>
            <div class="vq-splash-title">VISIONQUEST</div>
            <div class="vq-splash-sub" style="margin-top:.45rem;">Visual memory system</div>
          </div>
          <div class="vq-splash-sub">Initializing perception interface</div>
          <div class="vq-splash-line"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    _intro_time.sleep(2.4)
    splash.empty()
    st.session_state["vq_intro_shown"] = True


def get_secret(name: str) -> str:
    value = os.getenv(name, "").strip()
    if value:
        return value
    try:
        return str(st.secrets.get(name, "")).strip()
    except Exception:
        return ""


def password_gate() -> bool:
    configured = get_secret("APP_PASSWORD")
    if not configured:
        st.markdown("<div class='hero'><div class='kicker'>Smriti AI · Cloud setup</div><h1>Finish secure setup</h1><p>The voice-upgraded VisionQuest build is deployed, but setup is incomplete. Add an app password in your hosting platform's Secrets settings to unlock the Vision, Memories, and Voice tabs.</p><span class='pill'>🔊 Voice companion included</span></div>", unsafe_allow_html=True)
        st.code('APP_PASSWORD = "replace-with-a-long-unique-password"', language="toml")
        st.warning("Do not put the password in your source code or Git repository. Configure it using the hosting platform's Secrets feature, then reboot the app.")
        return False
    if st.session_state.get("authenticated"):
        return True
    st.markdown("<div class='hero'><div class='kicker'>Smriti AI · VisionQuest</div><h1>Your visual memory companion</h1><p>Cloud demo · browser camera · image understanding · curated memory notes</p><span class='pill'>🔐 Password-protected demo</span></div>", unsafe_allow_html=True)
    with st.form("login_form"):
        secret_input = st.text_input("App access password", type="password", autocomplete="current-password")
        submitted = st.form_submit_button("Unlock VisionQuest", type="primary", use_container_width=True)
    if submitted:
        if hmac.compare_digest(secret_input, configured):
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("That password doesn't match. Try again.")
    st.caption("Use the app password configured by the person who deployed this demo.")
    return False


if not password_gate():
    st.stop()

init_db()
if "live_monitor_state" not in st.session_state:
    st.session_state["live_monitor_state"] = LiveMonitorState()
live_state: LiveMonitorState = st.session_state["live_monitor_state"]

status = model_status()
people = list_people()
memories = list_memories(limit=500)

st.markdown(
    "<div class='workspace-masthead'>"
    "<div class='workspace-brand'><div class='workspace-glyph'>⌗</div>"
    "<div><div class='workspace-kicker'>SMRITI AI // VISION SYSTEMS</div>"
    "<div class='workspace-title'>VISIONQUEST · CV WORKSPACE</div>"
    "<p class='workspace-subtitle'>LIVE FRAME PIPELINE / PIXEL ANALYSIS / VISUAL MEMORY INTERFACE</p></div></div>"
    "<div class='workspace-node'><span>CV ENGINE <b>OPENCV DNN</b></span><span>INPUT <b>DEVICE CAMERA</b></span>"
    "<span>DETECTOR <b>LOCAL YOLOX</b></span></div></div>",
    unsafe_allow_html=True,
)
st.markdown(
    f"<div class='workspace-readouts'><span>PERSON NODES <b>{len(people):03d}</b></span>"
    f"<span>MEMORY FRAGMENTS <b>{len(memories):03d}</b></span>"
    f"<span>WORKSPACE <b>LIVE / INTERACTIVE</b></span></div>",
    unsafe_allow_html=True,
)
if status[0]:
    st.success(status[1])
else:
    st.warning(status[1])
st.caption(
    "Privacy notice: the camera stream is processed on the app host. Local object detection runs there using OpenCV DNN; sampled scene-explanation frames and uploaded images are sent to the configured external vision-language provider. "
    "The app does not intentionally save source photos, but the provider processes the image request. Use fictional test scenes and obtain consent; do not upload private medical or sensitive imagery."
)

vision_tab, people_tab, memory_tab, voice_tab, search_tab, settings_tab = st.tabs(["◉ CV WORKSPACE", "PERSON NODES", "MEMORY INDEX", "VOICE I/O", "QUERY", "RUNTIME"])

with vision_tab:
    live_tab, image_lab_tab = st.tabs(["🎥 LIVE SENSING // WEBCAM", "📸 IMAGE LAB"])

    with live_tab:
        st.markdown("<div class='cv-workbench-label'>01 / LIVE SENSOR INPUT · WEBCAM</div><div class='cv-feed-heading'>LIVE OPTICAL FEED // REAL-TIME CV PIPELINE</div>", unsafe_allow_html=True)
        st.write("VisionQuest prioritizes a responsive live video feed. Local OpenCV DNN object detection runs asynchronously on a recent frame instead of blocking every video frame; boxes update periodically and disappear when too old. A separate hosted vision-language model explains sampled scenes.")
        st.caption("Camera video is sent to the hosted app for processing. The default HD 1280×720 capture balances detail and latency; Full HD remains selectable. Local object detection runs asynchronously at a limited cadence so it does not hold up the video callback. Only scene-explanation samples use the external vision-language provider. Use consented test scenes.")
        with st.expander("⚙️ PERCEPTION / CAMERA CONTROLS", expanded=False):
            live_auto_analyze = st.checkbox(
                "Enable automatic scene explanation",
                value=bool(status[0]),
                disabled=not status[0],
                key="live_auto_observe_always",
                help="A hosted vision-language model gives the written scene explanation from sampled color frames.",
            )
            live_object_detection = st.checkbox(
                "Enable continuous local object detection + bounding boxes",
                value=True,
                key="live_object_detection",
                help="Local YOLOX inference runs asynchronously on recent frames at a limited cadence so it does not block the video stream.",
            )
            camera_resolution_label = st.selectbox(
                "Camera capture / processing resolution",
                options=[
                    "Full HD · 1920 × 1080",
                    "HD · 1280 × 720",
                    "Standard · 640 × 360",
                ],
                index=1,
                key="live_camera_resolution_latency",
                help="Defaults to HD for a smoother cloud stream. Full HD is available, but uses more upload bandwidth and server CPU; the device/browser may negotiate a lower actual resolution.",
            )
            if camera_resolution_label.startswith("Full HD"):
                live_capture_width, live_capture_height, live_processing_max_side = 1920, 1080, 1920
            elif camera_resolution_label.startswith("HD"):
                live_capture_width, live_capture_height, live_processing_max_side = 1280, 720, 1280
            else:
                live_capture_width, live_capture_height, live_processing_max_side = 640, 360, 640

            detection_threshold = st.slider(
                "Minimum object confidence",
                min_value=15,
                max_value=80,
                value=35,
                step=5,
                key="live_detection_confidence",
                format="%d%%",
                help="Lower values show more detections but can include false positives; higher values hide weaker predictions.",
            ) / 100.0
            live_opencv_always = True
            live_interval = st.select_slider(
                "Scene explanation target interval",
                options=[1, 2, 5, 10],
                value=5,
                key="live_interval_seconds_latency",
                format_func=lambda x: "Every 1 second" if x == 1 else f"Every {x} seconds",
                help="Defaults to every 5 seconds to reduce remote inference load. Its actual cadence depends on provider latency.",
            )
            live_mode_label = st.selectbox(
                "Scene analysis task",
                options=list(MODE_LABELS.values()),
                index=0,
                key="live_auto_mode_label",
            )
            live_mode = next(k for k, v in MODE_LABELS.items() if v == live_mode_label)
            live_question = ""
            if live_mode == "visual_question":
                live_question = st.text_input(
                    "Question about each sampled frame",
                    max_chars=300,
                    key="live_visual_question",
                    placeholder="What objects and actions are visible?",
                )
            live_detail = st.selectbox(
                "AI observation image detail",
                options=[
                    "Fast · 512 px",
                    "Balanced · 768 px",
                    "Maximum detail · 1280 px",
                ],
                index=1,
                key="live_detail_latency",
                help="Balanced is the recommended speed/detail trade-off. Maximum detail is slower and best reserved for still frames or fine text.",
            )
            if live_detail.startswith("Fast"):
                live_max_side = 512
            elif live_detail.startswith("Balanced"):
                live_max_side = 768
            else:
                live_max_side = 1280
            st.markdown("<div class='cv-workbench-label'>DISPLAY PIPELINE · OPENCV OPERATORS</div>", unsafe_allow_html=True)
            pipe_mode_col, pipe_low_col, pipe_high_col = st.columns([1.5, 1, 1], gap="small")
            with pipe_mode_col:
                live_view_mode = st.selectbox(
                    "CV display operator",
                    options=["COLOR + HUD", "GRAYSCALE", "CANNY EDGE MAP", "CONTOUR MAP", "ADAPTIVE THRESHOLD", "HSV SATURATION"],
                    index=0,
                    key="live_cv_display_mode",
                )
            with pipe_low_col:
                canny_low = st.slider("Canny low", min_value=0, max_value=200, value=55, step=5, key="live_canny_low")
            with pipe_high_col:
                canny_high = st.slider("Canny high", min_value=20, max_value=255, value=140, step=5, key="live_canny_high")
            live_enhance = st.checkbox(
                "Apply OpenCV contrast enhancement to AI input",
                value=False,
                key="live_opencv_enhance",
            )
            recording_seconds = st.select_slider(
                "Maximum silent recording length",
                options=[15, 30, 60],
                value=30,
                key="live_record_seconds",
                format_func=lambda x: f"{x} seconds",
            )
            st.caption(
                "Local YOLOX detects object categories on each frame delivered to the server. "
                "The written scene explanation still uses sampled frames through the configured external vision provider. "
                "Camera access needs your permission and a click on START; once started, the feed continues until stopped."
            )

        realtime_detector = None
        detector_init_error = ""
        if live_object_detection:
            try:
                with st.spinner("Loading local YOLOX object detector… first load downloads the compact model"):
                    realtime_detector = get_realtime_detector()
            except Exception as exc:
                detector_init_error = f"{type(exc).__name__}: {exc}"
                live_state.finish_detection(error=detector_init_error)
                st.error(f"LOCAL DETECTOR UNAVAILABLE · {detector_init_error}")

        rtc_configuration = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        turn_url = get_secret("TURN_SERVER_URL")
        turn_username = get_secret("TURN_USERNAME")
        turn_credential = get_secret("TURN_CREDENTIAL")
        if turn_url and turn_username and turn_credential:
            rtc_configuration["iceServers"].append({
                "urls": [part.strip() for part in turn_url.split(",") if part.strip()],
                "username": turn_username,
                "credential": turn_credential,
            })

        def on_live_video_frame(frame):
            try:
                frame_bgr = frame.to_ndarray(format="bgr24")
                # The detector receives the same current frame shown to the user.
                # It runs locally for each frame delivered by WebRTC; no old API box
                # result is copied onto a newer frame.
                processed_bgr, analysis_jpeg, metrics = process_live_frame(
                    frame_bgr, max_side=live_processing_max_side, enhance=live_enhance
                )
                if live_object_detection and realtime_detector is not None:
                    # Reserve at most one inference task and skip new frames while it is
                    # busy. The expensive neural-network call runs on a daemon worker,
                    # never inside WebRTC's frame callback.
                    if live_state.reserve_live_detection(interval_seconds=0.6):
                        try:
                            Thread(
                                target=run_local_detector_worker,
                                args=(
                                    live_state,
                                    realtime_detector,
                                    processed_bgr.copy(),
                                    detection_threshold,
                                ),
                                daemon=True,
                            ).start()
                        except Exception as exc:
                            live_state.finish_detection(
                                error=f"Could not start the detector worker: {type(exc).__name__}: {exc}"
                            )

                detection_status = live_state.detection_snapshot()
                detection_age = detection_status.get("age_seconds")
                detection_fresh = (
                    live_object_detection
                    and realtime_detector is not None
                    and detection_age is not None
                    and detection_age <= 1.5
                )
                detections = detection_status["detections"] if detection_fresh else []
                elapsed_detection = detection_status.get("elapsed_seconds")
                metrics["detector_ms"] = (
                    round(float(elapsed_detection) * 1000.0, 1)
                    if isinstance(elapsed_detection, (int, float)) and elapsed_detection > 0
                    else None
                )
                metrics["detector_fps"] = (
                    round(1.0 / max(float(elapsed_detection), 0.001), 2)
                    if isinstance(elapsed_detection, (int, float)) and elapsed_detection > 0
                    else 0.0
                )
                metrics["detector_age_ms"] = (
                    round(float(detection_age) * 1000.0, 1)
                    if detection_age is not None else None
                )
                metrics["detector_busy"] = bool(detection_status.get("busy"))

                display_bgr = render_cv_view(
                    processed_bgr,
                    mode=live_view_mode,
                    canny_low=canny_low,
                    canny_high=canny_high,
                    frame_count=live_state.frame_count + 1,
                )
                if detections:
                    # Detection is asynchronous; overlay only results that are recent.
                    display_bgr = draw_detection_overlay(
                        display_bgr,
                        detections,
                        age_seconds=float(detection_age),
                        max_age_seconds=1.5,
                    )

                if live_object_detection and realtime_detector is not None:
                    if isinstance(detection_age, (int, float)) and detection_age <= 1.5 and metrics["detector_ms"] is not None:
                        telemetry = f"YOLOX ASYNC  {metrics['detector_ms']:.0f} ms  |  AGE {detection_age:.1f}s"
                    elif detection_status.get("busy"):
                        telemetry = "YOLOX ASYNC  SCANNING"
                    else:
                        telemetry = "YOLOX ASYNC  WAITING"
                    cv2.putText(
                        display_bgr, telemetry,
                        (max(10, display_bgr.shape[1] - 340), 21),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 255, 180), 1, cv2.LINE_AA,
                    )

                ok, display_encoded = cv2.imencode(
                    ".jpg", display_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90]
                )
                display_jpeg = display_encoded.tobytes() if ok else analysis_jpeg
                # The live view and recording use the annotated frame; scene-analysis
                # samples retain the clean color image without boxes or diagnostic text.
                live_state.update_frame(
                    display_bgr, display_jpeg, metrics, analysis_jpeg=analysis_jpeg
                )
                return av.VideoFrame.from_ndarray(display_bgr, format="bgr24")
            except Exception as exc:
                live_state.finish_detection(error=f"OpenCV live detection error: {type(exc).__name__}: {exc}")
                live_state.finish_analysis(error=f"Live frame processing error: {type(exc).__name__}: {exc}")
                return frame

        try:
            live_ctx = webrtc_streamer(
                key="visionquest-live-camera",
                mode=WebRtcMode.SENDRECV,
                video_frame_callback=on_live_video_frame,
                rtc_configuration=rtc_configuration,
                media_stream_constraints={
                    "video": {
                        "width": {"ideal": live_capture_width},
                        "height": {"ideal": live_capture_height},
                        "frameRate": {"ideal": 12, "max": 15},
                    },
                    "audio": False,
                },
                async_processing=True,
                on_video_ended=live_state.close,
                video_html_attrs=VideoHTMLAttributes(
                    autoPlay=True,
                    controls=True,
                    muted=True,
                    style={
                        "width": "100%",
                        "height": "calc(100vh - 190px)",
                        "minHeight": "480px",
                        "objectFit": "contain",
                        "backgroundColor": "#000000",
                    },
                ),
            )
        except Exception as exc:
            live_ctx = None
            st.error(f"Live camera component could not start: {type(exc).__name__}: {exc}")
            st.info("Try refreshing the page, allowing camera permission, or checking whether your network permits WebRTC.")

        if live_ctx is not None:
            @st.fragment(run_every="1s")
            def live_monitor_panel():
                snapshot_jpeg, frame_metrics, frame_count, last_frame_time = live_state.snapshot()
                is_playing = bool(live_ctx.state.playing)

                manual_analyze = st.button(
                    "⚡ ANALYZE CURRENT FRAME NOW",
                    type="primary",
                    use_container_width=True,
                    disabled=(not is_playing or not snapshot_jpeg or live_state.get_status()["analysis_busy"]),
                    key="manual_live_analyze",
                )

                # The local detector runs inside the video callback on every delivered frame.
                # Only natural-language scene explanation uses an asynchronous remote request.
                if manual_analyze:
                    sample = live_state.take_analysis_sample(live_interval, force=True)
                    if sample is None:
                        st.warning("No fresh frame is ready or the scene analyzer is already busy.")
                    else:
                        Thread(
                            target=run_live_vision_worker,
                            args=(live_state, sample, live_mode, live_question, live_max_side),
                            daemon=True,
                        ).start()
                        st.info("Scene analysis started. The live camera and object detector continue while the result is computed.")
                elif is_playing and live_auto_analyze:
                    sample = live_state.take_analysis_sample(live_interval)
                    if sample is not None:
                        Thread(
                            target=run_live_vision_worker,
                            args=(live_state, sample, live_mode, live_question, live_max_side),
                            daemon=True,
                        ).start()

                current_status = live_state.get_status()
                if not is_playing:
                    st.info("Live monitor is idle. Click START in the camera player above and allow browser camera permission.")
                elif frame_count:
                    st.success(f"Live stream active · {frame_count:,} frames received")
                else:
                    st.info("Waiting for the first camera frame…")

                if snapshot_jpeg:
                    with st.expander("Latest OpenCV-processed frame", expanded=False):
                        preview_image = Image.open(io.BytesIO(snapshot_jpeg))
                        st.image(preview_image, use_container_width=True)

                if frame_metrics:
                    metric_cols = st.columns(3)
                    if live_opencv_always:
                        metric_cols[0].metric("OpenCV brightness", f"{frame_metrics['brightness']}/255", frame_metrics["brightness_label"])
                        metric_cols[1].metric("Focus detail", f"{frame_metrics['sharpness']}", frame_metrics["sharpness_label"])
                    else:
                        metric_cols[0].metric("OpenCV", "OFF")
                        metric_cols[1].metric("Image metrics", "Paused")
                    metric_cols[2].metric("Frame", f"{frame_metrics['width']} × {frame_metrics['height']}")
                    st.caption(f"Latest frame JPEG: {frame_metrics['jpeg_bytes']:,} bytes. Focus is a rough heuristic, not a guarantee of image quality.")

                st.markdown("<div class='cv-workbench-label'>03 / ASYNC OBJECT DETECTION · RECENT RESULTS</div>", unsafe_allow_html=True)
                detection_status = live_state.detection_snapshot()
                current_detections = detection_status["detections"] if detection_status["age_seconds"] is not None and detection_status["age_seconds"] <= 1.5 else []
                inference_ms = frame_metrics.get("detector_ms") if frame_metrics else None
                inference_fps = frame_metrics.get("detector_fps") if frame_metrics else None
                detector_cols = st.columns(3, gap="small")
                detector_cols[0].metric("Objects in current frame", len(current_detections))
                detector_cols[1].metric(
                    "Local detector",
                    "LIVE / ON" if live_object_detection and realtime_detector is not None else ("OFF" if not live_object_detection else "UNAVAILABLE"),
                )
                detector_cols[2].metric(
                    "Inference / frame",
                    f"{inference_ms:.0f} ms" if isinstance(inference_ms, (int, float)) else "—",
                    f"Result age {detection_status['age_seconds']:.1f}s" if isinstance(detection_status.get("age_seconds"), (int, float)) else "Waiting for first result",
                )
                if detection_status["error"]:
                    st.warning(f"Local object detector notice: {detection_status['error']}")
                elif not live_object_detection:
                    st.caption("Continuous object detection is disabled. Open PERCEPTION / CAMERA CONTROLS to re-enable it.")
                elif realtime_detector is None:
                    st.error("The local YOLOX model is unavailable; no simulated boxes are shown. Check the model-download error above.")
                elif detection_status["age_seconds"] is None:
                    st.info("Waiting for a live camera frame. Click START in the camera player and allow camera access.")
                elif detection_status["age_seconds"] is not None and detection_status["age_seconds"] > 1.5:
                    st.info("The previous detection result expired. Waiting for the next background scan; live video continues.")
                elif current_detections:
                    st.dataframe(
                        [
                            {
                                "OBJECT": item["label"],
                                "CONFIDENCE": f"{item['score']:.0%}",
                                "POSITION / BOX": (
                                    f"{item['box']['xmin']}, {item['box']['ymin']} → "
                                    f"{item['box']['xmax']}, {item['box']['ymax']}"
                                ),
                            }
                            for item in current_detections
                        ],
                        use_container_width=True,
                        hide_index=True,
                    )
                    st.caption(
                        f"{detection_status['model']} · results are produced asynchronously from a recent frame; boxes expire after 1.5 seconds. "
                        "Confidence is a model score, not a guarantee of correctness."
                    )
                else:
                    st.caption(
                        "No object category passed the confidence threshold in the latest processed frame. "
                        "This does not prove that the scene contains no objects."
                    )

                if current_status["latest_analysis_error"]:
                    st.error(f"Live processing/analysis notice: {current_status['latest_analysis_error']}")
                elif current_status["analysis_busy"]:
                    st.info("AI analysis is in progress. Additional frames are skipped until it finishes.")

                record_col, stop_col = st.columns(2)
                with record_col:
                    if st.button(
                        "⏺ Start MP4 recording",
                        type="primary",
                        use_container_width=True,
                        disabled=(not is_playing or current_status["recording"]),
                        key="start_live_recording",
                    ):
                        try:
                            live_state.start_recording(recording_seconds)
                            st.rerun(scope="app")
                        except Exception as exc:
                            st.error(f"Could not start recording: {exc}")
                with stop_col:
                    if st.button(
                        "■ Stop & finalize recording",
                        use_container_width=True,
                        disabled=not current_status["recording"],
                        key="stop_live_recording",
                    ):
                        live_state.stop_recording("Recording stopped by user.")
                        st.rerun(scope="app")

                current_status = live_state.get_status()
                if current_status["recording"]:
                    st.warning(
                        f"Recording silent MP4 · {current_status['recording_duration']:.1f}/"
                        f"{current_status['recording_limit']} seconds · "
                        f"{current_status['video_frame_count']} frames"
                    )
                elif current_status["recording_notice"]:
                    st.caption(current_status["recording_notice"])

                clip_bytes, clip_name = live_state.get_completed_video()
                if clip_bytes:
                    st.download_button(
                        "⬇️ Download recorded video (MP4)",
                        data=clip_bytes,
                        file_name=clip_name,
                        mime="video/mp4",
                        use_container_width=True,
                        key="download_live_mp4",
                    )
                    if st.button("Clear temporary recording", key="clear_live_mp4"):
                        live_state.clear_completed_video()
                        st.rerun(scope="app")

                observation = current_status["latest_observation"]
                detection_status = live_state.detection_snapshot()
                if observation and st.session_state.get("live_result_sequence") != observation["sequence"]:
                    st.session_state["latest_result"] = {
                        "text": observation["text"],
                        "mode": f"Live · {live_mode}",
                        "model": observation["model"],
                        "elapsed": observation["elapsed"],
                        "timestamp": observation["timestamp"],
                        "source_hash": observation["source_hash"],
                        "origin": "live",
                    }
                    st.session_state["live_result_sequence"] = observation["sequence"]

                live_report = build_live_report(
                    observation,
                    detection_status,
                    frame_metrics,
                    current_status.get("frame_count", 0),
                )
                report_available = bool(
                    observation
                    or detection_status.get("age_seconds") is not None
                    or current_status.get("frame_count", 0)
                )
                st.markdown(
                    "<div class='cv-workbench-label'>04 / COMPLETE OBSERVATION · ANALYSIS · SPOKEN REPORT</div>",
                    unsafe_allow_html=True,
                )
                if report_available:
                    st.markdown("<div class='cv-feed-heading'>WHAT VISIONQUEST CURRENTLY SEES</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='result'>{escape(live_report)}</div>", unsafe_allow_html=True)

                    # The live report can include changing brightness and frame count every
                    # second. Keep spoken text stable between actual model-result changes,
                    # so the browser speech player is not recreated on every UI refresh.
                    speech_detection_status = dict(detection_status)
                    speech_detection_status["busy"] = False
                    if speech_detection_status.get("age_seconds") is not None:
                        speech_detection_status["age_seconds"] = (
                            10.0 if float(speech_detection_status["age_seconds"]) > 9.0 else 0.0
                        )
                    speech_metrics = {
                        key: frame_metrics[key]
                        for key in ("width", "height")
                        if frame_metrics and key in frame_metrics
                    }
                    speech_report = build_live_report(
                        observation,
                        speech_detection_status,
                        speech_metrics,
                        frame_count=0,
                    )
                    if st.button(
                        "Freeze / refresh report for spoken playback",
                        key="freeze_live_report_for_speech",
                        help="Capture this report into a stable snapshot so the live refresh will not replace the words while the browser reads them.",
                    ):
                        st.session_state["live_report_speech_snapshot"] = speech_report

                    speech_snapshot = st.session_state.get("live_report_speech_snapshot", "")
                    if speech_snapshot:
                        st.caption("Spoken report is frozen at the snapshot time. Press Freeze / refresh to read the latest detections.")
                        st.components.v1.html(
                            build_speech_html(
                                speech_snapshot,
                                language=st.session_state.get("voice_language", "en-IN"),
                                rate=float(st.session_state.get("voice_rate", 0.92)),
                                widget_label="🔊 SPEAK FULL LIVE REPORT",
                            ),
                            height=92,
                            scrolling=False,
                        )
                    else:
                        st.caption("Freeze a report snapshot below, then tap Speak Full Live Report to hear the scene and every detected object.")

                    if st.button("Load full analysis into Voice tab", key="live_report_to_voice"):
                        st.session_state["voice_text"] = live_report
                        st.rerun(scope="app")
                else:
                    st.info("Start the camera stream to see the live scene report, object inventory, confidence scores, and OpenCV frame measurements.")

                if observation:
                    st.caption(
                        f"Latest scene-model result · {observation['model']} · "
                        f"{observation['elapsed']:.1f}s · {observation['timestamp']}"
                    )
                    current_people = list_people()
                    if current_people:
                        names = {p["id"]: f'{p["name"]} — {p["relationship"]}' for p in current_people}
                        with st.form(f"save_live_observation_{observation['sequence']}"):
                            selected_person = st.selectbox(
                                "Manually choose a person for this reviewed note",
                                options=list(names),
                                format_func=lambda p_id: names[p_id],
                                key=f"live_person_{observation['sequence']}",
                            )
                            reviewed_text = st.text_area(
                                "Review or edit before saving",
                                value=observation["text"],
                                max_chars=4000,
                                height=100,
                                key=f"live_note_{observation['sequence']}",
                            )
                            save_live_note = st.form_submit_button("Save reviewed live observation")
                            if save_live_note:
                                try:
                                    save_memory(
                                        selected_person,
                                        reviewed_text,
                                        f"Live OpenCV observation ({observation['mode']}); model: {observation['model']}",
                                    )
                                    st.success("Reviewed note saved to the selected person's demo memory.")
                                    st.rerun(scope="app")
                                except Exception as exc:
                                    st.error(str(exc))
                    else:
                        st.caption("Add a fictional person in the People tab before saving a live observation.")

            live_monitor_panel()
    with image_lab_tab:
        left, right = st.columns([1.05, .95], gap="large")
        with left:
            st.subheader("Give VisionQuest an image")
            source_tab, upload_tab = st.tabs(["📸 Capture photo", "📁 Upload image"])
            with source_tab:
                camera_image = st.camera_input("Capture a photo", key="cloud_camera_input", help="Capture one still image in this device's browser over HTTPS.")
            with upload_tab:
                uploaded_image = st.file_uploader("Choose a JPG, PNG, or WEBP image", type=["jpg", "jpeg", "png", "webp"], key="cloud_upload_input")
            file_obj = camera_image if camera_image is not None else uploaded_image
            image_bytes = file_obj.getvalue() if file_obj is not None else None
            if image_bytes:
                try:
                    preview, dimensions = prepare_image(image_bytes, max_side=512)
                    st.image(preview, caption=f"Image ready · {preview.size[0]} × {preview.size[1]} pixels after resize (original {dimensions[0]} × {dimensions[1]})", use_container_width=True)
                except Exception as exc:
                    st.error(f"The image could not be prepared: {exc}")
                    image_bytes = None
            mode_label = st.selectbox("Vision task", list(MODE_LABELS.values()), index=0)
            mode = next(k for k, v in MODE_LABELS.items() if v == mode_label)
            question = ""
            if mode == "visual_question":
                question = st.text_input("Question about the image", max_chars=300, placeholder="What objects are on the table?")
            detail_level = st.select_slider("Image detail", options=["Fast", "Balanced", "Text detail"], value="Fast", help="Cloud CPU inference can take time. Fast is best for an initial test.")
            max_side = {"Fast": 448, "Balanced": 512, "Text detail": 640}[detail_level]
            use_opencv = st.checkbox(
                "Use OpenCV image processing", value=True,
                help="Runs OpenCV on the captured/uploaded image in the cloud: decoding, resizing, lighting/focus estimates, and optional enhancement."
            )
            enhance_opencv = st.checkbox(
                "Enhance contrast and edge detail with OpenCV", value=False,
                disabled=not use_opencv,
                help="Optional enhancement can help some images, but may alter fine details. Compare results when reading text."
            )
            analyze = st.button("✨ Analyze image", type="primary", use_container_width=True, disabled=not bool(image_bytes) or (mode == "visual_question" and not question.strip()))
        with right:
            st.subheader("Visual observation")
            st.markdown("<div class='small-note'>Task-specific prompts ask the model to stay grounded in the image, mention uncertainty, and avoid guessing identities.</div>", unsafe_allow_html=True)
            if analyze and image_bytes:
                try:
                    with st.spinner("Processing with OpenCV and analyzing the image. The first cloud model load may take longer…"):
                        model_image_bytes = image_bytes
                        cv_metrics = None
                        if use_opencv:
                            model_image_bytes, cv_metrics = process_image_opencv(
                                image_bytes, max_side=max_side, enhance=enhance_opencv
                            )
                        result = analyze_image(model_image_bytes, mode=mode, focus=question, max_side=max_side)
                    st.session_state["latest_result"] = {
                        "text": result.text, "mode": mode, "model": result.model,
                        "elapsed": result.elapsed_seconds,
                        "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
                        "source_hash": hashlib.sha256(image_bytes).hexdigest(),
                        "opencv_enabled": bool(use_opencv), "opencv_metrics": cv_metrics,
                    }
                    st.success(f"Analysis completed in {result.elapsed_seconds:.1f} seconds · {result.model}")
                except Exception as exc:
                    st.error(f"Vision analysis failed: {exc}")
                    st.info("Try Fast detail, a smaller image, or retry after the model has finished loading.")
            latest = st.session_state.get("latest_result")
            current_hash = hashlib.sha256(image_bytes).hexdigest() if image_bytes else None
            latest_matches_current = bool(latest and current_hash and hmac.compare_digest(latest.get("source_hash", ""), current_hash))
            if latest and latest_matches_current:
                st.markdown(f"<div class='result'>{escape(latest['text'])}</div>", unsafe_allow_html=True)
                st.caption(f"{latest['mode']} · {latest['model']} · {latest['timestamp']}")
                if latest.get("opencv_enabled") and latest.get("opencv_metrics"):
                    cvm = latest["opencv_metrics"]
                    st.markdown("**OpenCV preprocessing report**")
                    cv1, cv2c, cv3 = st.columns(3)
                    cv1.metric("Brightness", f"{cvm['brightness']}/255", cvm["brightness_label"])
                    cv2c.metric("Focus detail", f"{cvm['sharpness_score']}", cvm["sharpness_label"])
                    cv3.metric("Processed image", f"{cvm['processed_width']} × {cvm['processed_height']}")
                    st.caption(f"OpenCV resized {cvm['original_width']} × {cvm['original_height']} to {cvm['processed_width']} × {cvm['processed_height']}; output JPEG {cvm['output_bytes']:,} bytes. Focus score is a rough heuristic, not a guarantee of image quality.")
                elif latest.get("opencv_enabled"):
                    st.caption("OpenCV ran, but no preprocessing metrics were returned.")
                st.components.v1.html(
                    build_speech_html(
                        latest["text"],
                        language=st.session_state.get("voice_language", "en-IN"),
                        rate=float(st.session_state.get("voice_rate", 0.92)),
                        widget_label="Read observation aloud",
                    ),
                    height=92,
                    scrolling=False,
                )
                with st.form("save_observation_form"):
                    names = {p["id"]: f'{p["name"]} — {p["relationship"]}' for p in list_people()}
                    if names:
                        chosen_id = st.selectbox("Manually choose a person for this note", list(names), format_func=lambda p_id: names[p_id])
                        corrected = st.text_area("Review or edit the observation before saving", value=latest["text"], max_chars=4000, height=120)
                        save_clicked = st.form_submit_button("Save reviewed note", type="primary")
                        if save_clicked:
                            try:
                                save_memory(chosen_id, corrected, f"Vision observation ({latest['mode']}); model: {latest['model']}")
                                st.success("Saved to the selected person's cloud demo memory vault.")
                            except Exception as exc:
                                st.error(str(exc))
                    else:
                        st.caption("Add a person in the People tab before saving a memory note.")
                st.caption("The selected person is linked manually; the AI has not verified their identity in the image.")
            elif latest:
                st.info("A previous image was analyzed. Analyze the currently selected image to refresh this result before saving it, or open the Voice tab to replay the last observation.")
            else:
                st.info("Your image description will appear here after you analyze an image.")

with people_tab:
    st.subheader("People vault")
    st.write("Use fictional demonstration profiles. This is a shared cloud demo vault for anyone with the app password.")
    with st.form("add_person_form", clear_on_submit=True):
        a, b = st.columns(2)
        name = a.text_input("Display name", max_chars=100)
        relationship = b.text_input("Relationship / label", max_chars=100, placeholder="Example: Family member")
        notes = st.text_area("Optional context", max_chars=1000, height=90)
        add = st.form_submit_button("Add person", type="primary")
        if add:
            try:
                save_person(name, relationship, notes)
                st.success("Person added.")
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
    current_people = list_people()
    if current_people:
        st.dataframe([{"Name": p["name"], "Label": p["relationship"], "Notes": p["notes"], "Added": p["created_at"]} for p in current_people], use_container_width=True, hide_index=True)
    else:
        st.info("No people added yet.")

with memory_tab:
    st.subheader("Curated memory notes")
    current_people = list_people()
    if not current_people:
        st.info("Add a person first in the People tab.")
    else:
        labels = {p["id"]: f'{p["name"]} — {p["relationship"]}' for p in current_people}
        chosen_id = st.selectbox("Person", list(labels), format_func=lambda pid: labels[pid], key="memory_person_filter")
        with st.form("manual_memory_form", clear_on_submit=True):
            note = st.text_area("Memory or helpful context", max_chars=4000, height=130, placeholder="A short, factual note that a caregiver has reviewed.")
            submitted = st.form_submit_button("Save memory note", type="primary")
            if submitted:
                try:
                    save_memory(chosen_id, note, "Manually entered")
                    st.success("Memory saved to the cloud demo vault.")
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))
        person_memories = [m for m in list_memories(limit=500) if m["person_id"] == chosen_id]
        if person_memories:
            for memory in person_memories:
                with st.container(border=True):
                    st.caption(f'{memory["created_at"]} · {memory["source"]}')
                    st.write(memory["note"])
        else:
            st.info("No memory notes saved for this person yet.")

with voice_tab:
    st.subheader("Voice companion")
    st.write("Read visual observations and reviewed memory notes aloud using speech voices available on this device's browser. Speech is generated in the browser; this module does not send the text to a separate text-to-speech service or create an audio file.")
    settings_col, rate_col = st.columns([1, 1])
    with settings_col:
        st.selectbox(
            "Voice language",
            options=list(SUPPORTED_LANGUAGES),
            format_func=lambda code: SUPPORTED_LANGUAGES[code],
            key="voice_language",
            help="The browser chooses the closest installed voice for this language. Voice availability depends on the device.",
        )
    with rate_col:
        st.slider("Speaking speed", min_value=0.65, max_value=1.25, value=0.92, step=0.05, key="voice_rate")

    latest_for_voice = st.session_state.get("latest_result")
    if "voice_text" not in st.session_state:
        st.session_state["voice_text"] = latest_for_voice.get("text", "") if latest_for_voice else ""

    action_col1, action_col2 = st.columns(2)
    with action_col1:
        if st.button("Load latest visual observation", use_container_width=True, disabled=not bool(latest_for_voice)):
            st.session_state["voice_text"] = latest_for_voice["text"]
            st.rerun()
    available_memories = list_memories(limit=500)
    if available_memories:
        memory_labels = {
            row["id"]: f'{row["person_name"]} · {row["created_at"]} · {row["note"][:75]}'
            for row in available_memories
        }
        with action_col2:
            memory_to_read = st.selectbox(
                "Saved note to read",
                options=list(memory_labels),
                format_func=lambda memory_id: memory_labels[memory_id],
                key="voice_memory_choice",
                label_visibility="collapsed",
            )
            if st.button("Load selected memory note", use_container_width=True):
                chosen_memory = next((row for row in available_memories if row["id"] == memory_to_read), None)
                if chosen_memory:
                    st.session_state["voice_text"] = chosen_memory["note"]
                    st.rerun()
    else:
        with action_col2:
            st.caption("Save a memory first to load it here.")

    voice_text = st.text_area(
        "Text to read aloud",
        key="voice_text",
        max_chars=4000,
        height=160,
        placeholder="Load a visual observation, select a saved note, or type a short reminder…",
    )
    if voice_text.strip():
        try:
            st.components.v1.html(
                build_speech_html(
                    voice_text,
                    language=st.session_state.get("voice_language", "en-IN"),
                    rate=float(st.session_state.get("voice_rate", 0.92)),
                    widget_label="Speak text",
                ),
                height=92,
                scrolling=False,
            )
        except ValueError as exc:
            st.warning(str(exc))
    else:
        st.info("Enter text or load an observation or saved note. Then tap Speak text in the player.")
    st.caption("Tip: Chrome and Edge generally have broad Web Speech API support. Available voices are provided by your device/browser and can differ by language. If speech does not start, check device volume and browser support.")

with search_tab:
    st.subheader("Search saved memories")
    query = st.text_input("Search names, labels, notes, and observations", max_chars=200)
    if query.strip():
        found = search_memories(query.strip(), limit=100)
        if found:
            st.dataframe([{"Person": x["person_name"], "Label": x["relationship"], "Date": x["created_at"], "Memory": x["note"], "Source": x["source"]} for x in found], use_container_width=True, hide_index=True)
        else:
            st.info("No matching records.")
    with st.expander("Export demo memory notes"):
        st.caption("Anyone with the app password can export this shared demo vault. Do not store sensitive personal data here.")
        st.download_button("Download memories as CSV", data=export_memories_csv(), file_name="visionquest_demo_memories.csv", mime="text/csv", use_container_width=True)

with settings_tab:
    st.subheader("Cloud deployment status")
    ready, status_message = model_status()
    st.write(f"**Vision engine:** {status_message}")
    st.write("**Model:** HuggingFaceTB/SmolVLM-256M-Instruct")
    st.write("**OpenCV:** Headless OpenCV processes each live frame and renders selectable diagnostic views with detection overlays.")
    st.write(f"**Object detector:** {REALTIME_DETECTOR_LABEL}. The ONNX model runs on the app host using OpenCV DNN for each frame delivered by WebRTC; object detection does not call an external inference API.")
    st.write("**Camera:** Browser still capture plus WebRTC live stream. OpenCV processes each incoming frame on the cloud host. The Live camera & recording tab supports periodic AI analysis and download of silent MP4 clips up to 60 seconds/25 MB; WebRTC depends on camera permissions and STUN/TURN network access.")
    st.write("**Voice:** Browser-native speech synthesis reads observations and manually selected memory notes aloud; no external TTS API key is required.")
    st.write("**Memory storage:** Shared SQLite demo database on the host. Its filesystem may reset on redeploy/restart; backups and durable production database setup are not included.")
    st.write("**Privacy:** Image bytes are processed on the hosting server and are not intentionally saved as image files. Memory notes are stored on the host. The app uses one shared password, not individual user accounts.")
    st.write("**Not production-ready for care:** no individual accounts, role permissions, medical validation, guaranteed persistent storage, or automatic face identification.")
    if st.button("Lock this session", key="logout_button"):
        st.session_state["authenticated"] = False
        st.rerun()

st.divider()
st.caption("Smriti AI / VisionQuest Cloud Demo · Not a medical device · Use fictional demonstration data only")
