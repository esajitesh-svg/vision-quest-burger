"""Minimal functional Smriti AI UI: camera, scene analysis, Q&A and spoken replies."""
import base64, hashlib, html, os, subprocess, threading, time, uuid
from datetime import datetime
import av, cv2, streamlit as st
from pathlib import Path
from streamlit_webrtc import WebRtcMode, webrtc_streamer
from core.local_granite import analyze_with_granite, ask_granite, local_granite_status, GRANITE_MODEL
from core.cloud_live import process_live_frame, render_cv_view, draw_detection_overlay
from core.realtime_detector import REALTIME_DETECTOR_LABEL, RealtimeYOLOXDetector
from core.cloud_db import (
 init_db, list_people, save_person, save_person_portrait, get_person_portrait,
 search_activity_for_voice, save_conversation_turn, save_observation,
)
from core.demo_mode import demo_observation, demo_reply
from core.face_recognition import (
 decode_image_bytes, detect_faces, encode_jpeg, recognize_faces,
)
from core.voice import build_speech_html
from core.demo_mode import render_demo_hub
from core.local_archive import get_archive_dir, archive_event, save_media, write_text_export

st.set_page_config(page_title="Smriti AI",page_icon="✦",layout="wide",initial_sidebar_state="collapsed")

experience_mode = st.radio(
    "Choose experience",
    ["Live Mode", "Demo Mode"],
    horizontal=True,
    key="experience_mode",
    help="Demo Mode uses scripted sample replies without cloud AI requests; face matching runs locally when available.",
)

with st.expander("LOCAL MEMORY + CAPTURE ARCHIVE", expanded=True):
    st.caption("Automatic text memory is local. Audio and video clips are recorded only when you explicitly press their recording buttons; this app does not secretly record people.")
    st.checkbox("Automatically save observations, detected objects, and conversation text", value=True, key="archive_auto_save")
    st.checkbox("Save a camera still with each completed live observation", value=True, key="archive_save_observation_frames")
    st.text_input("People / names (manual labels only)", key="archive_people_tags", placeholder="Names you have permission to record")
    st.text_input("Place / location label (entered manually)", key="archive_place_tag", placeholder="e.g. home, school, garden")
    st.text_input("Activity / context", key="archive_activity_tag", placeholder="e.g. reading, gardening, meeting")
    try:
        archive_root = get_archive_dir()
        st.caption(f"Local archive folder: {archive_root}")
        activity_log_path = archive_root / "activity.jsonl"
        if activity_log_path.is_file():
            st.download_button("Download local activity log", activity_log_path.read_bytes(), file_name="smriti_activity.jsonl", mime="application/x-ndjson", key="download_local_activity_log")
    except Exception as archive_exc:
        st.warning(f"Local archive is unavailable: {type(archive_exc).__name__}: {archive_exc}")

def current_archive_tags():
    return {
        "people": st.session_state.get("archive_people_tags", ""),
        "place": st.session_state.get("archive_place_tag", ""),
        "activity": st.session_state.get("archive_activity_tag", ""),
    }

if experience_mode == "Demo Mode":
    render_demo_hub()
    st.stop()

browser_voice_component=st.components.v1.declare_component("smriti_browser_voice",path=str(Path(__file__).resolve().parent/"components"/"browser_voice"))
st.markdown("""<style>
.stApp,[data-testid="stAppViewContainer"],[data-testid="stMain"]{background:#050a10;color:#d9edf4}
[data-testid="stHeader"]{background:#050a10}[data-testid="stMainBlockContainer"]{max-width:1500px;padding:1rem 1.4rem}
h1,h2,h3,p,label{color:#d9edf4}.panel{background:linear-gradient(145deg,#0b1821,#071018);border:1px solid #21414c;border-radius:16px;padding:18px;margin-bottom:12px}
.kicker{font-size:11px;letter-spacing:3px;color:#5ef4e5;margin-bottom:12px}.brand{font-size:12px;letter-spacing:5px;color:#5ef4e5}.sub{font-size:12px;color:#8aa6b5;letter-spacing:2px}
.orbwrap{display:flex;justify-content:center;align-items:center;min-height:190px}.orb{width:125px;height:125px;border-radius:50%;border:1px solid #69f7ed;background:radial-gradient(circle at 45% 40%,#b8fff7 0,#3ef2e1 5%,#087c88 22%,#062c3a 48%,#07111b 70%);box-shadow:0 0 18px #18daca77,0 0 65px #13d6d633,inset 0 0 25px #51fff066;animation:pulse 3s ease-in-out infinite}
@keyframes pulse{50%{transform:scale(1.04);filter:brightness(1.25)}}.status{font-size:12px;color:#75f4e7;text-align:center;letter-spacing:2px}
.reply{background:#0a1b23;border-left:2px solid #50ebdf;border-radius:4px;padding:12px;line-height:1.6;white-space:pre-wrap}.log{font-family:monospace;color:#9ab7c2;font-size:12px;line-height:1.9}
div.stButton>button{background:#0e343c;color:#bffff8;border:1px solid #287c7b;border-radius:8px;min-height:42px}[data-testid="stTextInput"] input{background:#08151c;color:#e5fbff;border-color:#244a55}
</style>""",unsafe_allow_html=True)
st.markdown('<div class="brand">SMR // SMRITI AI</div><h1 style="margin:4px 0">Your visual memory companion</h1><div class="sub">LIVE CAMERA · VOICE COMPANION · CONTEXTUAL MEMORY</div>',unsafe_allow_html=True)
# Use the single top-level selector as the source of truth. A second selector
# defaulting to DEMO MODE previously caused Analyze to return sample text even
# while the user had selected Live Mode above.
mode = experience_mode.upper()
st.caption("LIVE MODE — camera analysis and answers use local Granite via Ollama.")
st.markdown("<hr>", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def get_local_object_detector():
 """Load the OpenCV DNN object detector once; weights are cached locally."""
 return RealtimeYOLOXDetector()

class ObserverState:
 """Thread-safe mailbox used by Smriti's asynchronous live observer."""
 def __init__(self):
  self.lock = threading.Lock()
  self.busy = False
  self.last_started = 0.0
  self.text = "Waiting for a live camera frame."
  self.time_label = "Not run"
  self.status = "WAITING FOR CAMERA"
  self.error = ""
  self.elapsed_seconds = 0.0

def run_observer_worker(state, image_bytes, max_side, tags, auto_save, save_frame):
 """Generate one Granite scene observation off the UI thread and persist it locally."""
 started = time.perf_counter()
 try:
  result = analyze_with_granite(image_bytes, mode="auto_observe", max_side=max_side)
  time_label = datetime.now().strftime("%H:%M:%S")
  status = f"OBSERVER COMPLETE · {result.elapsed_seconds:.1f}s"
  errors = []
  if auto_save:
   frame_hash = hashlib.sha256(image_bytes).hexdigest()
   tag_line = " · ".join(f"{key}: {value}" for key, value in tags.items() if value)
   stored_text = f"[{tag_line}] — {result.text}" if tag_line else result.text
   try:
    save_observation(stored_text, source="Smriti local Live Observer", model=result.model, frame_hash=frame_hash)
   except Exception as save_exc:
    errors.append(f"Database save failed: {type(save_exc).__name__}: {save_exc}")
   try:
    log = archive_event("observation", {
     "observation": result.text,
     "model": result.model,
     "inference_seconds": result.elapsed_seconds,
     "frame_sha256": frame_hash,
    }, tags)
    if save_frame:
     save_media(
      "frames", image_bytes, "jpg", "image/jpeg", tags,
      {"model": result.model, "inference_seconds": result.elapsed_seconds, "frame_sha256": frame_hash},
     )
   except Exception as archive_exc:
    errors.append(f"Local file archive failed: {type(archive_exc).__name__}: {archive_exc}")
  with state.lock:
   state.text = result.text
   state.time_label = time_label
   state.status = status
   state.error = " · ".join(errors)
   state.elapsed_seconds = max(result.elapsed_seconds, time.perf_counter() - started)
 except Exception as exc:
  with state.lock:
   state.status = "OBSERVER ERROR"
   state.error = f"{type(exc).__name__}: {exc}"
 finally:
  with state.lock:
   state.busy = False

class CamState:
 def __init__(self):
  self.lock=threading.Lock()
  self.frame=None
  self.updated_at=0.0
  self.video_writer=None
  self.video_path=None
  self.video_started=0.0
  self.video_frames=0
  self.video_tags={}
  self.video_notice=""
class Processor:
 def __init__(self, state):
  # Receive the per-session state explicitly. WebRTC callbacks execute on a
  # worker thread, so they must not look up st.session_state themselves.
  self.state = state
 def recv(self, frame):
  arr = frame.to_ndarray(format="bgr24")
  now = time.monotonic()
  with self.state.lock:
   self.state.frame = arr.copy()
   self.state.updated_at = now
   writer = self.state.video_writer
   if writer is not None:
    try:
     if now - self.state.video_started >= 60:
      writer.release()
      self.state.video_writer = None
      self.state.video_notice = "60-second recording limit reached. Press SAVE MP4 to archive the clip."
     else:
      small = cv2.resize(arr, (640, 360), interpolation=cv2.INTER_AREA)
      writer.write(small)
      self.state.video_frames += 1
    except Exception as record_exc:
     try:
      writer.release()
     except Exception:
      pass
     self.state.video_writer = None
     self.state.video_notice = f"Video writer error: {type(record_exc).__name__}: {record_exc}"
  return av.VideoFrame.from_ndarray(arr, format="bgr24")

def start_local_video_capture(state, tags):
 """Begin a clearly user-started, max-60-second local MP4 recording."""
 with state.lock:
  if state.video_path is not None:
   raise RuntimeError("Save or discard the previous clip before starting another recording.")
  frame = state.frame.copy() if state.frame is not None else None
  frame_age = time.monotonic() - state.updated_at if state.updated_at else 9999.0
 if frame is None or frame_age > 5.0:
  raise RuntimeError("No fresh camera frame is available. Start the camera and wait a moment.")
 folder = get_archive_dir() / "video"
 folder.mkdir(parents=True, exist_ok=True)
 pending_path = folder / (".pending_" + uuid.uuid4().hex + ".mp4")
 writer = cv2.VideoWriter(
  str(pending_path), cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (640, 360)
 )
 if not writer.isOpened():
  writer.release()
  pending_path.unlink(missing_ok=True)
  raise RuntimeError("OpenCV could not initialise an MP4 writer on this computer.")
 with state.lock:
  state.video_path = pending_path
  state.video_writer = writer
  state.video_started = time.monotonic()
  state.video_frames = 0
  state.video_tags = dict(tags or {})
  state.video_notice = "Recording is active. Stop and save within 60 seconds."

def finalize_local_video_capture(state):
 """Stop recording and commit a completed clip into the tagged local archive."""
 with state.lock:
  writer = state.video_writer
  state.video_writer = None
  pending_path = state.video_path
  tags = dict(state.video_tags)
  frame_count = int(state.video_frames)
  started = float(state.video_started)
  notice = state.video_notice
  if writer is not None:
   writer.release()
 if pending_path is None or not pending_path.is_file():
  raise RuntimeError(notice or "No completed video clip is available to save.")
 try:
  data = pending_path.read_bytes()
  if not data or frame_count < 1:
   raise RuntimeError("The clip contains no usable frames; start recording after the camera feed appears.")
  duration = max(0.0, time.monotonic() - started) if started else frame_count / 15.0
  archive_path = save_media(
   "video", data, "mp4", "video/mp4", tags,
   {"frames": frame_count, "duration_seconds": min(60.0, duration), "captured_width": 640, "captured_height": 360},
  )
  try:
   archive_event("video_recording", {
    "file": str(archive_path), "frames": frame_count,
    "duration_seconds": round(min(60.0, duration), 2),
   }, tags)
   notice_text = f"Saved video clip: {archive_path}"
  except Exception as log_exc:
   notice_text = f"Video saved to {archive_path}; activity-log write failed: {type(log_exc).__name__}: {log_exc}"
  with state.lock:
   if state.video_path == pending_path:
    state.video_path = None
    state.video_frames = 0
    state.video_started = 0.0
    state.video_tags = {}
   state.video_notice = notice_text
  pending_path.unlink(missing_ok=True)
  return archive_path
 except Exception as save_exc:
  with state.lock:
   state.video_notice = f"Save failed; the temporary clip is kept so you can retry or discard it: {type(save_exc).__name__}: {save_exc}"
  raise

def discard_local_video_capture(state):
 """Discard an explicitly started recording without adding it to the archive."""
 with state.lock:
  writer = state.video_writer
  state.video_writer = None
  pending_path = state.video_path
  state.video_path = None
  state.video_frames = 0
  state.video_started = 0.0
  state.video_tags = {}
  state.video_notice = "Pending video clip discarded by user."
 if writer is not None:
  writer.release()
 if pending_path is not None:
  pending_path.unlink(missing_ok=True)

@st.fragment(run_every=1)
def render_video_archive_controls():
 with st.expander("LIVE VIDEO ARCHIVE · USER-CONTROLLED RECORDING", expanded=False):
  st.caption("Recording is never started automatically. Press START to capture a clip; it stops at 60 seconds. Clips are saved on this computer in the configured archive folder.")
  state = st.session_state.cam_state
  with state.lock:
   active = state.video_writer is not None
   has_pending = state.video_path is not None
   frame_age = time.monotonic() - state.updated_at if state.updated_at else 9999.0
   duration = min(60.0, time.monotonic() - state.video_started) if active else 0.0
   frames = state.video_frames
   notice = state.video_notice
  if not active and not has_pending:
   if st.button("START VIDEO RECORDING · MAX 60 SECONDS", use_container_width=True, key="start_local_video_recording", disabled=frame_age > 5.0):
    try:
     start_local_video_capture(state, current_archive_tags())
     archive_event("video_recording_started", {"limit_seconds": 60}, current_archive_tags())
     st.success("Video recording started.")
    except Exception as exc:
     st.error(f"Could not start video recording: {type(exc).__name__}: {exc}")
  elif active:
   st.error(f"● RECORDING VIDEO · {duration:.1f}/60 seconds · {frames} frames")
   st.progress(min(1.0, duration / 60.0))
  if has_pending:
   save_col, discard_col = st.columns(2)
   with save_col:
    if st.button("STOP & SAVE MP4", use_container_width=True, key="save_local_video_recording"):
     try:
      saved_path = finalize_local_video_capture(state)
      st.success(f"Video saved: {saved_path}")
     except Exception as exc:
      st.error(f"Could not save video: {type(exc).__name__}: {exc}")
   with discard_col:
    if st.button("DISCARD CLIP", use_container_width=True, key="discard_local_video_recording"):
     try:
      discard_local_video_capture(state)
      st.info("Pending clip discarded.")
     except Exception as exc:
      st.error(f"Could not discard clip: {type(exc).__name__}: {exc}")
   if notice:
    st.caption(notice)
  elif notice:
   st.caption(notice)
defaults={
 "cam_state":CamState(),
 "analysis":"No scene analysis yet. Start the camera and observe a frame.",
 "analysis_time":"Not run","chat":[],"audio_hash":"","voice_event_id":"",
 "last_transcript":"","speech_text":"","speech_seq":0,"status":"READY",
 "opencv_result_rgb":None,"opencv_metrics":{},"opencv_detections":[],
 "opencv_detection_last_run":0.0,"opencv_detection_error":"",
 "opencv_detection_elapsed":0.0,"opencv_frame_counter":0,"opencv_detector_backend":"Not loaded",
 "observer_last_run_monotonic":0.0,
 "observer_state":ObserverState()
}
for k,v in defaults.items():
 if k not in st.session_state: st.session_state[k]=v
if "smriti_db_init_attempted" not in st.session_state:
 try:
  init_db()
  st.session_state.smriti_db_init_error = ""
 except Exception as db_exc:
  st.session_state.smriti_db_init_error = f"{type(db_exc).__name__}: {db_exc}"
 st.session_state.smriti_db_init_attempted = True

@st.fragment(run_every=1)
def render_live_analyser():
 with st.expander("04 / LIVE OPENCV ANALYSER + OBJECT DETECTOR", expanded=True):
  st.caption("This panel refreshes from the newest camera frame about once per second. OpenCV display processing is local; optional YOLOX labels run locally on the CPU.")
  cv_mode = st.selectbox(
   "OpenCV display operator",
   ["COLOR + HUD","GRAYSCALE","CANNY EDGE MAP","CONTOUR MAP","ADAPTIVE THRESHOLD","HSV SATURATION"],
   key="smriti_cv_mode",
  )
  ca, cb = st.columns(2)
  with ca:
   canny_low = st.slider("Canny lower threshold",0,200,55,5,key="smriti_canny_low")
  with cb:
   canny_high = st.slider("Canny upper threshold",20,255,140,5,key="smriti_canny_high")
  enhance_cv = st.checkbox("Enhance local contrast before display",value=False,key="smriti_cv_enhance")
  detect_objects = st.checkbox("Enable live object detection (OpenCV DNN / YOLOX)",value=False,key="smriti_cv_detect_objects")
  confidence_pct = st.slider("Minimum object confidence",15,80,35,5,key="smriti_cv_confidence")
  now = time.monotonic()
  with st.session_state.cam_state.lock:
   cv_frame = st.session_state.cam_state.frame.copy() if st.session_state.cam_state.frame is not None else None
   frame_age = now - st.session_state.cam_state.updated_at if st.session_state.cam_state.updated_at else 9999.0

  if cv_frame is None or frame_age > 5.0:
   st.info("Waiting for a fresh camera frame. Start the camera and keep the live feed running.")
  else:
   try:
    processed, _, metrics = process_live_frame(
     cv_frame,max_side=960,enhance=enhance_cv,encode_jpeg=False,compute_metrics=True
    )
    st.session_state.opencv_frame_counter += 1
    display = render_cv_view(
     processed,mode=cv_mode,canny_low=canny_low,canny_high=canny_high,
     frame_count=st.session_state.opencv_frame_counter,
    )
    if detect_objects and not st.session_state.opencv_detection_error:
     run_detector = (
      st.session_state.opencv_detection_last_run <= 0.0
      or now - st.session_state.opencv_detection_last_run >= 2.5
     )
     if run_detector:
      try:
       with st.spinner("Loading/running local YOLOX detector… first run downloads the model"):
        detector = get_local_object_detector()
        result = detector.detect(
         processed,confidence_threshold=confidence_pct/100.0,max_detections=30
        )
       st.session_state.opencv_detections = result.detections
       st.session_state.opencv_detection_last_run = time.monotonic()
       st.session_state.opencv_detection_elapsed = result.elapsed_seconds
       st.session_state.opencv_detector_backend = getattr(detector, "acceleration_backend", "OpenCV DNN CPU")
       if st.session_state.get("archive_auto_save", True):
        try:
         archive_event("object_detection", {
          "detections": result.detections,
          "model": REALTIME_DETECTOR_LABEL,
          "inference_seconds": result.elapsed_seconds,
          "frame_width": result.image_size[0],
          "frame_height": result.image_size[1],
         }, current_archive_tags())
        except Exception as archive_exc:
         st.session_state.opencv_detection_error = "Detected objects, but local archive failed: " + str(archive_exc)
      except Exception as detector_exc:
       st.session_state.opencv_detection_error = f"{type(detector_exc).__name__}: {detector_exc}"
    elif not detect_objects:
     st.session_state.opencv_detections = []
     st.session_state.opencv_detection_error = ""
     st.session_state.opencv_detection_last_run = 0.0

    detection_age = (
     time.monotonic() - st.session_state.opencv_detection_last_run
     if st.session_state.opencv_detection_last_run > 0.0 else 9999.0
    )
    active_detections = (
     st.session_state.opencv_detections if detect_objects and detection_age <= 3.5 else []
    )
    if active_detections:
     display = draw_detection_overlay(display,active_detections,age_seconds=detection_age,max_age_seconds=3.5)
    st.session_state.opencv_result_rgb = cv2.cvtColor(display,cv2.COLOR_BGR2RGB)
    st.session_state.opencv_metrics = metrics
    st.image(st.session_state.opencv_result_rgb,caption="LIVE OpenCV output · refreshed from current camera frame",use_container_width=True)
    metrics_cols = st.columns(4)
    metrics_cols[0].metric("Frame size",f"{metrics.get('width','—')} × {metrics.get('height','—')}")
    metrics_cols[1].metric("Brightness",str(metrics.get("brightness","—")))
    metrics_cols[2].metric("Sharpness",str(metrics.get("sharpness","—")))
    metrics_cols[3].metric("Objects",str(len(active_detections)) if detect_objects else "Off")
    if metrics.get("brightness_label"):
     st.caption(f"Lighting: {metrics['brightness_label']} · Focus estimate: {metrics.get('sharpness_label','unavailable')} · frame age {frame_age:.1f}s")
    if detect_objects:
     if st.session_state.opencv_detection_error:
      st.error("Live object detector unavailable: " + st.session_state.opencv_detection_error)
      st.caption("Uncheck and recheck object detection to retry after correcting the issue.")
     else:
      st.caption(f"Detector: {REALTIME_DETECTOR_LABEL} · backend: {st.session_state.opencv_detector_backend} · last inference {st.session_state.opencv_detection_elapsed*1000:.0f} ms · detections refresh about every 2.5 seconds")
      if active_detections:
       st.write("**Current object labels**")
       for item in active_detections:
        st.write(f"• {item['label']} — {item['score']:.0%} confidence")
      elif st.session_state.opencv_detection_last_run > 0:
       st.caption("No supported objects exceeded the selected confidence threshold in the latest scan.")
   except Exception as cv_exc:
    st.error(f"Live OpenCV analyser failed: {type(cv_exc).__name__}: {cv_exc}")


l,r=st.columns([.8,1.5],gap="large")
with l:
 st.markdown('<div class="panel"><div class="kicker">01 / COMPANION CORE</div><div class="orbwrap"><div class="orb"></div></div><div class="status">✦ SMRITI ONLINE</div><p style="text-align:center;color:#8aa6b5;font-size:13px">Ask a question. Get an AI answer and play it aloud.</p></div>',unsafe_allow_html=True)
 st.markdown('<div class="panel"><div class="kicker">02 / ASK SMRITI</div>',unsafe_allow_html=True)
 question=st.text_input("Question",placeholder="What can you see right now?",label_visibility="collapsed",key="question")
 send=st.button("SEND QUESTION",use_container_width=True)
 voice_event=browser_voice_component(enabled=True,busy=st.session_state.status.startswith("THINKING"),language="en-IN",rate=.92,speech_text=st.session_state.speech_text,speech_sequence=st.session_state.speech_seq,speech_kind="reply",last_transcript=st.session_state.last_transcript,voice_error="",key="smriti_live_browser_voice",default=None)
 voice_question=""
 if isinstance(voice_event,dict):
  eid=str(voice_event.get("event_id",""))
  if eid and eid!=st.session_state.voice_event_id:
   st.session_state.voice_event_id=eid
   if voice_event.get("event")=="transcript":
    voice_question=str(voice_event.get("transcript","")).strip()
   elif voice_event.get("event")=="status":
    st.session_state.status=str(voice_event.get("message","READY"))[:100]
   elif voice_event.get("event")=="audio_archive":
    try:
     encoded_audio = str(voice_event.get("audio_base64",""))
     if not encoded_audio or len(encoded_audio) > 12_000_000:
      raise ValueError("Audio clip is empty or too large. Record a shorter clip.")
     audio_bytes = base64.b64decode(encoded_audio, validate=True)
     mime_type = str(voice_event.get("mime_type","audio/webm")).split(";")[0].strip().lower()
     extension = "m4a" if mime_type == "audio/mp4" else ("wav" if mime_type == "audio/wav" else "webm")
     path = save_media("audio", audio_bytes, extension, mime_type, current_archive_tags(),
                       {"maximum_recording_seconds":30, "source":"explicit browser audio recording"})
     archive_event("audio_recording", {"file":str(path), "maximum_recording_seconds":30}, current_archive_tags())
     st.success(f"Microphone clip saved locally: {path}")
    except Exception as audio_exc:
     st.error(f"Could not save the audio clip: {type(audio_exc).__name__}: {audio_exc}")
 st.markdown('</div>',unsafe_allow_html=True)
with r:
 st.markdown('<div class="panel"><div class="kicker">03 / LIVE CAMERA FEED</div>',unsafe_allow_html=True)
 camera_state = st.session_state.cam_state
 camera_ctx = webrtc_streamer(
  key="smriti-camera",
  mode=WebRtcMode.SENDRECV,
  video_processor_factory=lambda: Processor(camera_state),
  rtc_configuration={"iceServers":[{"urls":["stun:stun.l.google.com:19302"]}]},
  media_stream_constraints={
   "video":{
    "facingMode":"user",
    "width":{"ideal":960,"max":1280},
    "height":{"ideal":540,"max":720},
    "frameRate":{"ideal":20,"max":24},
   },
   "audio":False,
  },
  async_processing=True,
 )
 if camera_ctx.state.playing:
  st.success("Camera is streaming. You can analyse the current frame.")
 else:
  st.info("Click START in the camera controls and allow camera access in Chrome/Edge. For local use, open http://localhost:8501 rather than a LAN IP or plain HTTP remote link.")
 st.caption("Camera frames are kept in local app memory. OpenCV processing, optional object detection, and scene observation are available below.")
 st.markdown('</div>',unsafe_allow_html=True)

 render_live_analyser()
 render_video_archive_controls()

@st.fragment(run_every=1)
def render_live_observer():
 with st.expander("05 / LIVE OBSERVER // GRANITE VISION", expanded=True):
  st.caption("Camera input now requests up to 960×540 for clearer detail. Granite runs in the background and skips overlapping requests, so the UI stays responsive when inference takes longer than one second.")
  detail_label = st.selectbox(
   "Observation detail / speed",
   ["Fast · 512 px","Balanced · 640 px","Higher detail · 768 px"],
   index=1,
   key="smriti_observer_detail",
  )
  detail_side = 512 if detail_label.startswith("Fast") else (768 if detail_label.startswith("Higher") else 640)
  auto_observe = st.checkbox("Enable automatic observation every second", value=True, key="smriti_auto_observer")
  run_now = st.button("OBSERVE CURRENT FRAME NOW", use_container_width=True, key="smriti_observe_now")
  now = time.monotonic()
  observer_state = st.session_state.observer_state

  with observer_state.lock:
   busy = observer_state.busy
   last_started = observer_state.last_started

  should_start = run_now or (auto_observe and not busy and now - last_started >= 1.0)
  if should_start and not busy:
   with st.session_state.cam_state.lock:
    frame = st.session_state.cam_state.frame.copy() if st.session_state.cam_state.frame is not None else None
    updated_at = st.session_state.cam_state.updated_at
   frame_age = now - updated_at if updated_at else 9999.0

   if frame is None or frame_age > 5.0:
    st.info("Waiting for a fresh webcam frame. Start the camera and leave the feed running.")
   else:
    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 84])
    if not ok:
     st.error("OpenCV could not encode the current camera frame.")
    else:
     with observer_state.lock:
      if not observer_state.busy and (run_now or now - observer_state.last_started >= 1.0):
       observer_state.busy = True
       observer_state.last_started = now
       launch = True
      else:
       launch = False
     if launch:
      try:
       threading.Thread(
        target=run_observer_worker,
        args=(
         observer_state, encoded.tobytes(), detail_side, current_archive_tags(),
         bool(st.session_state.get("archive_auto_save", True)),
         bool(st.session_state.get("archive_save_observation_frames", True)),
        ),
        daemon=True,
        name="smriti-live-observer",
       ).start()
      except Exception as exc:
       with observer_state.lock:
        observer_state.busy = False
        observer_state.status = "OBSERVER ERROR"
        observer_state.error = f"Could not start observer worker: {type(exc).__name__}: {exc}"
  elif run_now and busy:
   st.info("The current frame is still being analysed. Please wait for this result; another request will not be queued.")

  with observer_state.lock:
   busy = observer_state.busy
   latest_text = observer_state.text
   time_label = observer_state.time_label
   status = observer_state.status
   error = observer_state.error
   elapsed = observer_state.elapsed_seconds

  if busy:
   st.info("Observer is analysing the latest frame in the background. It will pick up a new frame when this analysis finishes.")
  if error:
   st.warning(error)
  st.markdown("**LATEST LIVE OBSERVATION**")
  st.write(latest_text)
  st.caption(f"{status} · Last completed observation: {time_label}" + (f" · {elapsed:.1f}s inference" if elapsed else ""))

render_live_observer()


with st.expander("06 / FACE RECOGNITION — LOCAL ENROLMENT", expanded=False):
 st.write("Register a reference portrait with the person's permission, then compare faces in the current live camera frame. Portraits are stored in the configured app database and shared with Demo Mode.")
 st.caption("Experimental matching can make mistakes. Confirm identity before relying on it. Faces that do not match confidently are shown as Unknown.")
 consent = st.checkbox("I have permission to store and use this person's reference photo.", key="face_consent")
 person_name = st.text_input("Person's chosen display name", key="face_name", max_chars=100)
 relationship = st.text_input("Relationship / label", value="Friend / family", key="face_relationship", max_chars=100)
 portrait_file = st.file_uploader("Reference photo (JPEG, PNG, WEBP; up to 3 MB)", type=["jpg", "jpeg", "png", "webp"], key="face_portrait_upload")
 if st.button("REGISTER / UPDATE REFERENCE FACE", use_container_width=True, key="register_face_button"):
  try:
   if not consent:
    st.warning("Please confirm permission before registering a face.")
   elif not person_name.strip() or not relationship.strip():
    st.warning("Enter a display name and relationship/label.")
   elif portrait_file is None:
    st.warning("Choose a reference photo first.")
   else:
    portrait_bytes = portrait_file.getvalue()
    reference_frame = decode_image_bytes(portrait_bytes)
    if not detect_faces(reference_frame):
     st.warning("No face was detected in that photo. Choose a clear, front-facing photo with one visible face.")
    else:
     existing_people = list_people()
     existing = next((p for p in existing_people if str(p.get("name", "")).casefold() == person_name.strip().casefold()), None)
     if existing:
      person_id = int(existing["id"])
     else:
      save_person(person_name.strip(), relationship.strip(), "Face matching enabled only with permission.")
      person_id = int(next(p["id"] for p in list_people() if str(p.get("name", "")).casefold() == person_name.strip().casefold()))
     stored_bytes = encode_jpeg(reference_frame)
     save_person_portrait(person_id, stored_bytes, "image/jpeg")
     st.success(f"Reference portrait saved for {person_name.strip()}.")
  except Exception as face_exc:
   st.error(f"Could not register portrait: {type(face_exc).__name__}: {face_exc}")
 try:
  registered_people = list_people()
  with_portraits = []
  for person in registered_people:
   try:
    if get_person_portrait(int(person["id"])):
     with_portraits.append(person)
   except Exception:
    continue
  st.caption(f"Registered portraits available: {len(with_portraits)}")
 except Exception as list_exc:
  with_portraits = []
  st.warning(f"Person storage is unavailable: {type(list_exc).__name__}: {list_exc}")
 if st.button("RECOGNIZE FACES IN CURRENT CAMERA FRAME", use_container_width=True, key="recognize_live_faces"):
  try:
   with st.session_state.cam_state.lock:
    current_frame = st.session_state.cam_state.frame.copy() if st.session_state.cam_state.frame is not None else None
   if current_frame is None:
    st.warning("Start the camera and wait until a live frame arrives.")
   elif not with_portraits:
    st.warning("Register at least one consented reference portrait first.")
   else:
    with st.spinner("Matching current faces to locally stored reference portraits…"):
     annotated, matches = recognize_faces(current_frame, with_portraits, get_person_portrait)
    st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Local face matches · verify uncertain results", use_container_width=True)
    if not matches:
     st.info("No face was detected in the current frame.")
    else:
     for match in matches:
      label = match["name"] if match["recognized"] else "Unknown person"
      st.write(f"{label} · match distance {match['distance']} (lower is closer)")
  except Exception as face_exc:
   st.error(f"Face recognition could not run: {type(face_exc).__name__}: {face_exc}")

a,b=st.columns([1,1.2],gap="large")
with a:
 st.markdown('<div class="panel"><div class="kicker">04 / ANALYSIS STATUS</div>',unsafe_allow_html=True)
 st.markdown(f'<div class="log">STATUS / {html.escape(st.session_state.status)}<br>LAST RUN / {html.escape(st.session_state.analysis_time)}</div>',unsafe_allow_html=True)
 st.write(st.session_state.analysis); st.markdown('</div>',unsafe_allow_html=True)
with b:
 st.markdown('<div class="panel"><div class="kicker">05 / CONVERSATION</div>',unsafe_allow_html=True)
 if not st.session_state.chat: st.caption("Your questions and replies appear here.")
 for t in st.session_state.chat[-4:]:
  st.markdown("**YOU**"); st.write(t["q"]); st.markdown("**SMRITI**"); st.markdown(f'<div class="reply">{html.escape(t["a"])}</div>',unsafe_allow_html=True)
 st.markdown('</div>',unsafe_allow_html=True)

if send or voice_question:
 try:
  transcript=question.strip() if send else voice_question
  if not transcript: st.warning("Type a question or record a voice question first.")
  else:
   st.session_state.status="THINKING / GENERATING REPLY"
   with st.spinner("Smriti is preparing a reply…"):
    with st.session_state.cam_state.lock: frame=st.session_state.cam_state.frame.copy() if st.session_state.cam_state.frame is not None else None
    live=None
    if frame is not None:
     ok,buf=cv2.imencode(".jpg",frame,[int(cv2.IMWRITE_JPEG_QUALITY),78])
     if ok: live=buf.tobytes()
    if mode == "DEMO MODE":
     reply = demo_reply(transcript)
    else:
     try:
      memories=search_activity_for_voice(transcript,limit=8)
     except Exception:
      memories=[]
     observation_text = st.session_state.analysis
     latest = None if observation_text.startswith("DEMO SAMPLE") else {"text":observation_text,"timestamp":st.session_state.analysis_time,"mode":"auto_observe"}
     visual_terms = ("see", "look", "camera", "in front of me", "around me", "what is that", "what's that", "what is this", "what's this", "describe this", "describe what", "read this", "read the", "who is in front")
     needs_live_image = any(term in transcript.casefold() for term in visual_terms)
     # Avoid running a vision-model pass for ordinary chat; use image bytes only for visual questions.
     reply=ask_granite(transcript,latest_observation=latest,memories=memories,conversation_history=[{"question":x["q"],"reply":x["a"]} for x in st.session_state.chat[-4:]],image_bytes=live if needs_live_image else None)
   st.session_state.chat.append({"q":transcript,"a":reply}); st.session_state.chat=st.session_state.chat[-12:]; st.session_state.status="DEMO REPLY READY" if mode == "DEMO MODE" else "REPLY READY"
   st.session_state.last_transcript=transcript; st.session_state.speech_text=reply; st.session_state.speech_seq+=1
   if mode == "LIVE MODE" and st.session_state.get("archive_auto_save", True):
    try: save_conversation_turn(transcript,reply,source="Simple Smriti",used_live_camera=live is not None)
    except Exception: pass
   if st.session_state.get("archive_auto_save", True):
    try:
     archive_event("conversation_turn", {
      "question": transcript,
      "answer": reply,
      "mode": mode,
      "used_live_camera": live is not None,
     }, current_archive_tags())
    except Exception as archive_exc:
     st.warning(f"Conversation reply succeeded, but local archive failed: {type(archive_exc).__name__}: {archive_exc}")
   st.rerun()
 except Exception as e: st.session_state.status="VOICE / ANSWER ERROR"; st.error(f"Could not complete the conversation: {type(e).__name__}: {e}")
if st.session_state.chat:
 with st.expander("🔊 Speak latest reply"):
  st.components.v1.html(build_speech_html(st.session_state.chat[-1]["a"],language="en-IN",rate=.92,widget_label="▶ SPEAK REPLY",autoplay=False),height=100,scrolling=False)
with st.expander("Service status / setup"):
 if st.session_state.get("smriti_db_init_error"):
  st.write("Memory database:", st.session_state.smriti_db_init_error)
 else:
  st.write("Memory database: initialized")
 ready,msg=local_granite_status(); st.write("Local AI service:",msg)
 st.write("Selected mode:", mode)
 st.write("Demo Mode does not call an AI model. Live Mode uses local Ollama Granite: " + GRANITE_MODEL + ".")
 st.write("Voice uses browser speech recognition in a hands-free turn cycle after you press ACTIVATE LISTENING. Raw mic audio is saved only if you explicitly press RECORD AUDIO CLIP; video is saved only after you explicitly start recording.")
 st.caption("Observation text, detected-object labels, and conversations are archived locally when automatic saving is enabled. Audio/video recordings are separate user-started clips. Names and places are manually entered tags, not inferred identity or GPS.")
 try:
  archive_root = get_archive_dir()
  st.write("Local archive:", str(archive_root))
 except Exception as archive_exc:
  st.warning(f"Archive directory unavailable: {archive_exc}")
 try:
  opencl_available = bool(cv2.ocl.haveOpenCL())
  st.write("OpenCV OpenCL support:", "Available in this OpenCV build" if opencl_available else "Not available in this OpenCV build")
  st.caption("Local YOLOX tries OpenCL when available and falls back to CPU if the driver or device cannot execute it.")
 except Exception as accel_exc:
  st.write("OpenCV GPU diagnostics:", f"Unavailable ({type(accel_exc).__name__})")
 try:
  ps = subprocess.run(["ollama", "ps"], capture_output=True, text=True, timeout=3, check=False)
  if ps.returncode == 0 and ps.stdout.strip():
   st.write("Ollama loaded-model / processor status:")
   st.code(ps.stdout.strip()[:2000], language="text")
  elif ps.returncode == 0:
   st.caption("Ollama is installed, but no model is currently listed as loaded. Start a Live Observer request, then check this status again.")
  else:
   st.caption("Could not query the Ollama process status; verify Ollama is running in PowerShell.")
 except (OSError, subprocess.TimeoutExpired):
  st.caption("Ollama command-line diagnostics unavailable in this process; verify Ollama is running.")
 st.components.v1.html(
  """<div style="font:12px system-ui;padding:9px;border:1px solid #24505c;border-radius:8px;background:#07131a;color:#d7eef5">
  <b>Browser graphics capability</b><div id="smriti-gpu-diagnostic" style="margin-top:6px">Checking WebGL / WebGPU…</div>
  <div style="color:#87a5b2;margin-top:4px">WebGL/WebGPU support does not automatically accelerate Python or Ollama inference. Model acceleration is reported separately above.</div></div>
  <script>
  (async()=>{
   const node=document.getElementById("smriti-gpu-diagnostic");
   let gl="unavailable";
   try{const canvas=document.createElement("canvas");gl=canvas.getContext("webgl2")?"WebGL 2 available":(canvas.getContext("webgl")?"WebGL available":"WebGL unavailable");}catch(e){}
   let gpu="WebGPU unavailable";
   try{if(navigator.gpu){const adapter=await navigator.gpu.requestAdapter();gpu=adapter?"WebGPU adapter available in browser":"WebGPU API present, but no adapter";}}catch(e){gpu="WebGPU adapter unavailable in this context";}
   node.textContent=gl+" · "+gpu;
  })();
  </script>""",
  height=105,
 )
 st.caption("For local inference, run Ollama on this same PC and confirm granite3.2-vision:2b appears in ollama list. This local connection will not work from a cloud-hosted Streamlit server.")
