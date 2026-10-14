"""Minimal functional Smriti AI UI: camera, scene analysis, Q&A and spoken replies."""
import hashlib, html, threading, time
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

st.set_page_config(page_title="Smriti AI",page_icon="✦",layout="wide",initial_sidebar_state="collapsed")

experience_mode = st.radio(
    "Choose experience",
    ["Live Mode", "Demo Mode"],
    horizontal=True,
    key="experience_mode",
    help="Demo Mode uses scripted sample replies without cloud AI requests; face matching runs locally when available.",
)
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

class CamState:
 def __init__(self): self.lock=threading.Lock(); self.frame=None
class Processor:
 def __init__(self, state):
  # Receive the per-session state explicitly. WebRTC callbacks execute on a
  # worker thread, so they must not look up st.session_state themselves.
  self.state = state
 def recv(self, frame):
  arr = frame.to_ndarray(format="bgr24")
  with self.state.lock:
   self.state.frame = arr.copy()
  return av.VideoFrame.from_ndarray(arr, format="bgr24")
defaults={
 "cam_state":CamState(),
 "analysis":"No scene analysis yet. Start the camera and observe a frame.",
 "analysis_time":"Not run","chat":[],"audio_hash":"","voice_event_id":"",
 "last_transcript":"","speech_text":"","speech_seq":0,"status":"READY",
 "opencv_result_rgb":None,"opencv_metrics":{},"opencv_detections":[],
 "observer_last_run_monotonic":0.0
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
   if voice_event.get("event")=="transcript": voice_question=str(voice_event.get("transcript","")).strip()
   elif voice_event.get("event")=="status": st.session_state.status=str(voice_event.get("message","READY"))[:100]
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
    "width":{"ideal":640,"max":1280},
    "height":{"ideal":480,"max":720},
    "frameRate":{"ideal":15,"max":20},
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

 with st.expander("04 / OPENCV ANALYSER + OBJECT DETECTOR", expanded=False):
  st.caption("OpenCV operators run on the current camera frame. Optional YOLOX object detection uses OpenCV DNN on the CPU; first use downloads its compact model.")
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
  enhance_cv = st.checkbox("Enhance local contrast before analysis",value=False,key="smriti_cv_enhance")
  detect_objects = st.checkbox("Also detect common objects (OpenCV DNN / YOLOX)",value=False,key="smriti_cv_detect_objects")
  confidence_pct = st.slider("Minimum object confidence",15,80,35,5,key="smriti_cv_confidence")
  if st.button("RUN OPENCV ANALYSER",use_container_width=True,key="smriti_run_cv"):
   try:
    with st.session_state.cam_state.lock:
     cv_frame = st.session_state.cam_state.frame.copy() if st.session_state.cam_state.frame is not None else None
    if cv_frame is None:
     st.warning("Start the camera and wait for a live frame first.")
    else:
     with st.spinner("Processing the current frame with OpenCV…"):
      processed, _, metrics = process_live_frame(
       cv_frame,max_side=960,enhance=enhance_cv,encode_jpeg=False,compute_metrics=True
      )
      display = render_cv_view(
       processed,mode=cv_mode,canny_low=canny_low,canny_high=canny_high,
       frame_count=int(time.monotonic()),
      )
      detections = []
      if detect_objects:
       with st.spinner("Loading/running local OpenCV YOLOX object detection…"):
        detector = get_local_object_detector()
        detection_result = detector.detect(
         processed,confidence_threshold=confidence_pct/100.0,max_detections=30
        )
        detections = detection_result.detections
        display = draw_detection_overlay(display,detections,age_seconds=0.0,max_age_seconds=1.5)
        metrics["detector_model"] = REALTIME_DETECTOR_LABEL
        metrics["detector_ms"] = round(detection_result.elapsed_seconds*1000.0,1)
     st.session_state.opencv_result_rgb = cv2.cvtColor(display,cv2.COLOR_BGR2RGB)
     st.session_state.opencv_metrics = metrics
     st.session_state.opencv_detections = detections
   except Exception as cv_exc:
    st.error(f"OpenCV analyser failed: {type(cv_exc).__name__}: {cv_exc}")
  if st.session_state.opencv_result_rgb is not None:
   st.image(st.session_state.opencv_result_rgb,caption="OpenCV analyser output · processed frame",use_container_width=True)
   m = st.session_state.opencv_metrics
   metrics_cols = st.columns(4)
   metrics_cols[0].metric("Frame size",f"{m.get('width','—')} × {m.get('height','—')}")
   metrics_cols[1].metric("Brightness",str(m.get("brightness","—")))
   metrics_cols[2].metric("Sharpness",str(m.get("sharpness","—")))
   metrics_cols[3].metric("Objects",str(len(st.session_state.opencv_detections)))
   if m.get("brightness_label"):
    st.caption(f"Lighting: {m['brightness_label']} · Focus estimate: {m.get('sharpness_label','unavailable')}")
   if m.get("detector_model"):
    st.caption(f"Detector: {m['detector_model']} · inference {m.get('detector_ms','—')} ms")
   if st.session_state.opencv_detections:
    st.write("**Detected objects**")
    for item in st.session_state.opencv_detections:
     st.write(f"• {item['label']} — {item['score']:.0%} confidence")
   elif m.get("detector_model"):
    st.info("No supported objects exceeded the selected confidence threshold in this frame.")
@st.fragment(run_every=10)
def render_live_observer():
 st.markdown('<div class="panel"><div class="kicker">05 / LIVE OBSERVER</div><h3>Scene Observer // Local Granite Vision</h3></div>',unsafe_allow_html=True)
 st.caption("Observe the current webcam frame on demand, or enable periodic local vision analysis. Automatic observation is off by default to keep your PC responsive.")
 auto_observe = st.checkbox("Enable automatic scene observation",value=False,key="smriti_auto_observer")
 interval = st.select_slider("Automatic observation interval",options=[15,30,60],value=30,format_func=lambda n:f"Every {n} seconds",key="smriti_observer_interval")
 run_now = st.button("OBSERVE CURRENT FRAME NOW",use_container_width=True,key="smriti_observe_now")
 now = time.monotonic()
 due = auto_observe and (now-float(st.session_state.get("observer_last_run_monotonic",0.0)) >= interval)
 if run_now or due:
  st.session_state.observer_last_run_monotonic = now
  with st.session_state.cam_state.lock:
   observer_frame = st.session_state.cam_state.frame.copy() if st.session_state.cam_state.frame is not None else None
  if observer_frame is None:
   st.warning("No live frame is available. Start the camera, wait for the video, and try again.")
  else:
   try:
    with st.spinner("Observer is analysing the current frame with local Granite…"):
     ok, encoded = cv2.imencode(".jpg",observer_frame,[int(cv2.IMWRITE_JPEG_QUALITY),82])
     if not ok:
      raise RuntimeError("OpenCV could not encode the current camera frame.")
     image_bytes = encoded.tobytes()
     result = analyze_with_granite(image_bytes,mode="auto_observe",max_side=768)
    st.session_state.analysis = result.text
    st.session_state.analysis_time = datetime.now().strftime("%H:%M:%S")
    st.session_state.status = f"OBSERVER COMPLETE · {result.elapsed_seconds:.1f}s"
    try:
     save_observation(result.text,source="Smriti local Live Observer",model=result.model,frame_hash=hashlib.sha256(image_bytes).hexdigest())
    except Exception:
     pass
   except Exception as observer_exc:
    st.session_state.status = "OBSERVER ERROR"
    st.error(f"Live Observer failed: {type(observer_exc).__name__}: {observer_exc}")
 st.markdown("**LATEST OBSERVATION**")
 st.write(st.session_state.analysis)
 st.caption(f"Last observation: {st.session_state.analysis_time} · {st.session_state.status}")

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
   if mode == "LIVE MODE":
    try: save_conversation_turn(transcript,reply,source="Simple Smriti",used_live_camera=live is not None)
    except Exception: pass
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
 st.write("Voice uses browser speech recognition in a hands-free turn cycle after you press ACTIVATE LISTENING. The browser pauses while Smriti answers and resumes after speaking; typed questions remain available.")
 st.caption("For Live Mode, run Ollama on this same PC and confirm the Granite model appears in ollama list. This local Ollama connection will not work from a cloud-hosted Streamlit server.")
