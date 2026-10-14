"""Minimal functional Smriti AI UI: camera, scene analysis, Q&A and spoken replies."""
import hashlib, html, threading, time
from datetime import datetime
import av, cv2, streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer
from core.cloud_vision import analyze_image, model_status
from core.cloud_voice import answer_voice_query, transcribe_audio
from core.cloud_db import search_activity_for_voice, save_conversation_turn, save_observation
from core.voice import build_speech_html

st.set_page_config(page_title="Smriti AI",page_icon="✦",layout="wide",initial_sidebar_state="collapsed")
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
st.markdown('<div class="brand">SMR // SMRITI AI</div><h1 style="margin:4px 0">Your visual memory companion</h1><div class="sub">LIVE CAMERA · VOICE COMPANION · CONTEXTUAL MEMORY</div><hr>',unsafe_allow_html=True)

class CamState:
 def __init__(self): self.lock=threading.Lock(); self.frame=None
class Processor:
 def __init__(self): self.state=st.session_state.cam_state
 def recv(self,frame):
  arr=frame.to_ndarray(format="bgr24")
  with self.state.lock: self.state.frame=arr.copy()
  return av.VideoFrame.from_ndarray(arr,format="bgr24")
defaults={"cam_state":CamState(),"analysis":"No scene analysis yet. Start the camera and analyse a frame.","analysis_time":"Not run","chat":[],"audio_hash":"","status":"READY"}
for k,v in defaults.items():
 if k not in st.session_state: st.session_state[k]=v

l,r=st.columns([.8,1.5],gap="large")
with l:
 st.markdown('<div class="panel"><div class="kicker">01 / COMPANION CORE</div><div class="orbwrap"><div class="orb"></div></div><div class="status">✦ SMRITI ONLINE</div><p style="text-align:center;color:#8aa6b5;font-size:13px">Ask a question. Get an AI answer and play it aloud.</p></div>',unsafe_allow_html=True)
 st.markdown('<div class="panel"><div class="kicker">02 / ASK SMRITI</div>',unsafe_allow_html=True)
 question=st.text_input("Question",placeholder="What can you see right now?",label_visibility="collapsed",key="question")
 send=st.button("SEND QUESTION",use_container_width=True)
 audio=st.audio_input("Or record a voice question",sample_rate=16000,key="voice_input")
 ah=hashlib.sha256(audio.getvalue()).hexdigest() if audio else ""
 send_audio=bool(audio and ah!=st.session_state.audio_hash)
 st.markdown('</div>',unsafe_allow_html=True)
with r:
 st.markdown('<div class="panel"><div class="kicker">03 / LIVE CAMERA FEED</div>',unsafe_allow_html=True)
 webrtc_streamer(key="smriti-camera",mode=WebRtcMode.SENDRECV,video_processor_factory=Processor,media_stream_constraints={"video":{"width":{"ideal":1280},"height":{"ideal":720},"frameRate":{"ideal":20,"max":24}},"audio":False},async_processing=True)
 st.caption("Allow camera access. Scene analysis runs when requested.")
 analyze=st.button("ANALYZE CURRENT FRAME",use_container_width=True)
 st.markdown('</div>',unsafe_allow_html=True)
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

if analyze:
 try:
  with st.session_state.cam_state.lock: frame=st.session_state.cam_state.frame.copy() if st.session_state.cam_state.frame is not None else None
  if frame is None: st.error("No camera frame received yet. Start the camera, wait, then retry.")
  else:
   st.session_state.status="ANALYSING FRAME"
   with st.spinner("The vision model is analysing the current frame…"):
    ok,buf=cv2.imencode(".jpg",frame,[int(cv2.IMWRITE_JPEG_QUALITY),82])
    if not ok: raise RuntimeError("Could not encode camera frame")
    res=analyze_image(buf.tobytes(),mode="auto_observe",max_side=768)
   st.session_state.analysis=res.text; st.session_state.analysis_time=datetime.now().strftime("%H:%M:%S"); st.session_state.status=f"ANALYSIS COMPLETE · {res.elapsed_seconds:.1f}s"
   try: save_observation(res.text,source="Simple Smriti live camera",model=res.model,frame_hash=hashlib.sha256(buf.tobytes()).hexdigest())
   except Exception: pass
   st.rerun()
 except Exception as e: st.session_state.status="ANALYSIS ERROR"; st.error(f"Scene analysis failed: {type(e).__name__}: {e}")

if send or send_audio:
 try:
  transcript=question.strip() if send else ""
  if send_audio:
   raw=audio.getvalue()
   with st.spinner("Transcribing your voice…"): transcript=transcribe_audio(raw,content_type=getattr(audio,"type",None) or "audio/wav")
   st.session_state.audio_hash=ah
  if not transcript: st.warning("Type a question or record a voice question first.")
  else:
   st.session_state.status="THINKING / GENERATING REPLY"
   with st.spinner("Smriti is preparing a reply…"):
    with st.session_state.cam_state.lock: frame=st.session_state.cam_state.frame.copy() if st.session_state.cam_state.frame is not None else None
    live=None
    if frame is not None:
     ok,buf=cv2.imencode(".jpg",frame,[int(cv2.IMWRITE_JPEG_QUALITY),78])
     if ok: live=buf.tobytes()
    memories=search_activity_for_voice(transcript,limit=8)
    reply=answer_voice_query(transcript,latest_observation={"text":st.session_state.analysis,"timestamp":st.session_state.analysis_time,"mode":"auto_observe"},memories=memories,conversation_history=[{"question":x["q"],"reply":x["a"]} for x in st.session_state.chat[-4:]],live_image_bytes=live)
   st.session_state.chat.append({"q":transcript,"a":reply}); st.session_state.chat=st.session_state.chat[-12:]; st.session_state.status="REPLY READY"
   try: save_conversation_turn(transcript,reply,source="Simple Smriti",used_live_camera=live is not None)
   except Exception: pass
   st.rerun()
 except Exception as e: st.session_state.status="VOICE / ANSWER ERROR"; st.error(f"Could not complete the conversation: {type(e).__name__}: {e}")
if st.session_state.chat:
 with st.expander("🔊 Speak latest reply"):
  st.components.v1.html(build_speech_html(st.session_state.chat[-1]["a"],language="en-IN",rate=.92,widget_label="▶ SPEAK REPLY",autoplay=False),height=100,scrolling=False)
with st.expander("Service status / setup"):
 ready,msg=model_status(); st.write("AI service:",msg)
 st.write("Voice input is record-then-send. Spoken output uses the browser speech engine; press Speak latest reply if autoplay is blocked.")
 st.caption("Configure HF_TOKEN in Streamlit Secrets or environment variables. Never commit API keys. Camera access requires browser permission and HTTPS.")
