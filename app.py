"""Cloud-hosted Smriti AI / VisionQuest demo.

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
from datetime import datetime
from pathlib import Path

import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

from core.cloud_db import init_db, list_people, list_memories, save_person, save_memory, search_memories, export_memories_csv
from core.cloud_vision import MODE_LABELS, analyze_image, model_status, prepare_image
from core.voice import SUPPORTED_LANGUAGES, build_speech_html

ROOT = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Smriti AI · VisionQuest Cloud",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CSS = r"""
<style>
:root { color-scheme: light; }
html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] { background:#f4f6fc !important; color:#1a2440 !important; }
[data-testid="stMainBlockContainer"] { max-width:1320px; padding-top:1.2rem; padding-bottom:3rem; }
[data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li, [data-testid="stCaptionContainer"], [data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] *, label, .stRadio label, .stCheckbox label { color:#26314e !important; }
h1,h2,h3,h4 { color:#151f3b !important; letter-spacing:-.02em; }
a { color:#5548d8 !important; }
.hero { padding:2rem 2.2rem; border-radius:24px; margin:.15rem 0 1.2rem; color:#fff !important; background:radial-gradient(circle at 90% 0%,rgba(144,129,255,.5),transparent 30%),linear-gradient(115deg,#10182f 0%,#28285e 52%,#5f50d5 100%); box-shadow:0 15px 38px rgba(49,47,115,.14); }
.hero * { color:#fff !important; }
.hero .kicker { text-transform:uppercase; letter-spacing:.12em; font-size:.75rem; font-weight:800; opacity:.88; }
.hero h1 { font-size:clamp(2rem,3vw,2.75rem); margin:.65rem 0 .35rem; line-height:1.1; }
.hero p { color:#e3e7ff !important; max-width:800px; line-height:1.65; margin:0; }
.pill { display:inline-block; padding:.28rem .6rem; margin-top:.9rem; border:1px solid rgba(255,255,255,.26); background:rgba(255,255,255,.1); border-radius:999px; font-size:.78rem; }
[data-testid="stMetric"] { background:#fff !important; border:1px solid #e0e6f2 !important; border-radius:15px; padding:.85rem !important; }
[data-testid="stMetric"] * { color:#192541 !important; }
[data-baseweb="input"] input, [data-baseweb="textarea"] textarea, [data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea { background:#fff !important; color:#1b2541 !important; border-color:#cad3e4 !important; }
[data-baseweb="select"] * { color:#1b2541 !important; }
[data-testid="stFileUploader"] { background:#fff !important; border:1px dashed #a9b5cd !important; border-radius:14px; padding:.3rem; }
[data-testid="stFileUploader"] * { color:#26314e !important; }
.stButton button, [data-testid="stDownloadButton"] button { border-radius:11px !important; font-weight:700 !important; min-height:2.55rem; }
.stButton button[kind="primary"], [data-testid="stDownloadButton"] button[kind="primary"] { background:#594bd6 !important; color:white !important; border:1px solid #594bd6 !important; }
.stButton button[kind="primary"] * { color:white !important; }
[data-testid="stVerticalBlockBorderWrapper"] { background:#fff !important; border-color:#e1e7f1 !important; border-radius:16px !important; }
[data-testid="stAlert"] { border-radius:12px; }
.result { padding:1rem 1.15rem; border:1px solid #d9def8; border-left:4px solid #6559dc; background:#fbfbff; border-radius:12px; color:#202b47; line-height:1.65; white-space:pre-wrap; }
.small-note { font-size:.86rem; color:#65728a !important; }
hr { border-color:#e1e7f1 !important; }
@media(max-width:720px) { .hero { padding:1.3rem; border-radius:18px; } .hero h1 { font-size:1.9rem; } }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


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
        st.markdown("<div class='hero'><div class='kicker'>Smriti AI · Cloud setup</div><h1>Finish secure setup</h1><p>Add an app password in your hosting platform's Secrets settings before using this cloud demo.</p></div>", unsafe_allow_html=True)
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

with st.container():
    st.markdown("<div class='hero'><div class='kicker'>SMRITI AI · CLOUD EDITION</div><h1>VisionQuest</h1><p>A vision-first assistant for describing what a camera or image shows, with manually curated memory notes. The AI reports visual observations; it does not identify people or diagnose conditions.</p><span class='pill'>☁️ Hosted AI · HTTPS camera input · Available across devices</span></div>", unsafe_allow_html=True)

status = model_status()
people = list_people()
memories = list_memories(limit=500)
m1, m2, m3 = st.columns(3)
m1.metric("Saved people", len(people))
m2.metric("Memory notes", len(memories))
m3.metric("Vision model", "Ready to load" if status[0] else "Setup issue")
st.caption("Online demo: image processing and model inference happen on the hosting server. Uploaded/captured images are not written to disk by this app, but the hosting service processes the bytes. Use only fictional demo data until security and storage are production-ready.")

vision_tab, people_tab, memory_tab, voice_tab, search_tab, settings_tab = st.tabs(["📷 Vision", "👤 People", "🧠 Memories", "🔊 Voice", "🔎 Search", "⚙️ About & status"])

with vision_tab:
    left, right = st.columns([1.05, .95], gap="large")
    with left:
        st.subheader("Give VisionQuest an image")
        source_tab, upload_tab = st.tabs(["📸 Use this device's camera", "📁 Upload image"])
        with source_tab:
            camera_image = st.camera_input("Capture a photo", key="cloud_camera_input", help="Your browser requests camera permission. Works on the deployed HTTPS site on phones, tablets, and computers.")
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
        analyze = st.button("✨ Analyze image", type="primary", use_container_width=True, disabled=not bool(image_bytes) or (mode == "visual_question" and not question.strip()))
    with right:
        st.subheader("Visual observation")
        st.markdown("<div class='small-note'>Task-specific prompts ask the model to stay grounded in the image, mention uncertainty, and avoid guessing identities.</div>", unsafe_allow_html=True)
        if analyze and image_bytes:
            try:
                with st.spinner("Loading the cloud vision model and analyzing the image. The first request may take longer…"):
                    result = analyze_image(image_bytes, mode=mode, focus=question, max_side=max_side)
                st.session_state["latest_result"] = {"text": result.text, "mode": mode, "model": result.model, "elapsed": result.elapsed_seconds, "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"), "source_hash": hashlib.sha256(image_bytes).hexdigest()}
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
    st.write("**Camera:** Browser-based still capture; server-side OpenCV webcam access is intentionally disabled in cloud mode.")
    st.write("**Voice:** Browser-native speech synthesis reads observations and manually selected memory notes aloud; no external TTS API key is required.")
    st.write("**Memory storage:** Shared SQLite demo database on the host. Its filesystem may reset on redeploy/restart; backups and durable production database setup are not included.")
    st.write("**Privacy:** Image bytes are processed on the hosting server and are not intentionally saved as image files. Memory notes are stored on the host. The app uses one shared password, not individual user accounts.")
    st.write("**Not production-ready for care:** no individual accounts, role permissions, medical validation, guaranteed persistent storage, or automatic face identification.")
    if st.button("Lock this session", key="logout_button"):
        st.session_state["authenticated"] = False
        st.rerun()

st.divider()
st.caption("Smriti AI / VisionQuest Cloud Demo · Not a medical device · Use fictional demonstration data only")
