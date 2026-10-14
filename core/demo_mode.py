"""Deterministic, offline-only responses for the Smriti demonstration mode.

All returned text is explicitly sample content; this module never calls a network
service and must not be treated as analysis of a real camera frame.
"""
from __future__ import annotations


def demo_observation(camera_available: bool = False, detected_faces: int | None = None) -> str:
    """Return a clearly labelled sample observation without invoking an AI model."""
    if camera_available:
        face_note = (
            f" The local face detector found {detected_faces} face region(s)."
            if detected_faces is not None
            else ""
        )
        return (
            "DEMO SAMPLE — A person is seated near a desk with a notebook and a cup. "
            "This scene description is fictional sample content, not an analysis of the camera frame."
            + face_note
        )
    return (
        "DEMO SAMPLE — A person is seated near a desk with a notebook and a cup. "
        "No camera frame was received; this is fictional sample content."
    )


def demo_reply(question: str) -> str:
    """Answer common demo prompts immediately, without cloud or local model inference."""
    value = (question or "").strip()
    q = value.casefold()
    if not value:
        return "Demo mode is ready. Ask me a short question."

    if any(term in q for term in ("remember", "memory", "earlier", "last time", "where did")):
        return (
            "DEMO SAMPLE MEMORY: In this fictional example, you met Alex at the community garden "
            "and talked about growing herbs. This is not a real saved memory."
        )

    if any(term in q for term in ("who is", "who's", "recognize", "recognise", "identify", "face")):
        return (
            "Open the Face Recognition panel to register a person with their permission and test "
            "matching against their reference photo. This reply itself does not identify anyone."
        )

    if any(term in q for term in ("see", "camera", "look", "scene", "observe", "in front of me")):
        return (
            "I can show a sample visual observation in Demo Mode. It is fictional demonstration text, "
            "not a description of your live camera. Use Live Mode for model-generated observations."
        )

    if any(term in q for term in ("hello", "hi ", "hey", "good morning", "good evening")):
        return "Hello! Smriti Demo Mode is ready to demonstrate memory, voice, and visual-companion workflows."

    if any(term in q for term in ("voice", "speak", "talk", "hear")):
        return (
            "Voice playback is available through the browser speech controls. This is a demo response; "
            "press Test Voice to verify your speakers."
        )

    short_question = value[:180]
    return (
        f"DEMO RESPONSE: I received “{short_question}”. This is a scripted sample answer, "
        "not a live AI-generated response. Switch to Live Mode for actual model replies."
    )



def render_demo_hub() -> None:
    """Render a complete, offline-friendly demo workspace with local face matching."""
    import html
    import cv2
    import streamlit as st

    from core.face_recognition import (
        decode_image_bytes,
        detect_faces,
        encode_jpeg,
        face_recognizer_available,
        recognize_faces,
    )
    from core.cloud_db import (
        init_db,
        list_people,
        save_person,
        save_person_portrait,
        get_person_portrait,
        delete_person_portrait,
    )
    from core.local_archive import archive_event, save_media
    from pathlib import Path
    import base64


    def archive_tags():
        return {
            "people": st.session_state.get("archive_people_tags", ""),
            "place": st.session_state.get("archive_place_tag", ""),
            "activity": st.session_state.get("archive_activity_tag", ""),
        }

    def archive_demo_event(event_type, payload):
        if not st.session_state.get("archive_auto_save", True):
            return
        try:
            archive_event(event_type, payload, archive_tags())
        except Exception as archive_exc:
            st.warning(f"Local demo archive failed: {type(archive_exc).__name__}: {archive_exc}")

    st.markdown(
        """
        <div style="padding:1.4rem 1.6rem;border:1px solid #24717c;border-left:4px solid #4df3df;
        background:radial-gradient(circle at 90% 0%,#10303a,#07121a 65%);border-radius:12px">
          <div style="font:700 .72rem monospace;letter-spacing:.2em;color:#56f5e4">SMRITI // DEMONSTRATION MODE</div>
          <h1 style="margin:.4rem 0;color:#e4ffff">A complete companion demo</h1>
          <p style="color:#a8c8d2">Instant scripted responses, sample memory flows, browser voice playback, and local face matching.
          AI replies and sample scene descriptions are simulated; face matching only runs when you explicitly request it.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    tab_companion, tab_faces, tab_memory = st.tabs(
        ["✦ COMPANION DEMO", "◎ FACE RECOGNITION", "▤ MEMORY DEMO"]
    )

    with tab_companion:
        left, right = st.columns([1, 1.3], gap="large")
        with left:
            st.markdown("### SMRITI ONLINE")
            st.markdown(
                "<div style='height:135px;width:135px;margin:1rem auto;border:1px solid #5df8e7;"
                "border-radius:50%;background:radial-gradient(circle,#bafff5 0%,#18bcae 14%,#07313e 58%,#050b10 78%);"
                "box-shadow:0 0 38px #26dfd855'></div>",
                unsafe_allow_html=True,
            )
            st.caption("DEMO RESPONSE ENGINE · NO CLOUD AI REQUESTS")
            if st.button("Generate sample scene observation", type="primary", use_container_width=True):
                st.session_state["demo_observation"] = demo_observation(False)
                archive_demo_event("demo_sample_observation", {
                    "observation": st.session_state["demo_observation"],
                    "simulated": True,
                })
            st.info(st.session_state.get("demo_observation", "Press the button to display a fictional sample observation."))
            st.caption("This sample is not generated from your camera.")
        with right:
            st.markdown("### Ask Smriti")
            with st.form("demo_question_form", clear_on_submit=False):
                question = st.text_input(
                    "Question",
                    key="demo_question",
                    placeholder="Try: What do you remember?",
                )
                asked = st.form_submit_button("ASK SMRITI", type="primary", use_container_width=True)
            if asked:
                st.session_state["demo_reply"] = demo_reply(question)
                archive_demo_event("conversation_turn", {
                    "question": str(question or "")[:1200],
                    "answer": st.session_state["demo_reply"],
                    "mode": "DEMO MODE",
                    "simulated": True,
                })
            reply = st.session_state.get("demo_reply", "Hello! Demo Mode is ready. Ask a question to see a sample reply.")
            st.markdown(
                "<div style='padding:1rem;background:#091a22;border-left:3px solid #42e9d7;border-radius:6px'>"
                + html.escape(reply)
                + "</div>",
                unsafe_allow_html=True,
            )
            st.caption("SIMULATED TEXT RESPONSE")
            try:
                from core.voice import build_speech_html
                st.components.v1.html(
                    build_speech_html(reply, language="en-IN", rate=0.95, widget_label="▶ SPEAK DEMO REPLY"),
                    height=95,
                    scrolling=False,
                )
            except Exception:
                st.caption("Browser speech controls are unavailable in this deployment.")

            # Raw microphone audio recording is optional and requires a direct user click.
            try:
                voice_component = st.components.v1.declare_component(
                    "smriti_demo_audio_archive",
                    path=str(Path(__file__).resolve().parents[1] / "components" / "browser_voice"),
                )
                audio_event = voice_component(
                    enabled=False,
                    busy=False,
                    language="en-IN",
                    rate=0.95,
                    speech_text="",
                    speech_sequence=0,
                    speech_kind="reply",
                    last_transcript="",
                    voice_error="",
                    key="smriti_demo_audio_archive_widget",
                    default=None,
                )
                if isinstance(audio_event, dict) and audio_event.get("event") == "audio_archive":
                    event_id = str(audio_event.get("event_id", ""))
                    if event_id and event_id != st.session_state.get("demo_audio_archive_event_id"):
                        st.session_state["demo_audio_archive_event_id"] = event_id
                        encoded = str(audio_event.get("audio_base64", ""))
                        if not encoded or len(encoded) > 12_000_000:
                            st.error("Audio clip is empty or too large.")
                        else:
                            raw_audio = base64.b64decode(encoded, validate=True)
                            mime = str(audio_event.get("mime_type", "audio/webm")).split(";")[0].lower()
                            ext = "m4a" if mime == "audio/mp4" else ("wav" if mime == "audio/wav" else "webm")
                            saved_audio = save_media("audio", raw_audio, ext, mime, archive_tags(), {
                                "maximum_recording_seconds": 30,
                                "mode": "DEMO MODE",
                            })
                            st.success(f"Audio clip saved locally: {saved_audio}")
            except Exception as audio_exc:
                st.warning(f"Audio archive controls are unavailable: {type(audio_exc).__name__}: {audio_exc}")

    with tab_faces:
        st.markdown("### Local face recognition")
        st.warning(
            "Reference portraits are stored in Smriti's configured app database and are shared with Live Mode. "
            "Use only your own face or a person's photo with explicit permission. Demo AI replies and scene "
            "descriptions stay simulated; face matching runs only when you request it."
        )
        try:
            init_db()
            all_people = list_people()
            profiles = []
            for person in all_people:
                try:
                    if get_person_portrait(int(person["id"])):
                        profiles.append(person)
                except Exception:
                    continue
        except Exception as db_exc:
            profiles = []
            st.error(f"Face database is unavailable: {type(db_exc).__name__}: {db_exc}")

        if not face_recognizer_available():
            st.error(
                "Face matching is unavailable because this OpenCV build lacks the contrib recognizer. "
                "Install the OpenCV contrib package from requirements.txt and restart Smriti."
            )

        with st.expander("1 · Register or update a consented reference photo", expanded=not bool(profiles)):
            with st.form("demo_face_enroll_form", clear_on_submit=True):
                display_name = st.text_input("Name or display label", max_chars=100)
                relationship = st.text_input("Optional label", max_chars=100, value="Friend / family")
                reference = st.file_uploader(
                    "Reference photo (JPEG, PNG, WEBP; 3 MB max)",
                    type=["jpg", "jpeg", "png", "webp"],
                    key="demo_face_reference_upload",
                )
                consent = st.checkbox("I have permission to store and use this photo for face matching.")
                enroll = st.form_submit_button("REGISTER / UPDATE FACE", type="primary")
            if enroll:
                if not consent:
                    st.warning("Confirm permission before registering the photo.")
                elif not display_name.strip():
                    st.warning("Enter a name or label.")
                elif not relationship.strip():
                    st.warning("Enter a relationship or label.")
                elif reference is None:
                    st.warning("Choose a reference photo.")
                elif len(reference.getvalue()) > 3 * 1024 * 1024:
                    st.warning("Choose an image no larger than 3 MB.")
                else:
                    try:
                        reference_bytes = reference.getvalue()
                        ref_frame = decode_image_bytes(reference_bytes)
                        if not detect_faces(ref_frame):
                            st.error(
                                "No face was detected in that reference photo. Try a clear, front-facing "
                                "photo with good lighting."
                            )
                        else:
                            current_people = list_people()
                            existing = next(
                                (
                                    p for p in current_people
                                    if str(p.get("name", "")).casefold() == display_name.strip().casefold()
                                ),
                                None,
                            )
                            if existing:
                                person_id = int(existing["id"])
                            else:
                                save_person(
                                    display_name.strip(),
                                    relationship.strip(),
                                    "Reference portrait registered with permission.",
                                )
                                person_id = int(next(
                                    p["id"] for p in list_people()
                                    if str(p.get("name", "")).casefold() == display_name.strip().casefold()
                                ))
                            # Normalize and bound stored image bytes before persisting the consented portrait.
                            save_person_portrait(person_id, encode_jpeg(ref_frame), "image/jpeg")
                            st.success(
                                f"Reference face saved for {display_name.strip()}. "
                                "This profile is now available in Live Mode too."
                            )
                            st.rerun()
                    except Exception as exc:
                        st.error(f"Could not register this photo: {type(exc).__name__}: {exc}")

        try:
            all_people = list_people()
            profiles = []
            for person in all_people:
                try:
                    if get_person_portrait(int(person["id"])):
                        profiles.append(person)
                except Exception:
                    continue
        except Exception as db_exc:
            profiles = []
            st.warning(f"Could not load saved reference portraits: {type(db_exc).__name__}: {db_exc}")

        if profiles:
            st.write("**Saved reference profiles shared with Live Mode**")
            for person in profiles:
                pcol, bcol = st.columns([4, 1])
                pcol.write(f"{person.get('name', 'Unnamed')} · {person.get('relationship', '')}")
                if bcol.button("Remove portrait", key=f"demo_face_remove_{person['id']}"):
                    delete_person_portrait(int(person["id"]))
                    st.session_state.pop("demo_face_result", None)
                    st.rerun()

            st.markdown("### 2 · Check a camera snapshot or uploaded test photo")
            source_capture, source_upload = st.tabs(["📸 Capture with camera", "📁 Upload test photo"])
            with source_capture:
                test_capture = st.camera_input("Take a test photo", key="demo_face_capture")
            with source_upload:
                test_upload = st.file_uploader(
                    "Choose a test photo",
                    type=["jpg", "jpeg", "png", "webp"],
                    key="demo_face_test_upload",
                )
            test_file = test_capture if test_capture is not None else test_upload
            if st.button(
                "RECOGNIZE FACES",
                type="primary",
                disabled=test_file is None or not face_recognizer_available(),
                key="demo_recognize_shared_faces",
            ):
                try:
                    frame = decode_image_bytes(test_file.getvalue())
                    annotated, matches = recognize_faces(frame, profiles, get_person_portrait)
                    st.session_state["demo_face_result"] = {
                        "image": cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                        "faces": matches,
                    }
                except Exception as exc:
                    st.session_state.pop("demo_face_result", None)
                    st.error(f"Recognition failed: {type(exc).__name__}: {exc}")

            face_result = st.session_state.get("demo_face_result")
            if face_result:
                st.image(face_result["image"], caption="Local face boxes and experimental matching labels", use_container_width=True)
                if not face_result["faces"]:
                    st.info("No face was detected in the test photo.")
                for item in face_result["faces"]:
                    if item["recognized"]:
                        st.success(f"Possible match: {item['name']} · experimental distance {item['distance']}")
                    else:
                        st.info(f"Unknown person · no sufficiently close match (distance {item['distance']})")
                st.caption(
                    "This lightweight matcher is experimental and can confuse people under different lighting or poses. "
                    "Verify each match with the person. Do not use it for access control or safety-critical decisions."
                )
        else:
            st.info("No saved reference portraits yet. Register a consented photo above, then test a camera snapshot or uploaded image.")

    with tab_memory:
        st.markdown("### Sample memory retrieval")
        st.write(
            "The following is a scripted example demonstrating the intended workflow, not a real memory about you."
        )
        st.markdown(
            "<div style='padding:1rem;background:#091a22;border:1px solid #204653;border-radius:6px'>"
            "<b>DEMO SAMPLE MEMORY</b><br>In this fictional example, you met Alex at a community garden "
            "and talked about growing herbs.</div>",
            unsafe_allow_html=True,
        )
        query = st.text_input("Try a memory question", key="demo_memory_query", placeholder="Who did I meet at the garden?")
        if st.button("RECALL SAMPLE MEMORY", key="demo_recall_memory"):
            st.session_state["demo_memory_reply"] = demo_reply(query or "What do you remember?")
        if "demo_memory_reply" in st.session_state:
            st.write(st.session_state["demo_memory_reply"])
        st.caption("Demo memories are fictional and are not added to long-term storage.")

    st.divider()
    st.caption("DEMO MODE · AI responses are simulated · local face matching is experimental · switch to Live Mode for real AI inference.")
