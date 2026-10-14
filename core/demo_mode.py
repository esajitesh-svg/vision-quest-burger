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
        face_recognizer_available,
        recognize_faces,
    )

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

    with tab_faces:
        st.markdown("### Local face recognition")
        st.warning(
            "Use only your own face or a person's photo with their permission. "
            "Photos stay in this browser session's app memory and are not saved to the persistent people database."
        )
        profiles = st.session_state.setdefault("demo_face_profiles", [])
        if not face_recognizer_available():
            st.error(
                "Face matching is unavailable because the current OpenCV build lacks the contrib recognizer. "
                "Install the OpenCV contrib headless package listed in the updated requirements, then restart the app."
            )

        with st.expander("1 · Register a consented reference photo", expanded=not bool(profiles)):
            with st.form("demo_face_enroll_form", clear_on_submit=True):
                display_name = st.text_input("Name or demo label", max_chars=80)
                relationship = st.text_input("Optional label", max_chars=80, value="Demo profile")
                reference = st.file_uploader(
                    "Reference photo (JPEG, PNG, WEBP; 3 MB max)",
                    type=["jpg", "jpeg", "png", "webp"],
                    key="demo_face_reference_upload",
                )
                consent = st.checkbox("I have permission to use this photo for this demo.")
                enroll = st.form_submit_button("REGISTER FACE", type="primary")
            if enroll:
                if not consent:
                    st.warning("Confirm permission before registering the photo.")
                elif not display_name.strip():
                    st.warning("Enter a name or label.")
                elif reference is None:
                    st.warning("Choose a reference photo.")
                elif any(p["name"].casefold() == display_name.strip().casefold() for p in profiles):
                    st.warning("That name is already registered in this demo session.")
                elif len(reference.getvalue()) > 3 * 1024 * 1024:
                    st.warning("Choose an image no larger than 3 MB.")
                else:
                    try:
                        ref_frame = decode_image_bytes(reference.getvalue())
                        found = detect_faces(ref_frame)
                        if not found:
                            st.error("No face was detected in that reference photo. Try a clear, front-facing photo with good lighting.")
                        else:
                            profiles.append(
                                {
                                    "id": max([int(p["id"]) for p in profiles], default=0) + 1,
                                    "name": display_name.strip()[:80],
                                    "relationship": relationship.strip()[:80] or "Demo profile",
                                    "image_data": reference.getvalue(),
                                }
                            )
                            st.session_state["demo_face_profiles"] = profiles
                            st.success("Reference face registered for this session.")
                            st.rerun()
                    except Exception as exc:
                        st.error(f"Could not register this photo: {type(exc).__name__}: {exc}")

        if profiles:
            st.write("**Registered reference profiles**")
            for p in profiles:
                pcol, bcol = st.columns([4, 1])
                pcol.write(f"{p['name']} · {p['relationship']}")
                if bcol.button("Remove", key=f"demo_face_remove_{p['id']}"):
                    st.session_state["demo_face_profiles"] = [x for x in profiles if x["id"] != p["id"]]
                    st.session_state.pop("demo_face_result", None)
                    st.rerun()

            st.markdown("### 2 · Check a test photo")
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
            if st.button("RECOGNIZE FACES", type="primary", disabled=test_file is None or not face_recognizer_available()):
                try:
                    frame = decode_image_bytes(test_file.getvalue())
                    annotated, result = recognize_faces(
                        frame,
                        profiles,
                        lambda profile_id: next(
                            ({"image_data": item["image_data"]} for item in profiles if int(item["id"]) == int(profile_id)),
                            None,
                        ),
                    )
                    st.session_state["demo_face_result"] = {
                        "image": cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                        "faces": result,
                    }
                except Exception as exc:
                    st.session_state.pop("demo_face_result", None)
                    st.error(f"Recognition failed: {type(exc).__name__}: {exc}")

            face_result = st.session_state.get("demo_face_result")
            if face_result:
                st.image(face_result["image"], caption="Local face boxes and matching labels", use_container_width=True)
                if not face_result["faces"]:
                    st.info("No face detected in the test photo.")
                for item in face_result["faces"]:
                    if item["recognized"]:
                        st.success(f"Recognized match: {item['name']} · experimental distance {item['distance']}")
                    else:
                        st.info(f"Unknown person · no sufficiently close match (distance {item['distance']})")
                st.caption(
                    "This lightweight matcher is experimental and can confuse people with similar appearance or different lighting. "
                    "Confirm identities with the person; do not use it for access control or safety-critical decisions."
                )
        else:
            st.info("Register a consented reference photo to enable matching against it.")

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
