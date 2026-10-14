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
