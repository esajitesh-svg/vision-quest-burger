"""Regression checks for live JARVIS input and visible fallback diagnostics."""
from pathlib import Path


APP_SOURCE = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")
VOICE_SOURCE = (
    Path(__file__).resolve().parents[1]
    / "components"
    / "browser_voice"
    / "index.html"
).read_text(encoding="utf-8")


def test_live_voice_questions_do_not_require_camera_to_be_running():
    component_start = APP_SOURCE.index("voice_event = browser_voice_component(")
    component_section = APP_SOURCE[component_start:component_start + 900]

    # General voice Q&A should work without a live frame; the camera is optional
    # context, rather than a prerequisite for browser recognition or AI replies.
    assert "enabled=bool(live_voice_enabled and status[0])" in component_section
    assert "enabled=bool(is_playing and live_voice_enabled" not in component_section
    assert "camera is optional for general questions" in APP_SOURCE
    assert "Start the camera only when you want answers about what is currently visible." in APP_SOURCE


def test_live_jarvis_has_typed_question_fallback_through_same_answer_worker():
    assert 'with st.form("live_jarvis_manual_question_form", clear_on_submit=True)' in APP_SOURCE
    assert '"SEND QUESTION TO SMRITI"' in APP_SOURCE
    assert "target=run_live_voice_text_worker" in APP_SOURCE
    assert 'voice_snapshot = live_voice_state.snapshot()' in APP_SOURCE
    assert 'if voice_snapshot["error"]' in APP_SOURCE


def test_browser_voice_recognition_errors_are_reported_in_app():
    assert "function sendStatus(code, message)" in VOICE_SOURCE
    assert 'sendStatus("listening", message)' in VOICE_SOURCE
    assert 'sendStatus("recognition_error", message)' in VOICE_SOURCE
    assert 'live_jarvis_browser_status_code' in APP_SOURCE
    assert "Browser microphone status:" in APP_SOURCE
