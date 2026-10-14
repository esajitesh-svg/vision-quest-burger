"""Regression checks for live JARVIS voice input independent of the camera."""
from pathlib import Path


def test_live_voice_questions_do_not_require_camera_to_be_running():
    source = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")
    component_start = source.index("voice_event = browser_voice_component(")
    component_section = source[component_start:component_start + 900]

    # General voice Q&A should work without a live frame; the camera is optional
    # context, rather than a prerequisite for browser recognition or AI replies.
    assert "enabled=bool(live_voice_enabled and status[0])" in component_section
    assert "enabled=bool(is_playing and live_voice_enabled" not in component_section
    assert "camera is optional for general questions" in source
    assert "Start the camera only when you want answers about what is currently visible." in source
