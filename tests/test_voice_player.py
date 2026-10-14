"""Tests for the browser-native voice player and auto-speak mode."""
from __future__ import annotations

from core.voice import build_speech_html


def test_auto_speak_uses_existing_browser_speech_engine_and_language():
    html = build_speech_html(
        "Hello, I'm VisionQuest.",
        language="en-IN",
        rate=0.95,
        widget_label="Replay reply",
        autoplay=True,
    )

    assert "window.speechSynthesis" in html
    assert "new SpeechSynthesisUtterance(message)" in html
    assert 'const preferredLanguage = "en-IN"' in html
    assert 'value="0.95"' in html
    assert "if (true)" in html
    assert "window.setTimeout" in html
    assert "Replay reply" in html


def test_read_aloud_does_not_autoplay_unless_requested():
    html = build_speech_html("A saved memory note.", autoplay=False)

    assert "if (false)" in html
    assert "window.speechSynthesis" in html
    assert "Tap Read aloud" in html
