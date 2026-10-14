"""Sanity checks for the no-build browser-native voice component."""
from __future__ import annotations

from pathlib import Path


def test_browser_voice_component_has_recognition_speech_and_permission_ui():
    source = (
        Path(__file__).resolve().parents[1]
        / "components"
        / "browser_voice"
        / "index.html"
    ).read_text(encoding="utf-8")

    assert "window.SpeechRecognition || window.webkitSpeechRecognition" in source
    assert "window.speechSynthesis" in source
    assert "streamlit:componentReady" in source
    assert "streamlit:setComponentValue" in source
    assert "ACTIVATE LISTENING" in source
    assert 'id="stop"' in source


def test_browser_voice_does_not_start_listening_without_user_action():
    source = (
        Path(__file__).resolve().parents[1]
        / "components"
        / "browser_voice"
        / "index.html"
    ).read_text(encoding="utf-8")
    assert "startButton.addEventListener(\"click\", startListening)" in source
    assert "desiredListening = true" in source
def test_browser_voice_has_manual_test_and_replay_for_silent_autoplay():
    source = (
        Path(__file__).resolve().parents[1]
        / "components"
        / "browser_voice"
        / "index.html"
    ).read_text(encoding="utf-8")
    assert 'id="test"' in source
    assert 'id="replay"' in source
    assert "testButton.addEventListener(\"click\"" in source
    assert "replayButton.addEventListener(\"click\"" in source
    assert "Text-to-speech error:" in source

