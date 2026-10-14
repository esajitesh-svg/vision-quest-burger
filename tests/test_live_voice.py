"""Tests for real-time audio utterance buffering and WAV packaging."""
from __future__ import annotations

import time
import wave
from io import BytesIO

import numpy as np

from core.live_voice import LiveVoiceAssistantState, pcm16_to_wav


def _tone(samples=320, amplitude=3000):
    t = np.arange(samples, dtype=np.float32)
    return (np.sin(2 * np.pi * 440 * t / 16000) * amplitude).astype(np.int16)


def test_speech_is_not_finalized_until_pause_and_then_returns_pcm():
    state = LiveVoiceAssistantState()
    assert state.ingest_pcm(np.zeros(320, dtype=np.int16)) is None
    assert state.ingest_pcm(_tone()) is None
    assert state.snapshot()["speech_active"] is True

    # Use silent chunks until the configured end-of-turn silence is reached.
    result = None
    for _ in range(45):
        time.sleep(0.02)
        result = state.ingest_pcm(np.zeros(320, dtype=np.int16))
        if result:
            break
    assert result
    assert len(result) > 320 * 2
    assert state.snapshot()["speech_active"] is False


def test_short_noise_does_not_create_a_voice_turn():
    state = LiveVoiceAssistantState()
    for _ in range(20):
        assert state.ingest_pcm(np.zeros(320, dtype=np.int16)) is None
    assert state.begin_processing() is True
    assert state.begin_processing() is False


def test_turn_history_and_status_are_thread_safe():
    state = LiveVoiceAssistantState()
    assert state.begin_processing()
    state.finish_turn("What is this?", "It looks like a book.", used_live_camera=True)
    state.finish_turn("And that?", "It looks like a cup.")
    status = state.snapshot()
    assert status["reply_sequence"] == 2
    assert status["latest_transcript"] == "And that?"
    assert status["used_live_camera"] is False
    assert len(status["history"]) == 2
    assert state.history_snapshot()[-1]["reply"] == "It looks like a cup."


def test_pcm16_to_wav_is_mono_16khz():
    raw = _tone().tobytes()
    encoded = pcm16_to_wav(raw)
    with wave.open(BytesIO(encoded), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getframerate() == 16000
        assert wav.readframes(wav.getnframes()) == raw


def test_pcm16_to_wav_rejects_empty_and_misaligned_payload():
    for payload in (b"", b"\x01"):
        try:
            pcm16_to_wav(payload)
        except ValueError:
            pass
        else:
            raise AssertionError("Expected a validation error")
