"""Tests for opt-in hosted voice input and grounded replies."""
from __future__ import annotations

import core.cloud_voice as voice


class FakeResponse:
    def __init__(self, status_code, body):
        self.status_code = status_code
        self._body = body
        self.ok = 200 <= status_code < 400
        self.text = str(body)

    def json(self):
        return self._body


def test_transcribe_audio_posts_raw_wav_and_returns_text(monkeypatch):
    calls = {}

    def fake_post(url, headers, data, timeout):
        calls.update(url=url, headers=headers, data=data, timeout=timeout)
        return FakeResponse(200, {"text": "What did I see a moment ago?"})

    monkeypatch.setattr(voice, "get_hf_token", lambda: "hf-test-token")
    monkeypatch.setattr(voice.requests, "post", fake_post)
    result = voice.transcribe_audio(b"fake-wav", "audio/wav")

    assert result == "What did I see a moment ago?"
    assert calls["url"] == voice.ASR_URL
    assert calls["headers"]["Authorization"] == "Bearer hf-test-token"
    assert calls["headers"]["Content-Type"] == "audio/wav"
    assert calls["data"] == b"fake-wav"
    assert calls["timeout"] == (10, 60)


def test_transcribe_audio_requires_token(monkeypatch):
    monkeypatch.setattr(voice, "get_hf_token", lambda: "")
    try:
        voice.transcribe_audio(b"fake-wav")
    except RuntimeError as exc:
        assert "HF_TOKEN" in str(exc)
    else:
        raise AssertionError("Expected missing-token error")


def test_voice_reply_uses_scene_and_memory_context(monkeypatch):
    calls = {}

    def fake_post(url, headers, json, timeout):
        calls.update(url=url, headers=headers, payload=json, timeout=timeout)
        return FakeResponse(200, {
            "choices": [{"message": {"content": "The latest note says the keys are on the desk."}}]
        })

    monkeypatch.setattr(voice, "get_hf_token", lambda: "hf-test-token")
    monkeypatch.setattr(voice.requests, "post", fake_post)
    result = voice.answer_voice_query(
        "Where are the keys?",
        latest_observation={"text": "A set of keys is visible on a desk.", "timestamp": "now"},
        memories=[{"person_name": "Demo", "relationship": "test profile", "note": "Keys usually stay on the desk."}],
    )

    assert result == "The latest note says the keys are on the desk."
    assert calls["url"] == voice.CHAT_URL
    assert calls["payload"]["max_tokens"] == 180
    user_content = calls["payload"]["messages"][1]["content"]
    assert "A set of keys is visible on a desk." in user_content
    assert "Keys usually stay on the desk." in user_content


def test_voice_reply_rejects_empty_question():
    try:
        voice.answer_voice_query("   ")
    except ValueError as exc:
        assert "empty" in str(exc)
    else:
        raise AssertionError("Expected empty-question error")
