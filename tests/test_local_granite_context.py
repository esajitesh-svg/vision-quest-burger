"""Regression tests for Ollama context sizing and prompt bounds."""
from __future__ import annotations

from core import local_granite


class FakeResponse:
    status_code = 200
    ok = True
    text = ""

    def raise_for_status(self):
        return None

    def json(self):
        return {"message": {"content": "Hello from Granite"}}


def test_context_size_defaults_to_four_thousand_and_ninety_six(monkeypatch):
    monkeypatch.delenv("SMRITI_OLLAMA_CONTEXT", raising=False)
    assert local_granite._context_size() == 4096


def test_context_size_is_bounded_and_bad_values_fall_back(monkeypatch):
    monkeypatch.setenv("SMRITI_OLLAMA_CONTEXT", "6144")
    assert local_granite._context_size() == 6144
    monkeypatch.setenv("SMRITI_OLLAMA_CONTEXT", "2048")
    assert local_granite._context_size() == 4096
    monkeypatch.setenv("SMRITI_OLLAMA_CONTEXT", "99999")
    assert local_granite._context_size() == 8192
    monkeypatch.setenv("SMRITI_OLLAMA_CONTEXT", "not-a-number")
    assert local_granite._context_size() == 4096


def test_chat_sends_the_configured_context_window(monkeypatch):
    captured = {}

    def fake_post(url, *, json, timeout):
        captured["url"] = url
        captured["payload"] = json
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(local_granite.requests, "post", fake_post)
    monkeypatch.delenv("SMRITI_OLLAMA_CONTEXT", raising=False)

    result = local_granite._chat("Hi", system="You are Smriti.")

    assert result.text == "Hello from Granite"
    assert captured["payload"]["options"]["num_ctx"] == 4096
    assert captured["payload"]["options"]["num_predict"] == 180


def test_question_and_saved_context_are_bounded(monkeypatch):
    captured = {}

    def fake_chat(prompt, **kwargs):
        captured["prompt"] = prompt
        return local_granite.GraniteResult("answer", local_granite.GRANITE_MODEL, 0.01)

    monkeypatch.setattr(local_granite, "_chat", fake_chat)
    question = "Q" * 3000
    observation = "O" * 3000
    memories = [{"note": f"MEMORY{i}:" + "M" * 1000} for i in range(7)]
    history = [
        {"question": f"HQ{i}:" + "q" * 500, "reply": f"HA{i}:" + "a" * 500}
        for i in range(6)
    ]

    result = local_granite.ask_granite(
        question,
        latest_observation={"text": observation},
        memories=memories,
        conversation_history=history,
    )

    prompt = captured["prompt"]
    assert result == "answer"
    assert "O" * 701 not in prompt
    assert "Q" * 1201 not in prompt
    assert "MEMORY0:" in prompt and "MEMORY1:" in prompt and "MEMORY2:" in prompt
    assert "MEMORY3:" not in prompt and "MEMORY6:" not in prompt
    assert "HQ3:" not in prompt
    assert "HQ4:" in prompt and "HQ5:" in prompt


def test_auto_observer_uses_small_image_and_short_output(monkeypatch):
    from io import BytesIO
    from PIL import Image

    source = BytesIO()
    Image.new("RGB", (1280, 720), color=(40, 80, 120)).save(source, format="PNG")
    captured = {}

    def fake_chat(prompt, *, system, image_bytes=None, num_predict=180):
        captured.update(
            prompt=prompt,
            system=system,
            image_bytes=image_bytes,
            num_predict=num_predict,
        )
        return local_granite.GraniteResult("A person is near a desk.", local_granite.GRANITE_MODEL, 0.01)

    monkeypatch.setattr(local_granite, "_chat", fake_chat)
    result = local_granite.analyze_with_granite(
        source.getvalue(), mode="auto_observe", max_side=384
    )

    assert result.text == "A person is near a desk."
    assert captured["num_predict"] == 96
    assert "1–2 short sentences" in captured["prompt"]
    with Image.open(BytesIO(captured["image_bytes"])) as image:
        assert image.format == "JPEG"
        assert image.size == (384, 216)
