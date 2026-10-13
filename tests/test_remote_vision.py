"""Tests for the remote VLM path without calling a real paid/hosted endpoint."""
from __future__ import annotations

from io import BytesIO

from PIL import Image
import pytest

import core.cloud_vision as vision


def _png_bytes() -> bytes:
    image = Image.new("RGB", (96, 64), color=(20, 110, 160))
    buf = BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


class FakeResponse:
    def __init__(self, status_code: int, body: dict):
        self.status_code = status_code
        self._body = body
        self.ok = 200 <= status_code < 400
        self.text = str(body)

    def json(self):
        return self._body


def test_remote_inference_posts_image_and_returns_text(monkeypatch):
    calls = {}

    def fake_post(url, headers, json, timeout):
        calls.update(url=url, headers=headers, payload=json, timeout=timeout)
        return FakeResponse(200, {
            "choices": [{"message": {"content": "A blue object sits on a table."}}]
        })

    monkeypatch.setattr(vision, "get_hf_token", lambda: "hf-test-token")
    monkeypatch.setattr(vision.requests, "post", fake_post)

    result = vision.analyze_image(_png_bytes(), mode="scene", max_side=64)

    assert result.text == "A blue object sits on a table."
    assert result.original_size == (96, 64)
    assert result.processed_size == (64, 43)
    assert calls["url"] == vision.INFERENCE_URL
    assert calls["headers"]["Authorization"] == "Bearer hf-test-token"
    content = calls["payload"]["messages"][0]["content"]
    assert content[0]["type"] == "text"
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert calls["timeout"] == vision.REQUEST_TIMEOUT


def test_remote_inference_refuses_to_call_without_token(monkeypatch):
    called = False

    def fake_post(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("network must not be used without a token")

    monkeypatch.setattr(vision, "get_hf_token", lambda: "")
    monkeypatch.setattr(vision.requests, "post", fake_post)

    with pytest.raises(RuntimeError, match="HF_TOKEN"):
        vision.analyze_image(_png_bytes())

    assert not called


@pytest.mark.parametrize(
    ("status_code", "message"),
    [
        (401, "rejected HF_TOKEN"),
        (403, "rejected HF_TOKEN"),
        (402, "billing/credit or rate-limit"),
        (429, "billing/credit or rate-limit"),
        (503, "loading or temporarily unavailable"),
    ],
)
def test_remote_inference_translates_provider_errors(monkeypatch, status_code, message):
    monkeypatch.setattr(vision, "get_hf_token", lambda: "hf-test-token")
    monkeypatch.setattr(
        vision.requests,
        "post",
        lambda *args, **kwargs: FakeResponse(status_code, {"error": "test"}),
    )

    with pytest.raises(RuntimeError, match=message):
        vision.analyze_image(_png_bytes(), max_side=64)
