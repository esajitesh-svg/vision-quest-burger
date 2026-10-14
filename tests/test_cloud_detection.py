"""Tests for hosted object detection without making a network request."""
from __future__ import annotations

from io import BytesIO

from PIL import Image
import pytest

import core.cloud_detection as detection


def _jpeg_bytes() -> bytes:
    image = Image.new("RGB", (120, 80), color=(35, 105, 155))
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


class FakeResponse:
    def __init__(self, status_code: int, body):
        self.status_code = status_code
        self._body = body
        self.ok = 200 <= status_code < 400
        self.text = str(body)

    def json(self):
        return self._body


def test_object_detection_returns_boxes_labels_and_confidence(monkeypatch):
    calls = {}

    def fake_post(url, headers, data, timeout):
        calls.update(url=url, headers=headers, data=data, timeout=timeout)
        return FakeResponse(200, [
            {"label": "cup", "score": 0.93, "box": {"xmin": 10, "ymin": 12, "xmax": 80, "ymax": 65}},
            {"label": "book", "score": 0.22, "box": {"xmin": 1, "ymin": 1, "xmax": 30, "ymax": 25}},
            {"label": "bottle", "score": 0.71, "box": {"xmin": -10, "ymin": 15, "xmax": 200, "ymax": 200}},
        ])

    monkeypatch.setattr(detection, "get_hf_token", lambda: "hf-test-token")
    monkeypatch.setattr(detection.requests, "post", fake_post)

    result = detection.detect_objects(_jpeg_bytes(), score_threshold=0.4)

    assert result.image_size == (120, 80)
    assert [item["label"] for item in result.detections] == ["cup", "bottle"]
    assert result.detections[0]["score"] == 0.93
    assert result.detections[1]["box"] == {"xmin": 0, "ymin": 15, "xmax": 119, "ymax": 79}
    assert calls["url"] == detection.OBJECT_DETECTION_URL
    assert calls["headers"]["Authorization"] == "Bearer hf-test-token"
    assert calls["data"] == _jpeg_bytes()
    assert calls["timeout"] == detection.REQUEST_TIMEOUT


def test_object_detection_requires_token(monkeypatch):
    called = False

    def fake_post(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("network should not be used without token")

    monkeypatch.setattr(detection, "get_hf_token", lambda: "")
    monkeypatch.setattr(detection.requests, "post", fake_post)

    with pytest.raises(RuntimeError, match="HF_TOKEN"):
        detection.detect_objects(_jpeg_bytes())

    assert not called


@pytest.mark.parametrize(
    ("status_code", "message"),
    [
        (401, "rejected the detection request"),
        (403, "rejected the detection request"),
        (402, "credit, billing, or rate-limit"),
        (429, "credit, billing, or rate-limit"),
        (503, "loading or temporarily unavailable"),
    ],
)
def test_object_detection_maps_provider_errors(monkeypatch, status_code, message):
    monkeypatch.setattr(detection, "get_hf_token", lambda: "hf-test-token")
    monkeypatch.setattr(
        detection.requests,
        "post",
        lambda *args, **kwargs: FakeResponse(status_code, {"error": "provider unavailable"}),
    )
    with pytest.raises(RuntimeError, match=message):
        detection.detect_objects(_jpeg_bytes())


def test_object_detection_skips_invalid_boxes(monkeypatch):
    monkeypatch.setattr(detection, "get_hf_token", lambda: "hf-test-token")
    monkeypatch.setattr(
        detection.requests,
        "post",
        lambda *args, **kwargs: FakeResponse(
            200,
            [{"label": "unknown", "score": 0.9, "box": {"xmin": "NaN", "ymin": 1, "xmax": 5, "ymax": 5}}],
        ),
    )
    result = detection.detect_objects(_jpeg_bytes())
    assert result.detections == []
