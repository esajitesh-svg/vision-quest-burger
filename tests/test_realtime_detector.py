"""Tests for genuine per-frame local YOLOX detection using a fake OpenCV DNN net."""
from __future__ import annotations

import numpy as np
import pytest

from core.realtime_detector import (
    COCO_CLASSES,
    MODEL_INPUT_SIZE,
    REALTIME_DETECTOR_LABEL,
    RealtimeYOLOXDetector,
)


class FakeNet:
    def __init__(self, output):
        self.output = output
        self.last_input = None
        self.forward_calls = 0

    def setPreferableBackend(self, backend):
        self.backend = backend

    def setPreferableTarget(self, target):
        self.target = target

    def setInput(self, value):
        self.last_input = value

    def getUnconnectedOutLayersNames(self):
        return ["output"]

    def forward(self, names):
        self.forward_calls += 1
        return [self.output]


def _mock_output() -> np.ndarray:
    # YOLOX-S output includes 80x80 + 40x40 + 20x20 grid cells, each with 85 values.
    output = np.zeros((1, 8400, 85), dtype=np.float32)
    # Stride-8 grid: cell x=10, y=5; decoded box should be (64,24)-(96,56).
    index = 5 * 80 + 10
    output[0, index, 0:4] = [0.0, 0.0, np.log(4.0), np.log(4.0)]
    output[0, index, 4] = 0.95
    output[0, index, 5] = 0.90  # COCO class 0: person
    return output


def test_detector_runs_real_forward_pass_for_each_call_and_returns_current_frame_box():
    net = FakeNet(_mock_output())
    detector = RealtimeYOLOXDetector(net=net)
    frame = np.zeros((360, 640, 3), dtype=np.uint8)

    first = detector.detect(frame, confidence_threshold=0.35)
    second = detector.detect(frame, confidence_threshold=0.35)

    assert net.forward_calls == 2
    assert net.last_input.shape == (1, 3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)
    assert first.image_size == (640, 360)
    assert first.fps > 0 and first.elapsed_seconds > 0
    assert first.detections == second.detections
    assert len(first.detections) == 1
    assert first.detections[0]["label"] == "person"
    assert first.detections[0]["score"] == pytest.approx(0.855, abs=0.002)
    assert first.detections[0]["box"] == {"xmin": 64, "ymin": 24, "xmax": 96, "ymax": 56}
    assert len(COCO_CLASSES) == 80
    assert "OpenCV DNN auto OpenCL/CPU" in REALTIME_DETECTOR_LABEL


def test_detector_threshold_filters_low_confidence_objects():
    net = FakeNet(_mock_output())
    detector = RealtimeYOLOXDetector(net=net)
    frame = np.zeros((360, 640, 3), dtype=np.uint8)

    result = detector.detect(frame, confidence_threshold=0.90)

    assert result.detections == []


@pytest.mark.parametrize("frame", [
    np.array([], dtype=np.uint8),
    np.zeros((50, 50), dtype=np.uint8),
    np.zeros((50, 50, 4), dtype=np.uint8),
])
def test_detector_rejects_invalid_frames(frame):
    detector = RealtimeYOLOXDetector(net=FakeNet(_mock_output()))
    with pytest.raises(ValueError):
        detector.detect(frame)


def test_detector_rejects_invalid_confidence_threshold():
    detector = RealtimeYOLOXDetector(net=FakeNet(_mock_output()))
    with pytest.raises(ValueError, match="between 5% and 95%"):
        detector.detect(np.zeros((100, 100, 3), dtype=np.uint8), confidence_threshold=1.0)
