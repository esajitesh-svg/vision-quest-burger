"""Tests for cloud-side OpenCV live frame processing and temporary recording."""
from __future__ import annotations

import os
import tempfile

import cv2
import numpy as np
import pytest

from core.cloud_live import LiveMonitorState, process_live_frame


def _test_frame(width: int = 640, height: int = 360) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.rectangle(frame, (30, 30), (width - 30, height - 30), (80, 160, 230), 4)
    cv2.putText(frame, "VISIONQUEST", (45, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
    return frame


def test_live_frame_processing_resizes_and_returns_jpeg():
    frame, encoded, metrics = process_live_frame(_test_frame(1280, 720), max_side=640)
    assert frame.shape[1] <= 640
    assert frame.shape[0] <= 640
    assert encoded[:2] == bytes([255, 216])
    assert metrics["width"] == frame.shape[1]
    assert metrics["height"] == frame.shape[0]
    assert "brightness" in metrics
    assert "sharpness" in metrics


def test_live_frame_rejects_empty_input():
    with pytest.raises(ValueError):
        process_live_frame(np.array([], dtype=np.uint8))


def test_recording_creates_downloadable_silent_mp4():
    state = LiveMonitorState()
    frame, encoded, metrics = process_live_frame(_test_frame())
    state.update_frame(frame, encoded, metrics)
    state.start_recording(seconds=15)

    # Push enough frames to create a small, decodable video.
    for _ in range(20):
        state.update_frame(frame, encoded, metrics)
    state.stop_recording()

    video_bytes, filename = state.get_completed_video()
    assert video_bytes
    assert b"ftyp" in video_bytes[:64]
    assert filename.endswith(".mp4")
    assert state.get_status()["recording"] is False
    state.clear_completed_video()
    assert state.get_completed_video()[0] is None


def test_recording_requires_a_frame():
    state = LiveMonitorState()
    with pytest.raises(RuntimeError, match="camera frame"):
        state.start_recording(15)


def test_live_analysis_reserves_one_frame_at_a_time():
    state = LiveMonitorState()
    frame, encoded, metrics = process_live_frame(_test_frame())
    state.update_frame(frame, encoded, metrics)

    first = state.take_analysis_sample(15)
    assert first == encoded
    assert state.take_analysis_sample(15) is None

    state.finish_analysis(observation={"text": "A test observation"})
    status = state.get_status()
    assert status["analysis_busy"] is False
    assert status["latest_observation"]["text"] == "A test observation"

def test_manual_analysis_can_bypass_periodic_interval():
    state = LiveMonitorState()
    frame, encoded, metrics = process_live_frame(_test_frame())
    state.update_frame(frame, encoded, metrics)

    first = state.take_analysis_sample(60)
    assert first == encoded
    state.finish_analysis(observation={"text": "first result"})

    # Normal periodic sampling should wait, but a manual analyze button can force a sample.
    assert state.take_analysis_sample(60) is None
    forced = state.take_analysis_sample(60, force=True)
    assert forced == encoded
    # It must still avoid overlapping inference attempts.
    assert state.take_analysis_sample(60, force=True) is None
    state.finish_analysis(observation={"text": "manual result"})

