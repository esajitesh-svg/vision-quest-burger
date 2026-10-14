"""Tests for the combined live scene, object detection, and OpenCV report."""
from __future__ import annotations

from core.live_report import build_live_report


def test_live_report_combines_scene_detections_positions_and_frame_metrics():
    observation = {
        "text": "A cup is on a desk beside a book.",
        "mode": "auto_observe",
        "model": "test-vlm",
        "elapsed": 1.7,
    }
    detections = [
        {
            "label": "cup",
            "score": 0.93,
            "box": {"xmin": 10, "ymin": 10, "xmax": 80, "ymax": 80},
        },
        {
            "label": "book",
            "score": 0.86,
            "box": {"xmin": 350, "ymin": 130, "xmax": 510, "ymax": 310},
        },
    ]
    report = build_live_report(
        observation,
        {
            "detections": detections,
            "age_seconds": 1.5,
            "busy": False,
            "error": "",
            "elapsed_seconds": 0.8,
            "model": "test-detector",
        },
        {
            "width": 640,
            "height": 360,
            "brightness": 118.4,
            "brightness_label": "normal",
            "sharpness": 72.0,
            "sharpness_label": "reasonable edge detail",
        },
        frame_count=123,
    )

    assert "SCENE ANALYSIS" in report
    assert "A cup is on a desk beside a book." in report
    assert "Detected 2 object instance(s) across 2 class(es)" in report
    assert "cup — 93% confidence" in report
    assert "approximate position: upper left" in report
    assert "box (10, 10) to (80, 80)" in report
    assert "book — 86% confidence" in report
    assert "OPENCV FRAME CHECK" in report
    assert "Average brightness: 118.4/255" in report
    assert "Camera frames received in this session: 123" in report
    assert "predictions, not certainty" in report


def test_live_report_explains_empty_detection_result_without_claiming_scene_is_empty():
    report = build_live_report(
        None,
        {
            "detections": [],
            "age_seconds": 2.0,
            "busy": False,
            "error": "",
            "model": "test-detector",
        },
        {"width": 640, "height": 360},
    )
    assert "The scene explanation has not returned a result yet." in report
    assert "No objects were returned above the selected confidence threshold." in report
    assert "does not prove the scene is empty" in report


def test_live_report_warns_when_object_scan_is_stale():
    report = build_live_report(
        None,
        {
            "detections": [{"label": "cup", "score": 0.9, "box": {"xmin": 10, "ymin": 10, "xmax": 20, "ymax": 25}}],
            "age_seconds": 15.0,
            "busy": False,
            "error": "",
            "model": "test-detector",
        },
        {"width": 100, "height": 100},
    )
    assert "object scan is stale" in report
    assert "wait for a fresh scan" in report


def test_live_report_handles_missing_results_and_metrics():
    report = build_live_report(None, None, None)
    assert "SCENE ANALYSIS" in report
    assert "No completed object scan is available yet." in report
    assert "LIMITATIONS" in report
