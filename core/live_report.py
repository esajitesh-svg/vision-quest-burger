"""Combine live vision-language and object-detector outputs into a readable report."""
from __future__ import annotations

from collections import Counter
from typing import Any


def _position_for_box(box: dict[str, Any], width: int, height: int) -> str:
    """Return a coarse, explainable image location from detector box pixels."""
    try:
        center_x = (float(box["xmin"]) + float(box["xmax"])) / 2.0
        center_y = (float(box["ymin"]) + float(box["ymax"])) / 2.0
    except (KeyError, TypeError, ValueError, OverflowError):
        return "position unavailable"

    horizontal = (
        "left" if center_x < width / 3
        else "right" if center_x >= width * 2 / 3
        else "center"
    )
    vertical = (
        "upper" if center_y < height / 3
        else "lower" if center_y >= height * 2 / 3
        else "middle"
    )
    if vertical == "middle" and horizontal == "center":
        return "center of the frame"
    if vertical == "middle":
        return f"{horizontal} side"
    if horizontal == "center":
        return f"{vertical} center"
    return f"{vertical} {horizontal}"


def build_live_report(
    observation: dict[str, Any] | None,
    detection_status: dict[str, Any] | None,
    frame_metrics: dict[str, Any] | None = None,
    frame_count: int = 0,
) -> str:
    """Create a detailed textual/voice-friendly snapshot of all available live outputs."""
    lines = ["VISIONQUEST — LIVE OBSERVATION & ANALYSIS REPORT"]

    if observation and str(observation.get("text", "")).strip():
        lines.extend([
            "",
            "SCENE ANALYSIS",
            str(observation["text"]).strip(),
        ])
        if observation.get("mode") or observation.get("model"):
            lines.append(
                "Scene model: "
                + " · ".join(
                    str(value) for value in (observation.get("model"), observation.get("mode"))
                    if value
                )
            )
        if observation.get("elapsed") is not None:
            lines.append(f"Scene analysis time: {float(observation['elapsed']):.1f} seconds.")
        if observation.get("timestamp"):
            lines.append(f"Scene observation timestamp: {observation['timestamp']}.")
    else:
        lines.extend(["", "SCENE ANALYSIS", "The scene explanation has not returned a result yet."])

    status = detection_status or {}
    detections = status.get("detections") or []
    age = status.get("age_seconds")
    metrics = frame_metrics or {}
    try:
        width = max(1, int(metrics.get("width", 640)))
        height = max(1, int(metrics.get("height", 360)))
    except (TypeError, ValueError, OverflowError):
        width, height = 640, 360

    lines.extend(["", "OBJECT DETECTION"])
    if status.get("error"):
        lines.append(f"Detector message: {status['error']}")
    if age is None:
        lines.append("No completed object scan is available yet.")
    else:
        age_value = max(0.0, float(age))
        lines.append("A completed object scan is available.")
        scan_is_stale = age_value > 1.5
        if scan_is_stale:
            lines.append("The last object scan is stale; wait for a fresh scan before relying on those boxes.")
        detector_elapsed = status.get("elapsed_seconds")
        if isinstance(detector_elapsed, (int, float)) and detector_elapsed > 0:
            lines.append(f"Object-detection inference time: {detector_elapsed:.1f} seconds.")

        visible_detections = [] if scan_is_stale else detections
        if visible_detections:
            counts = Counter(str(item.get("label", "object")) for item in visible_detections)
            inventory = ", ".join(
                f"{label} × {count}" if count > 1 else label
                for label, count in sorted(counts.items(), key=lambda pair: pair[0].lower())
            )
            lines.append(f"Detected {len(visible_detections)} object instance(s) across {len(counts)} class(es): {inventory}.")
            lines.append("DETECTION DETAILS")
            for index, item in enumerate(visible_detections, start=1):
                label = str(item.get("label", "object"))
                try:
                    confidence = max(0.0, min(1.0, float(item.get("score", 0.0))))
                except (TypeError, ValueError, OverflowError):
                    confidence = 0.0
                box = item.get("box") if isinstance(item.get("box"), dict) else {}
                position = _position_for_box(box, width, height)
                if box and all(key in box for key in ("xmin", "ymin", "xmax", "ymax")):
                    bounds = (
                        f"box ({box['xmin']}, {box['ymin']}) to "
                        f"({box['xmax']}, {box['ymax']})"
                    )
                else:
                    bounds = "box coordinates unavailable"
                lines.append(
                    f"{index}. {label} — {confidence:.0%} confidence; "
                    f"approximate position: {position}; {bounds}."
                )
        elif scan_is_stale:
            lines.append("The old object list is omitted because it is more than 1.5 seconds old.")
        else:
            lines.append(
                "No objects were returned above the selected confidence threshold. "
                "That does not prove the scene is empty."
            )

    if metrics:
        lines.extend(["", "OPENCV FRAME CHECK"])
        if metrics.get("width") is not None and metrics.get("height") is not None:
            lines.append(f"Processed frame: {metrics['width']} × {metrics['height']} pixels.")
        brightness = metrics.get("brightness")
        if isinstance(brightness, (int, float)):
            label = str(metrics.get("brightness_label", "not classified"))
            lines.append(f"Average brightness: {brightness:.1f}/255 ({label}).")
        sharpness = metrics.get("sharpness")
        if isinstance(sharpness, (int, float)):
            label = str(metrics.get("sharpness_label", "rough estimate"))
            lines.append(f"Focus / edge-detail score: {sharpness:.1f} ({label}).")
        detector_ms = metrics.get("detector_ms")
        detector_fps = metrics.get("detector_fps")
        if isinstance(detector_ms, (int, float)):
            lines.append(f"Local object-detection inference: {detector_ms:.0f} ms for the current frame.")
        if isinstance(detector_fps, (int, float)) and detector_fps > 0:
            lines.append(f"Measured local detector throughput: {detector_fps:.1f} frames per second (host CPU estimate).")
    if frame_count:
        lines.append(f"Camera frames received in this session: {int(frame_count):,}.")

    if status.get("busy"):
        lines.extend(["", "STATUS", "Object detector is processing another sampled frame."])
    if status.get("model"):
        lines.append(f"Object-detection model: {status['model']}.")

    lines.extend([
        "",
        "LIMITATIONS",
        "These are model predictions, not certainty. Small, blurry, hidden, unusual, or out-of-frame objects may be missed or mislabeled. "
        "Positions are approximate, and this report describes visible content only; it does not verify a person's identity.",
    ])
    return "\n".join(lines)
