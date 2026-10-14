"""Remote object detection for VisionQuest using Hugging Face Inference Providers.

The endpoint returns discrete class labels, confidence scores, and bounding boxes.
This is separate from the natural-language scene-description model.
"""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import math
import time
from typing import Any

import requests
from PIL import Image, UnidentifiedImageError

from core.cloud_vision import get_hf_token

OBJECT_DETECTION_MODEL = "facebook/detr-resnet-50"
OBJECT_DETECTION_URL = "https://router.huggingface.co/hf-inference/models/facebook/detr-resnet-50"
MAX_IMAGE_BYTES = 8 * 1024 * 1024
REQUEST_TIMEOUT = (10, 60)


@dataclass(frozen=True)
class DetectionResult:
    detections: list[dict[str, Any]]
    model: str
    elapsed_seconds: float
    image_size: tuple[int, int]


def detect_objects(
    image_bytes: bytes,
    token: str | None = None,
    score_threshold: float = 0.40,
    max_detections: int = 20,
) -> DetectionResult:
    """Detect common everyday objects in an image through a hosted model."""
    if not image_bytes:
        raise ValueError("The detector received an empty image.")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise ValueError("The detector image exceeds the 8 MB request limit.")
    if not 0.0 <= score_threshold <= 1.0:
        raise ValueError("Detection confidence threshold must be between 0 and 1.")
    if max_detections < 1 or max_detections > 100:
        raise ValueError("Maximum detections must be between 1 and 100.")

    try:
        with Image.open(BytesIO(image_bytes)) as image:
            image_size = image.size
            image.verify()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("The detector requires a valid image such as JPEG or PNG.") from exc

    access_token = (token or "").strip() or get_hf_token()
    if not access_token:
        raise RuntimeError(
            "Live object detection needs HF_TOKEN in Streamlit Secrets with Inference Providers permission."
        )

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "image/jpeg",
        "Accept": "application/json",
    }
    started = time.perf_counter()
    try:
        response = requests.post(
            OBJECT_DETECTION_URL,
            headers=headers,
            data=image_bytes,
            timeout=REQUEST_TIMEOUT,
        )
    except requests.Timeout as exc:
        raise RuntimeError("The object detector timed out. Wait a moment, or increase the detection interval.") from exc
    except requests.RequestException as exc:
        raise RuntimeError(f"Could not reach the object-detection provider ({type(exc).__name__}).") from exc

    if response.status_code in (401, 403):
        raise RuntimeError("Hugging Face rejected the detection request. Check HF_TOKEN and its Inference Providers permission.")
    if response.status_code in (402, 429):
        raise RuntimeError("The object-detection provider reported a credit, billing, or rate-limit issue.")
    if response.status_code in (503, 504):
        raise RuntimeError("The object-detection model/provider is loading or temporarily unavailable. Try again shortly.")
    if not response.ok:
        detail = ""
        try:
            payload = response.json()
            detail = str(payload.get("error", ""))[:240] if isinstance(payload, dict) else ""
        except (ValueError, TypeError):
            detail = getattr(response, "text", "")[:240]
        raise RuntimeError(
            f"Object detection failed with HTTP {response.status_code}. "
            f"{detail or 'The provider returned no additional details.'}"
        )

    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError("The object detector returned invalid JSON.") from exc
    if isinstance(payload, dict) and payload.get("error"):
        raise RuntimeError(f"Object detector error: {str(payload['error'])[:300]}")
    if not isinstance(payload, list):
        raise RuntimeError("The object detector returned an unexpected response format.")

    width, height = image_size
    detections: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        box = item.get("box")
        if not isinstance(box, dict):
            continue
        try:
            score = float(item.get("score", 0.0))
            coords = {
                name: float(box[name])
                for name in ("xmin", "ymin", "xmax", "ymax")
            }
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        if not math.isfinite(score) or not all(math.isfinite(value) for value in coords.values()):
            continue
        if score < score_threshold:
            continue

        xmin = max(0.0, min(float(width - 1), coords["xmin"]))
        xmax = max(0.0, min(float(width - 1), coords["xmax"]))
        ymin = max(0.0, min(float(height - 1), coords["ymin"]))
        ymax = max(0.0, min(float(height - 1), coords["ymax"]))
        if xmax <= xmin or ymax <= ymin:
            continue
        label = str(item.get("label", "object")).strip()[:80] or "object"
        detections.append({
            "label": label,
            "score": max(0.0, min(1.0, score)),
            "box": {
                "xmin": int(round(xmin)),
                "ymin": int(round(ymin)),
                "xmax": int(round(xmax)),
                "ymax": int(round(ymax)),
            },
        })

    detections.sort(key=lambda record: record["score"], reverse=True)
    elapsed = round(time.perf_counter() - started, 2)
    return DetectionResult(detections[:max_detections], OBJECT_DETECTION_MODEL, elapsed, image_size)
