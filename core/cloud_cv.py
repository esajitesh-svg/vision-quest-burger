"""OpenCV preprocessing for browser-captured or uploaded cloud images.

This module processes image bytes on the hosting server. It does not attempt to
open the visitor's webcam with cv2.VideoCapture; in cloud deployment the browser
must capture the image first.
"""
from __future__ import annotations

from io import BytesIO
import cv2
import numpy as np

MAX_IMAGE_BYTES = 8 * 1024 * 1024


def process_image_opencv(
    image_bytes: bytes,
    max_side: int = 512,
    enhance: bool = False,
) -> tuple[bytes, dict]:
    """Decode, resize, optionally enhance, and JPEG-encode an image with OpenCV.

    Returns the processed JPEG bytes plus basic image-quality metrics.
    """
    if not image_bytes:
        raise ValueError("Choose or capture an image first.")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise ValueError("Image is larger than 8 MB. Choose a smaller image.")
    if max_side < 64 or max_side > 2048:
        raise ValueError("Image processing size must be between 64 and 2048 pixels.")

    encoded = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if frame is None or frame.size == 0:
        raise ValueError("OpenCV could not decode this image. Try JPG or PNG.")

    original_height, original_width = frame.shape[:2]
    scale = min(1.0, max_side / max(original_width, original_height))
    if scale < 1.0:
        new_width = max(1, int(round(original_width * scale)))
        new_height = max(1, int(round(original_height * scale)))
        frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = float(cv2.mean(gray)[0])
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    if enhance:
        # Apply local contrast to luminance and gentle unsharp masking.
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        luminance, channel_a, channel_b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8))
        luminance = clahe.apply(luminance)
        frame = cv2.cvtColor(cv2.merge((luminance, channel_a, channel_b)), cv2.COLOR_LAB2BGR)
        blurred = cv2.GaussianBlur(frame, (0, 0), sigmaX=1.0)
        frame = cv2.addWeighted(frame, 1.12, blurred, -0.12, 0)

    ok, output = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise RuntimeError("OpenCV could not encode the processed image.")

    height, width = frame.shape[:2]
    if brightness < 55:
        brightness_label = "dark"
    elif brightness > 205:
        brightness_label = "bright"
    else:
        brightness_label = "normal"
    if sharpness < 45:
        sharpness_label = "possibly soft/blurry"
    else:
        sharpness_label = "reasonable edge detail"

    metrics = {
        "original_width": int(original_width),
        "original_height": int(original_height),
        "processed_width": int(width),
        "processed_height": int(height),
        "brightness": round(brightness, 1),
        "brightness_label": brightness_label,
        "sharpness_score": round(sharpness, 1),
        "sharpness_label": sharpness_label,
        "input_bytes": len(image_bytes),
        "output_bytes": int(output.nbytes),
        "enhanced": bool(enhance),
    }
    return output.tobytes(), metrics
