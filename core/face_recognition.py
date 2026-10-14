"""Small, local face matching prototype built on OpenCV's LBPH recognizer.

Only compare camera frames with explicitly enrolled reference portraits. This is an
experimental assistive feature, not a security or medical decision system.
"""
from __future__ import annotations

from io import BytesIO
from typing import Callable, Any

import cv2
import numpy as np


def _face_cascade() -> cv2.CascadeClassifier:
    path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(path)
    if cascade.empty():
        raise RuntimeError("OpenCV's frontal-face detector could not be loaded.")
    return cascade


def decode_image_bytes(image_bytes: bytes) -> np.ndarray:
    """Decode common uploaded image formats into an OpenCV BGR image."""
    if not image_bytes:
        raise ValueError("The image is empty.")
    image = cv2.imdecode(np.frombuffer(bytes(image_bytes), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise ValueError("Could not read that image. Choose a JPEG, PNG, or WEBP photo.")
    return image


def encode_jpeg(frame_bgr: np.ndarray, max_side: int = 960) -> bytes:
    """Encode a frame as a reasonably small JPEG suitable for portrait storage."""
    if not isinstance(frame_bgr, np.ndarray) or frame_bgr.size == 0:
        raise ValueError("No camera frame is available.")
    image = frame_bgr.copy()
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest > max_side:
        scale = max_side / float(longest)
        image = cv2.resize(
            image,
            (max(1, int(width * scale)), max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    for quality in (88, 78, 68):
        ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if ok and len(encoded) <= 3 * 1024 * 1024:
            return encoded.tobytes()
    raise ValueError("The reference image is too large to store; choose a smaller photo.")


def detect_faces(frame_bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Detect face regions (not identities) in a BGR frame."""
    if not isinstance(frame_bgr, np.ndarray) or frame_bgr.size == 0:
        return []
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError("Face detection requires a three-channel BGR image.")
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    cascade = _face_cascade()
    found = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )
    return [(int(x), int(y), int(w), int(h)) for x, y, w, h in found]


def face_recognizer_available() -> bool:
    """Whether this OpenCV build includes the optional contrib face recognizer."""
    return hasattr(cv2, "face") and hasattr(cv2.face, "LBPHFaceRecognizer_create")


def _normalized_face(frame_bgr: np.ndarray) -> np.ndarray | None:
    try:
        faces = detect_faces(frame_bgr)
    except (RuntimeError, ValueError, cv2.error):
        return None
    if not faces:
        return None
    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    crop = gray[y : y + h, x : x + w]
    if crop.size == 0:
        return None
    crop = cv2.resize(crop, (160, 160), interpolation=cv2.INTER_AREA)
    return cv2.equalizeHist(crop)


def recognize_faces(
    frame_bgr: np.ndarray,
    people: list[dict[str, Any]],
    get_portrait: Callable[[int], dict | None],
    distance_threshold: float = 65.0,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Recognize detected faces against saved, consented portraits; mark uncertain faces Unknown."""
    if not face_recognizer_available():
        raise RuntimeError(
            "Face matching needs the OpenCV contrib build. Reinstall dependencies with "
            "'python -m pip install --force-reinstall opencv-contrib-python-headless==4.14.0.94'."
        )
    if not isinstance(frame_bgr, np.ndarray) or frame_bgr.size == 0:
        raise ValueError("Start the camera and wait for a frame before recognizing faces.")

    samples: list[np.ndarray] = []
    labels: list[int] = []
    label_names: dict[int, str] = {}

    for person in people:
        try:
            person_id = int(person["id"])
            name = str(person.get("name") or "Known person").strip()
            portrait = get_portrait(person_id)
            if not portrait or not portrait.get("image_data"):
                continue
            reference = decode_image_bytes(bytes(portrait["image_data"]))
            sample = _normalized_face(reference)
            if sample is None:
                continue
            samples.append(sample)
            labels.append(person_id)
            label_names[person_id] = name
        except (KeyError, TypeError, ValueError, cv2.error):
            continue

    if not samples:
        raise ValueError(
            "No saved reference portraits contain a detectable face. Register a person using a clear, front-facing photo first."
        )

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(samples, np.asarray(labels, dtype=np.int32))

    annotated = frame_bgr.copy()
    gray = cv2.equalizeHist(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY))
    rectangles = detect_faces(frame_bgr)
    results: list[dict[str, Any]] = []
    for x, y, w, h in rectangles:
        crop = cv2.resize(gray[y : y + h, x : x + w], (160, 160), interpolation=cv2.INTER_AREA)
        crop = cv2.equalizeHist(crop)
        predicted_id, distance = recognizer.predict(crop)
        known = int(predicted_id) in label_names and float(distance) <= float(distance_threshold)
        name = label_names.get(int(predicted_id), "Unknown") if known else "Unknown"
        colour = (65, 235, 165) if known else (20, 190, 255)
        cv2.rectangle(annotated, (x, y), (x + w, y + h), colour, 2)
        cv2.rectangle(annotated, (x, max(0, y - 28)), (x + w, y), colour, -1)
        cv2.putText(
            annotated,
            name[:32],
            (x + 5, max(16, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (4, 14, 20),
            2,
            cv2.LINE_AA,
        )
        results.append(
            {
                "name": name,
                "recognized": bool(known),
                "distance": round(float(distance), 1),
                "box": (x, y, w, h),
            }
        )

    return annotated, results
