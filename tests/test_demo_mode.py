"""Offline demo and face-recognition helper tests."""
from __future__ import annotations

from io import BytesIO

import cv2
import numpy as np
import pytest
from PIL import Image

from core.demo_mode import demo_observation, demo_reply
from core.face_recognition import decode_image_bytes, detect_faces, face_recognizer_available


def test_demo_replies_are_immediate_scripted_and_explicit():
    assert demo_reply("Hello Smriti").startswith("Hello!")
    memory = demo_reply("What do you remember?")
    assert "DEMO SAMPLE MEMORY" in memory
    assert "not a real saved memory" in memory
    generic = demo_reply("Tell me something")
    assert "DEMO RESPONSE" in generic
    assert "scripted sample answer" in generic


def test_demo_observation_always_discloses_simulation():
    assert "fictional sample content" in demo_observation()
    assert "fictional sample content" in demo_observation(camera_available=True, detected_faces=1)


def test_face_helpers_validate_images_and_return_empty_for_no_faces():
    with pytest.raises(ValueError, match="image is empty"):
        decode_image_bytes(b"")

    with pytest.raises(ValueError, match="Could not read"):
        decode_image_bytes(b"not an image")

    image = np.zeros((180, 240, 3), dtype=np.uint8)
    assert detect_faces(image) == []


def test_face_recognizer_availability_matches_installed_opencv_build():
    # CI/runtime requirements install the contrib build for LBPH matching.
    assert face_recognizer_available(), (
        "Install opencv-contrib-python-headless rather than opencv-python-headless."
    )
