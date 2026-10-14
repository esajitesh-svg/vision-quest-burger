"""Offline demo and face-module regression tests."""
from __future__ import annotations

from io import BytesIO

import cv2
import numpy as np
from PIL import Image

from core.demo_mode import demo_observation, demo_reply
from core.face_recognition import decode_image_bytes, detect_faces, encode_jpeg


def test_demo_answer_is_immediate_scripted_and_labelled():
    assert demo_reply("What do you remember?").startswith("DEMO SAMPLE MEMORY:")
    assert "not a live AI-generated response" in demo_reply("Explain a rocket")
    assert "fictional sample content" in demo_observation(camera_available=False)


def test_demo_observation_discloses_camera_availability():
    result = demo_observation(camera_available=True, detected_faces=1)
    assert "not an analysis of the camera frame" in result
    assert "found 1 face region" in result


def test_face_image_codec_round_trip():
    image = np.zeros((180, 240, 3), dtype=np.uint8)
    image[40:140, 70:170] = (100, 130, 160)
    encoded = encode_jpeg(image)
    decoded = decode_image_bytes(encoded)
    assert decoded.shape == image.shape
    assert len(encoded) > 100


def test_face_detector_handles_blank_frame():
    blank = np.zeros((240, 320, 3), dtype=np.uint8)
    assert detect_faces(blank) == []


def test_face_image_decoder_rejects_bad_data():
    try:
        decode_image_bytes(b"not an image")
    except ValueError as exc:
        assert "Could not read" in str(exc)
    else:
        raise AssertionError("Invalid image data should be rejected")
