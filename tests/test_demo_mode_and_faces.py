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


from core.local_granite import clean_granite_reply, prepare_ollama_image
from core.local_archive import archive_event, save_media, get_archive_dir
import json


def test_granite_reply_cleaner_hides_reasoning_trace():
    raw = "Thinking...\nThinking Process:\n1. Analyse the prompt.\n2. Choose response."
    result = clean_granite_reply(raw)
    assert "Thinking Process" not in result
    assert len(result) < 100


def test_granite_reply_cleaner_removes_think_tag():
    assert clean_granite_reply("<think>private reasoning</think>Hello!") == "Hello!"


def test_granite_reply_cleaner_keeps_normal_short_reply():
    assert clean_granite_reply("Hello, I'm ready to help.") == "Hello, I'm ready to help."



def test_granite_observer_image_prep_respects_detail_bound():
    source = BytesIO()
    Image.new("RGB", (1280, 720), color=(80, 140, 180)).save(source, format="PNG")

    prepared = prepare_ollama_image(source.getvalue(), max_side=640, quality=84)

    with Image.open(BytesIO(prepared)) as result:
        assert result.format == "JPEG"
        assert result.size == (640, 360)


def test_granite_observer_image_prep_rejects_empty_image():
    try:
        prepare_ollama_image(b"", max_side=640)
    except ValueError as exc:
        assert "No camera image" in str(exc)
    else:
        raise AssertionError("Empty image bytes should be rejected")



def test_local_archive_writes_tagged_jsonl_event(tmp_path, monkeypatch):
    monkeypatch.setenv("SMRITI_ARCHIVE_DIR", str(tmp_path / "smriti-archive"))

    path = archive_event(
        "conversation_turn",
        {"question": "Where are my keys?", "answer": "I do not have a saved record."},
        {"people": "Sam", "place": "home", "activity": "  ", "ignored": "not a supported tag"},
    )

    assert path.is_file()
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["type"] == "conversation_turn"
    assert rows[0]["question"] == "Where are my keys?"
    assert rows[0]["tags"] == {"people": "Sam", "place": "home"}


def test_local_archive_saves_media_and_indexes_it(tmp_path, monkeypatch):
    root = tmp_path / "smriti-archive"
    monkeypatch.setenv("SMRITI_ARCHIVE_DIR", str(root))

    saved = save_media(
        "audio",
        b"test-audio-bytes",
        "webm",
        "audio/webm",
        {"people": "A person with permission", "place": "garden"},
        {"maximum_recording_seconds": 30},
    )

    assert saved.is_file()
    assert saved.read_bytes() == b"test-audio-bytes"
    assert saved.parent == root / "audio"
    events = [json.loads(line) for line in (root / "activity.jsonl").read_text(encoding="utf-8").splitlines()]
    assert events[-1]["type"] == "media_saved"
    assert events[-1]["media_kind"] == "audio"
    assert events[-1]["tags"]["place"] == "garden"


def test_local_archive_rejects_unknown_media_type(tmp_path, monkeypatch):
    monkeypatch.setenv("SMRITI_ARCHIVE_DIR", str(tmp_path / "smriti-archive"))
    try:
        save_media("audio", b"data", "exe", "application/octet-stream")
    except ValueError as exc:
        assert "Unsupported archive file type" in str(exc)
    else:
        raise AssertionError("Unsupported media types must be rejected")
