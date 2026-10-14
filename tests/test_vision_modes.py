"""Fast, model-free tests for VisionQuest's selectable vision tasks and image validation."""
from __future__ import annotations

from io import BytesIO

import pytest
from PIL import Image

from core.cloud_vision import MODE_LABELS, build_prompt, prepare_image, MAX_UPLOAD_BYTES


def test_every_visible_vision_mode_has_a_grounded_prompt():
    expected = {"auto_observe", "scene", "objects", "read_text", "accessibility", "live_companion", "visual_question"}
    assert set(MODE_LABELS) == expected
    for mode in MODE_LABELS:
        prompt = build_prompt(mode)
        assert "You are VisionQuest" in prompt
        assert "Task:" in prompt
        assert "Do not identify a person" in prompt
        assert "Keep the answer concise" in prompt


def test_visual_question_includes_user_question():
    prompt = build_prompt("visual_question", "Is there a red cup on the table?")
    assert "Is there a red cup on the table?" in prompt


def test_prepare_image_resizes_and_preserves_original_dimensions():
    image = Image.new("RGB", (1200, 600), color=(25, 60, 80))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    prepared, original = prepare_image(buffer.getvalue(), max_side=512)
    assert original == (1200, 600)
    assert prepared.size == (512, 256)


def test_prepare_image_rejects_empty_and_invalid_data():
    with pytest.raises(ValueError, match="Choose an image"):
        prepare_image(b"")
    with pytest.raises(ValueError, match="valid supported image"):
        prepare_image(b"this is not an image")


def test_prepare_image_rejects_oversized_upload():
    with pytest.raises(ValueError, match="8 MB"):
        prepare_image(b"x" * (MAX_UPLOAD_BYTES + 1))

def test_auto_observe_prompt_covers_scene_objects_actions_and_text():
    prompt = build_prompt("auto_observe")
    assert "overall scene and setting" in prompt
    assert "everyday objects" in prompt
    assert "visible actions or activities" in prompt
    assert "readable text" in prompt
    assert "Do not claim to detect everything" in prompt


