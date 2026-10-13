"""Small cloud-hosted image understanding through Transformers and SmolVLM."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
import logging
import os
import time

from PIL import Image, ImageOps, UnidentifiedImageError

MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"
MAX_UPLOAD_BYTES = 8 * 1024 * 1024

MODES: dict[str, str] = {
    "scene": "Describe the visible scene in 2–4 concise sentences. Mention only clearly visible objects, actions, and setting.",
    "objects": "List clearly visible objects and say where each appears in the frame (left, center, right, foreground, or background). Do not invent unseen objects.",
    "read_text": "Transcribe only text that is visibly readable in the image. Preserve wording. Mark uncertain text as [unclear]. Do not invent missing words.",
    "accessibility": "Give a calm spoken-style description in no more than 35 words. Prioritize important visible details. Do not infer identity or personal circumstances.",
    "visual_question": "Answer the user's question using only evidence from the image. If it cannot be determined, say so clearly.",
}
MODE_LABELS = {
    "scene": "Scene overview",
    "objects": "Find and locate objects",
    "read_text": "Read visible text",
    "accessibility": "Short spoken description",
    "visual_question": "Ask a question about the image",
}


@dataclass(frozen=True)
class VisionResult:
    text: str
    model: str
    elapsed_seconds: float
    original_size: tuple[int, int]
    processed_size: tuple[int, int]


def prepare_image(image_bytes: bytes, max_side: int = 512) -> tuple[Image.Image, tuple[int, int]]:
    """Validate and downscale an uploaded image before inference."""
    if not image_bytes:
        raise ValueError("Choose an image first.")
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise ValueError("Image is larger than 8 MB. Resize it or choose another image.")
    if max_side < 64:
        raise ValueError("Image detail must be at least 64 pixels.")
    try:
        image = Image.open(BytesIO(image_bytes))
        image = ImageOps.exif_transpose(image).convert("RGB")
        original_size = image.size
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        return image, original_size
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("This file is not a valid supported image. Try a JPG or PNG image.") from exc


def build_prompt(mode: str, focus: str = "") -> str:
    task = MODES.get(mode, MODES["scene"])
    prompt = (
        "You are VisionQuest, a cautious visual assistant. Report only what the image supports. "
        "Do not identify a person, infer a relationship, diagnose a health condition, or guess emotions "
        "or personal history. If unsure, say what is uncertain. Do not greet the user or give generic advice.\n\nTask: "
        + task
    )
    if focus.strip():
        prompt += "\n\nUser question/focus: " + focus.strip()[:300]
    return prompt + "\n\nKeep the answer concise and directly about this image."


def model_status() -> tuple[bool, str]:
    return True, "Cloud model loads on first analysis; no local Ollama installation required."


def _cpu_count() -> int:
    return os.cpu_count() or 1


@lru_cache(maxsize=1)
def _load_model():
    """Load one cached, reduced-memory CPU model instance."""
    import torch
    from transformers import AutoModelForVision2Seq, AutoProcessor

    # Keep the CPU pool small on shared cloud hosts and load 16-bit weights
    # directly from safetensors to avoid a full FP32 model copy in RAM.
    torch.set_num_threads(1)
    logging.getLogger(__name__).info(
        "VisionQuest: loading %s with bfloat16 CPU weights; image analysis is starting.",
        MODEL_ID,
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    model.eval()
    logging.getLogger(__name__).info("VisionQuest: vision model loaded successfully.")
    return processor, model


def analyze_image(
    image_bytes: bytes,
    mode: str = "scene",
    focus: str = "",
    max_side: int = 512,
) -> VisionResult:
    """Run one image analysis and return observation plus processing metadata."""
    import torch

    image, original_size = prepare_image(image_bytes, max_side=max_side)
    try:
        import streamlit as st
        # Streamlit stores processor/model once per app process.
        processor, model = _streamlit_cached_model()
    except ImportError:
        processor, model = _load_model()

    chat = [{
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": build_prompt(mode, focus)}],
    }]
    prompt = processor.apply_chat_template(chat, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt")
    started = time.perf_counter()
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs, max_new_tokens=48, do_sample=False, num_beams=1, use_cache=False
        )
    new_tokens = generated_ids[:, inputs["input_ids"].shape[1]:]
    answer = processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
    elapsed = round(time.perf_counter() - started, 2)
    if not answer:
        raise RuntimeError("The vision model returned an empty answer. Try another image or retry once.")
    if len(answer) > 6000:
        answer = answer[:6000].rsplit(" ", 1)[0] + "…"
    return VisionResult(answer, MODEL_ID, elapsed, original_size, image.size)


def _streamlit_cached_model():
    import streamlit as st

    @st.cache_resource(show_spinner=False)
    def load_once():
        return _load_model()

    return load_once()
