"""Remote vision inference for VisionQuest Cloud.

The shared Streamlit host has limited RAM, so this module sends a resized image
to Hugging Face Inference Providers instead of loading PyTorch/Transformers inside
the Streamlit process. Configure HF_TOKEN in Streamlit Secrets.
"""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import base64
import os
import time
from typing import Any

import requests
from PIL import Image, ImageOps, UnidentifiedImageError

MODEL_ID = os.getenv("HF_VISION_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct:fastest")
INFERENCE_URL = "https://router.huggingface.co/v1/chat/completions"
MAX_UPLOAD_BYTES = 8 * 1024 * 1024
REQUEST_TIMEOUT = (10, 90)

MODES: dict[str, str] = {
    "auto_observe": (
        "Inspect the image carefully and produce a detailed, evidence-based observation using these headings: "
        "SCENE; VISIBLE OBJECTS AND POSITIONS; ACTIONS AND INTERACTIONS; FINE DETAILS; READABLE TEXT; UNCERTAINTIES. "
        "Under SCENE, report the overall scene and setting. Under VISIBLE OBJECTS AND POSITIONS, inventory clearly visible "
        "everyday objects separately where possible, with approximate position and foreground/background. Under ACTIONS AND "
        "INTERACTIONS, report visible actions or activities and spatial relationships only when supported. Include visible "
        "color, shape, relative size, orientation, and condition only when genuinely supported. Pay special attention to "
        "smaller objects, items being handled, signs, labels, and background details if they are clear enough to report. "
        "Under READABLE TEXT, copy only readable text and mark partially readable text as uncertain. Under UNCERTAINTIES, "
        "explain what is ambiguous or obscured. Do not infer identity, emotion, intent, relationships, or events outside "
        "this single frame. Do not claim to detect everything, and do not guess obscured or ambiguous objects."
    ),
    "scene": "Describe the visible scene in 2–4 concise sentences. Mention only clearly visible objects, actions, and setting.",
    "objects": "List clearly visible objects and say where each appears in the frame (left, center, right, foreground, or background). Do not invent unseen objects.",
    "read_text": "Transcribe only text that is visibly readable in the image. Preserve wording. Mark uncertain text as [unclear]. Do not invent missing words.",
    "accessibility": "Give a calm spoken-style description in no more than 35 words. Prioritize important visible details. Do not infer identity or personal circumstances.",
    "visual_question": "Answer the user's question using only evidence from the image. If it cannot be determined, say so clearly.",
}
MODE_LABELS = {
    "auto_observe": "Auto-observe: scene + objects + activities",
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


def get_hf_token() -> str:
    """Read the secret from the hosting platform without ever displaying it."""
    token = os.getenv("HF_TOKEN", "").strip() or os.getenv("HUGGINGFACEHUB_API_TOKEN", "").strip()
    if token:
        return token
    try:
        import streamlit as st
        token = str(st.secrets.get("HF_TOKEN", "")).strip()
    except Exception:
        token = ""
    return token


def model_status() -> tuple[bool, str]:
    if not get_hf_token():
        return False, (
            "Remote vision is not configured. Add HF_TOKEN under Streamlit → Manage app → Settings → Secrets. "
            "Do not paste the token into chat or GitHub."
        )
    return True, f"Remote vision endpoint configured ({MODEL_ID})."


def prepare_image(image_bytes: bytes, max_side: int = 512) -> tuple[Image.Image, tuple[int, int]]:
    """Validate and downscale an uploaded image before remote inference."""
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
        "You are VisionQuest, a precise visual observation assistant. Carefully inspect the supplied image and report "
        "specific visible evidence rather than a vague summary. Distinguish direct observations from uncertainty. Do not "
        "identify a person, infer relationships or personal history, diagnose health conditions, or guess emotions or intent. "
        "Do not invent off-screen details, and do not treat likely guesses as facts. Avoid repeating the same detail. "
        "Do not greet the user or give generic advice.\n\nTask: "
        + task
    )
    if focus.strip():
        prompt += "\n\nUser question/focus: " + focus.strip()[:300]
    return prompt + "\n\nKeep the answer concise but detailed enough to cover all clearly visible evidence; do not pad the response. Stay directly focused on this image."


def _extract_response_text(payload: dict[str, Any]) -> str:
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("The remote vision service returned an unexpected response format.") from exc
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        pieces = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                pieces.append(str(part.get("text", "")))
        return "\n".join(pieces).strip()
    return str(content or "").strip()


def analyze_image(
    image_bytes: bytes,
    mode: str = "scene",
    focus: str = "",
    max_side: int = 512,
) -> VisionResult:
    """Analyze one image using a remote hosted VLM without loading it into app RAM."""
    token = get_hf_token()
    if not token:
        raise RuntimeError(
            "Vision AI has not been configured. Add HF_TOKEN in Streamlit Secrets "
            "with Inference Providers permission, then reboot the app."
        )

    image, original_size = prepare_image(image_bytes, max_side=max_side)
    buffer = BytesIO()
    # Preserve fine visual details for the hosted vision-language model. Live camera
    # samples are downsized before this step, so high JPEG quality does not create pixels.
    image.save(buffer, format="JPEG", quality=95, optimize=True)
    image_b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
    data_url = "data:image/jpeg;base64," + image_b64

    # Detailed live observations need a larger response budget than simple scene summaries.
    max_tokens = 700 if mode == "auto_observe" else (
        500 if mode in ("objects", "read_text", "visual_question") else 320
    )
    payload = {
        "model": MODEL_ID,
        "stream": False,
        "max_tokens": max_tokens,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": build_prompt(mode, focus)},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }],
    }
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    started = time.perf_counter()
    try:
        response = requests.post(
            INFERENCE_URL,
            headers=headers,
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
    except requests.Timeout as exc:
        raise RuntimeError(
            "The remote vision request timed out after 90 seconds. Check the provider/model status and retry once."
        ) from exc
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Could not reach the remote vision service ({type(exc).__name__}). Check the network and retry."
        ) from exc

    if response.status_code in (401, 403):
        raise RuntimeError(
            "Hugging Face rejected HF_TOKEN. Check that the secret is valid and has Inference Providers permission."
        )
    if response.status_code in (402, 429):
        raise RuntimeError(
            "The hosted inference provider reported a billing/credit or rate-limit issue. Check Hugging Face Inference Providers usage before retrying."
        )
    if response.status_code in (503, 504):
        raise RuntimeError(
            "The selected vision model/provider is loading or temporarily unavailable. Wait briefly and retry."
        )
    if not response.ok:
        detail = ""
        try:
            detail = str(response.json().get("error", ""))[:300]
        except Exception:
            detail = response.text[:300]
        raise RuntimeError(
            f"Remote vision request failed with HTTP {response.status_code}. {detail or 'No provider detail was returned.'}"
        )

    try:
        answer = _extract_response_text(response.json())
    except ValueError as exc:
        raise RuntimeError("The remote vision service did not return valid JSON.") from exc
    elapsed = round(time.perf_counter() - started, 2)
    if not answer:
        raise RuntimeError("The remote vision model returned an empty answer. Try another frame.")
    if len(answer) > 6000:
        answer = answer[:6000].rsplit(" ", 1)[0] + "…"
    return VisionResult(answer, MODEL_ID, elapsed, original_size, image.size)
