"""Cloud speech-to-text and concise voice replies for VisionQuest.

Audio is submitted only when the user presses the transcribe/reply button.
The raw clip is not written to the memory database by this module.
"""
from __future__ import annotations

import base64
import json
import os
import time
from typing import Any

import requests

from core.cloud_vision import MODEL_ID, get_hf_token

ASR_URL = os.getenv(
    "HF_ASR_URL",
    "https://router.huggingface.co/hf-inference/models/openai/whisper-large-v3",
)
CHAT_URL = "https://router.huggingface.co/v1/chat/completions"
MAX_AUDIO_BYTES = 20 * 1024 * 1024


def transcribe_audio(audio_bytes: bytes, content_type: str = "audio/wav") -> str:
    """Transcribe a short user-submitted or live-session speech clip through Hugging Face."""
    if not audio_bytes:
        raise ValueError("Record a short voice message first.")
    if len(audio_bytes) > MAX_AUDIO_BYTES:
        raise ValueError("That audio clip is too large (20 MB limit). Record a shorter message.")

    token = get_hf_token()
    if not token:
        raise RuntimeError("Voice input needs HF_TOKEN in the app's Streamlit Secrets.")
    started = time.perf_counter()
    try:
        response = requests.post(
            ASR_URL,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": content_type or "audio/wav",
                "Accept": "application/json",
            },
            data=audio_bytes,
            timeout=(10, 60),
        )
    except requests.Timeout as exc:
        raise RuntimeError("Speech transcription timed out. Try a shorter voice message.") from exc
    except requests.RequestException as exc:
        raise RuntimeError(f"Could not reach the speech service ({type(exc).__name__}).") from exc

    if response.status_code in (401, 403):
        raise RuntimeError("Hugging Face rejected HF_TOKEN. Check its Inference Providers permission.")
    if response.status_code in (402, 429):
        raise RuntimeError("The speech provider reported a credits or rate-limit issue.")
    if response.status_code in (503, 504):
        raise RuntimeError("Speech recognition is temporarily unavailable or loading. Please retry shortly.")
    if not response.ok:
        detail = ""
        try:
            detail = str(response.json().get("error", ""))[:240]
        except Exception:
            detail = response.text[:240]
        raise RuntimeError(f"Speech transcription failed with HTTP {response.status_code}. {detail}")

    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError("The speech service returned an invalid response.") from exc
    transcript = payload.get("text", payload.get("generated_text", "")) if isinstance(payload, dict) else ""
    transcript = str(transcript or "").strip()
    if not transcript:
        raise RuntimeError("No speech was recognized. Try speaking clearly in a shorter clip.")
    if len(transcript) > 1200:
        transcript = transcript[:1200].rsplit(" ", 1)[0]
    return transcript


def answer_voice_query(
    question: str,
    latest_observation: dict[str, Any] | None = None,
    memories: list[dict[str, Any]] | None = None,
    conversation_history: list[dict[str, str]] | None = None,
    live_image_bytes: bytes | None = None,
) -> str:
    """Answer a spoken question using only bounded live-observation and memory context."""
    question = (question or "").strip()
    if not question:
        raise ValueError("The spoken question is empty.")
    if len(question) > 1200:
        question = question[:1200]

    token = get_hf_token()
    if not token:
        raise RuntimeError("Voice replies need HF_TOKEN in the app's Streamlit Secrets.")

    observation_context: dict[str, Any] | None = None
    if latest_observation:
        observation_context = {
            "observed_at": str(latest_observation.get("timestamp", ""))[:80],
            "task": str(latest_observation.get("mode", ""))[:80],
            "observation": str(latest_observation.get("text", ""))[:1800],
        }
        # The on-device/server-side object detector is intentionally a small
        # vocabulary detector; pass its recent labels as supporting evidence,
        # not as a complete inventory of everything visible.
        detector_labels = []
        for item in (latest_observation.get("object_detections") or [])[:12]:
            if not isinstance(item, dict):
                continue
            label = str(item.get("label", "")).strip()[:48]
            try:
                confidence = max(0.0, min(1.0, float(item.get("confidence", 0.0))))
            except (TypeError, ValueError):
                confidence = 0.0
            if label:
                detector_labels.append({
                    "label": label,
                    "confidence": round(confidence, 2),
                })
        if detector_labels:
            observation_context["recent_local_object_detections"] = detector_labels
            observation_context["object_detection_age_seconds"] = latest_observation.get(
                "object_detection_age_seconds"
            )
            observation_context["object_detection_model"] = str(
                latest_observation.get("object_detection_model", "")
            )[:100]
    memory_context = []
    for item in (memories or [])[:10]:
        memory_context.append({
            "person_label": str(item.get("person_name", ""))[:100],
            "relationship_label": str(item.get("relationship", ""))[:100],
            "saved_at": str(item.get("created_at", ""))[:80],
            "note": str(item.get("note", ""))[:600],
        })
    recent_turns = []
    for turn in (conversation_history or [])[-4:]:
        user_text = str(turn.get("question", "")).strip()[:500]
        assistant_text = str(turn.get("reply", "")).strip()[:700]
        if user_text and assistant_text:
            recent_turns.append({"user": user_text, "assistant": assistant_text})
    context = {
        "latest_visual_observation": observation_context,
        "recent_saved_memory_notes": memory_context,
        "recent_conversation_turns": recent_turns,
    }
    user_text = (
        "Context data (may be empty):\n"
        + json.dumps(context, ensure_ascii=False)
        + "\n\nSpoken question: "
        + question
        + (
            "\n\nA fresh camera frame from the live camera is attached. Base answers about the current scene on this image. "
            "You may use the recent observation and local detector labels as supporting evidence."
            if live_image_bytes else
            "\n\nNo fresh camera frame is attached. Use the latest cached visual observation only as prior evidence, "
            "not as proof that the scene is unchanged."
        )
    )
    user_content: str | list[dict[str, Any]] = user_text
    if live_image_bytes:
        user_content = [
            {"type": "text", "text": user_text},
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/jpeg;base64,"
                    + base64.b64encode(live_image_bytes).decode("ascii")
                },
            },
        ]

    messages = [
        {
            "role": "system",
            "content": (
                "You are Smriti AI, a calm, friendly voice companion. Reply in the user's language, "
                "using 1–4 short spoken sentences. For questions about what is happening now, what you can "
                "see, or what you just observed, ground the answer in the attached fresh camera frame when "
                "one is supplied. Cross-check it with the latest_visual_observation and recent_local_object_detections "
                "when present. The detector labels are supporting hints only, not a complete inventory; the "
                "image is the main visual evidence. If no fresh image is attached, answer from the saved "
                "observation only and be clear when that observation may not describe the current scene. "
                "For personal-history questions, use only supplied saved memory notes and conversation context. "
                "For broad recall requests such as 'I forgot' or 'help me remember', review the supplied notes and "
                "offer the closest explicitly saved fact. Phrase it as 'A note you saved says...' rather than claiming "
                "to know exactly what the user forgot; if several notes could fit, offer brief choices. "
                "Never invent observations or memories, identify a person from the camera, diagnose a condition, "
                "or claim certainty the evidence does not support. If the context does not contain the answer, "
                "say you do not have that information yet. General greetings and simple general questions are "
                "okay. Do not claim to have saved anything unless the app confirms it."
            ),
        },
        {"role": "user", "content": user_content},
    ]
    started = time.perf_counter()
    try:
        response = requests.post(
            CHAT_URL,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            json={
                "model": MODEL_ID,
                "stream": False,
                "max_tokens": 180,
                "temperature": 0.2,
                "messages": messages,
            },
            timeout=(10, 45),
        )
    except requests.Timeout as exc:
        raise RuntimeError("The voice reply took too long. Try asking a shorter question.") from exc
    except requests.RequestException as exc:
        raise RuntimeError(f"Could not reach the reply service ({type(exc).__name__}).") from exc

    if response.status_code in (401, 403):
        raise RuntimeError("Hugging Face rejected HF_TOKEN. Check its Inference Providers permission.")
    if response.status_code in (402, 429):
        raise RuntimeError("The reply provider reported a credits or rate-limit issue.")
    if response.status_code in (503, 504):
        raise RuntimeError("The reply model is temporarily unavailable or loading. Please retry shortly.")
    if not response.ok:
        detail = ""
        try:
            detail = str(response.json().get("error", ""))[:240]
        except Exception:
            detail = response.text[:240]
        raise RuntimeError(f"The voice reply failed with HTTP {response.status_code}. {detail}")

    try:
        payload = response.json()
        content = payload["choices"][0]["message"]["content"]
        if isinstance(content, list):
            content = " ".join(
                str(part.get("text", "")) for part in content if isinstance(part, dict)
            )
        answer = str(content or "").strip()
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("The reply service returned an unexpected response.") from exc
    if not answer:
        raise RuntimeError("The reply model returned an empty answer. Please ask again.")
    return answer[:1800]
