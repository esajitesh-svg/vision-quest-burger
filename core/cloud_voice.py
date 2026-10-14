"""Cloud speech-to-text and concise voice replies for VisionQuest.

Audio is submitted only when the user presses the transcribe/reply button.
The raw clip is not written to the memory database by this module.
"""
from __future__ import annotations

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
    """Transcribe one user-recorded audio clip through Hugging Face Inference Providers."""
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

    observation_context: dict[str, str] | None = None
    if latest_observation:
        observation_context = {
            "observed_at": str(latest_observation.get("timestamp", ""))[:80],
            "task": str(latest_observation.get("mode", ""))[:80],
            "observation": str(latest_observation.get("text", ""))[:1800],
        }
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
    messages = [
        {
            "role": "system",
            "content": (
                "You are Smriti AI, a calm, friendly voice companion. Reply in the user's language, "
                "using 1–4 short spoken sentences. For questions about what is currently visible, use "
                "only latest_visual_observation. For personal-history questions, use only the supplied "
                "saved memory notes. Never invent memories, identify a person from the camera, diagnose "
                "a condition, or claim certainty the evidence does not support. If the context does not "
                "contain the answer, say you do not have that information yet and suggest what note could "
                "be saved. General greetings and simple general questions are okay. Do not claim to have "
                "saved anything unless the app confirms it."
            ),
        },
        {
            "role": "user",
            "content": (
                "Context data (may be empty):\n"
                + json.dumps(context, ensure_ascii=False)
                + "\n\nSpoken question: "
                + question
            ),
        },
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
