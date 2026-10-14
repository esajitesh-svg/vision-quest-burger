"""Local Granite vision/chat integration through the Ollama HTTP API."""
from __future__ import annotations

import base64
import os
import re
import time
from dataclasses import dataclass
from typing import Any

import requests

GRANITE_MODEL = os.getenv("SMRITI_GRANITE_MODEL", "granite3.2-vision:2b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
REQUEST_TIMEOUT = (3.05, 90)


@dataclass(frozen=True)
class GraniteResult:
    text: str
    model: str
    elapsed_seconds: float


def clean_granite_reply(value: str) -> str:
    """Suppress common reasoning wrappers and return direct user-facing text."""
    text = str(value or "").strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
    final = re.search(r"(?:final\s+answer|final)\s*:\s*(.+)$", text, flags=re.IGNORECASE | re.DOTALL)
    if final:
        text = final.group(1).strip()
    elif re.match(r"^(?:thinking(?:\.\.\.|\s+process)?\s*:?|analysis\s*:)", text, flags=re.IGNORECASE):
        return "I couldn't form a concise answer just now. Please ask me again."
    text = re.sub(r"^\s*(?:assistant|smriti)\s*:\s*", "", text, flags=re.IGNORECASE)
    return text.strip() or "I couldn't form an answer just now. Please ask me again."


def local_granite_status() -> tuple[bool, str]:
    """Check that local Ollama is reachable and the selected Granite model is installed."""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=(2, 5))
        response.raise_for_status()
        payload = response.json()
        names = {
            str(item.get("name", "")).strip()
            for item in payload.get("models", [])
            if isinstance(item, dict)
        }
        if GRANITE_MODEL not in names:
            return False, (
                f"Ollama is running, but {GRANITE_MODEL} is not listed. "
                "Check ollama list and SMRITI_GRANITE_MODEL."
            )
        return True, f"Local Granite ready: {GRANITE_MODEL}."
    except requests.RequestException:
        return False, f"Cannot reach local Ollama at {OLLAMA_BASE_URL}. Start Ollama and check ollama list."
    except (ValueError, TypeError):
        return False, "Ollama returned an unexpected model-list response."


def _chat(
    prompt: str,
    *,
    system: str,
    image_bytes: bytes | None = None,
    num_predict: int = 180,
) -> GraniteResult:
    prompt = (prompt or "").strip()
    if not prompt:
        raise ValueError("Please enter a question.")
    message: dict[str, Any] = {"role": "user", "content": prompt}
    if image_bytes:
        if len(image_bytes) > 8 * 1024 * 1024:
            raise ValueError("The camera image exceeds the 8 MB local request limit.")
        message["images"] = [base64.b64encode(image_bytes).decode("ascii")]

    payload = {
        "model": GRANITE_MODEL,
        "stream": False,
        "messages": [
            {
                "role": "system",
                "content": (
                    system
                    + " Give the user only the direct answer. Never print private reasoning, "
                    "a thinking process, hidden analysis, or deliberation. Keep responses concise."
                ),
            },
            message,
        ],
        "options": {
            "temperature": 0.2,
            "num_ctx": 2048,
            "num_predict": max(32, min(int(num_predict), 500)),
        },
        "keep_alive": "10m",
    }
    started = time.perf_counter()
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
    except requests.ConnectionError as exc:
        raise RuntimeError(
            "Local Ollama is not responding. Open Ollama and verify ollama list works in PowerShell."
        ) from exc
    except requests.Timeout as exc:
        raise RuntimeError(
            "Granite took too long to respond. Close other memory-heavy apps and retry with a shorter question."
        ) from exc
    except requests.RequestException as exc:
        raise RuntimeError(f"Could not call local Ollama ({type(exc).__name__}).") from exc

    if response.status_code == 404:
        raise RuntimeError(f"Ollama cannot find model {GRANITE_MODEL}. Check ollama list.")
    if not response.ok:
        detail = response.text[:240] if response.text else "No details supplied."
        raise RuntimeError(f"Ollama returned HTTP {response.status_code}: {detail}")
    try:
        data = response.json()
        raw = str(data.get("message", {}).get("content", "")).strip()
    except (ValueError, TypeError, AttributeError) as exc:
        raise RuntimeError("Ollama returned an invalid chat response.") from exc

    return GraniteResult(
        text=clean_granite_reply(raw),
        model=GRANITE_MODEL,
        elapsed_seconds=round(max(0.001, time.perf_counter() - started), 3),
    )


def analyze_with_granite(
    image_bytes: bytes,
    mode: str = "auto_observe",
    max_side: int = 768,
) -> GraniteResult:
    """Describe a supplied image locally using the selected Granite vision model."""
    del max_side  # The UI already caps encoded camera images to a reasonable size.
    tasks = {
        "auto_observe": (
            "Describe the visible scene, distinct objects and their positions, visible actions, readable text, "
            "and uncertainties. Be specific but concise. Only report what this single image supports."
        ),
        "scene": "Describe this image in 2 to 4 concise sentences using visible evidence only.",
        "objects": "List clearly visible objects and their positions. Do not guess obscured objects.",
        "read_text": "Read only text clearly visible in the image. Mark uncertain portions as unclear.",
    }
    return _chat(
        tasks.get(mode, tasks["auto_observe"]),
        system=(
            "You are Smriti, a visual memory companion. Do not identify people by face, infer personal "
            "relationships, guess emotions or intent, or invent off-screen details."
        ),
        image_bytes=bytes(image_bytes or b""),
        num_predict=260 if mode == "auto_observe" else 160,
    )


def ask_granite(
    question: str,
    *,
    latest_observation: dict[str, Any] | None = None,
    memories: list[dict[str, Any]] | None = None,
    conversation_history: list[dict[str, str]] | None = None,
    image_bytes: bytes | None = None,
) -> str:
    """Answer locally, using optional observation, memories, conversation, and current image."""
    context_parts: list[str] = []
    if latest_observation and latest_observation.get("text"):
        context_parts.append(
            "LATEST OBSERVATION (may be a sample; check its source): "
            + str(latest_observation.get("text", ""))[:1800]
        )
    if memories:
        rows = []
        for row in memories[:5]:
            note = row.get("note") or row.get("text") or row.get("memory") or ""
            if note:
                rows.append(str(note)[:350])
        if rows:
            context_parts.append("RELEVANT SAVED MEMORIES:\n- " + "\n- ".join(rows))
    if conversation_history:
        turns = []
        for turn in conversation_history[-3:]:
            q = str(turn.get("question", ""))[:180]
            a = str(turn.get("reply", ""))[:240]
            if q and a:
                turns.append(f"User: {q}\nSmriti: {a}")
        if turns:
            context_parts.append("RECENT CONVERSATION:\n" + "\n".join(turns))
    context = "\n\n".join(context_parts)
    prompt = (f"Useful context:\n{context}\n\n" if context else "") + (
        "Answer the user's question in 1–3 short sentences. If evidence is insufficient, say so clearly.\n"
        f"User question: {str(question or '').strip()}"
    )
    result = _chat(
        prompt,
        system="You are Smriti, a clear, kind, practical memory companion.",
        image_bytes=image_bytes,
        num_predict=150,
    )
    return result.text
