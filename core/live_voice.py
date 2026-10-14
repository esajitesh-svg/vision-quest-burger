"""Thread-safe push-to-talk-by-pause state for the live VisionQuest mic stream.

PCM is retained only in memory while a short utterance is assembled. It is
converted to a WAV clip after speech pauses; no background audio file is kept.
"""
from __future__ import annotations

import io
import time
import wave
from threading import Lock

import numpy as np

SAMPLE_RATE = 16000
SAMPLE_WIDTH_BYTES = 2
PRE_ROLL_SECONDS = 0.18
SILENCE_END_SECONDS = 0.75
MIN_SPEECH_SECONDS = 0.30
MAX_UTTERANCE_SECONDS = 12.0
ENERGY_THRESHOLD = 420.0


class LiveVoiceAssistantState:
    """Collect one utterance at a time from the WebRTC audio callback."""

    def __init__(self) -> None:
        self.lock = Lock()
        self._pre_roll = bytearray()
        self._buffer = bytearray()
        self._speech_active = False
        self._speech_started_at = 0.0
        self._last_voice_at = 0.0
        self._voiced_samples = 0

        self._processing = False
        self._status = "Microphone off"
        self._error = ""
        self._latest_transcript = ""
        self._latest_reply = ""
        self._latest_reply_sequence = 0
        self._latest_reply_at = 0.0
        self._latest_used_live_camera = False
        self._history: list[dict[str, str]] = []
        self._dropped_turns = 0

    def set_streaming(self, active: bool) -> None:
        with self.lock:
            if active:
                if not self._processing and not self._speech_active:
                    self._status = "Microphone active — waiting for speech"
            else:
                self._pre_roll.clear()
                self._buffer.clear()
                self._speech_active = False
                self._voiced_samples = 0
                if not self._processing:
                    self._status = "Microphone off"

    def ingest_pcm(self, samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes | None:
        """Consume one mono int16 chunk; return WAV-ready PCM when an utterance ends."""
        rate = int(sample_rate)
        if rate != SAMPLE_RATE:
            raise ValueError(f"Audio must be resampled to {SAMPLE_RATE} Hz before ingestion.")
        pcm = np.asarray(samples, dtype=np.int16).reshape(-1)
        if pcm.size == 0:
            return None

        raw = pcm.tobytes()
        rms = float(np.sqrt(np.mean(pcm.astype(np.float32) ** 2)))
        now = time.monotonic()
        pre_roll_limit = int(rate * PRE_ROLL_SECONDS) * SAMPLE_WIDTH_BYTES

        with self.lock:
            if not self._speech_active:
                self._pre_roll.extend(raw)
                if len(self._pre_roll) > pre_roll_limit:
                    del self._pre_roll[:-pre_roll_limit]
                if rms < ENERGY_THRESHOLD:
                    return None
                self._speech_active = True
                self._speech_started_at = now
                self._last_voice_at = now
                self._buffer = bytearray(self._pre_roll)
                self._buffer.extend(raw)
                self._voiced_samples = int(pcm.size)
                self._status = "Speech detected — listening for the end of your sentence"
                return None

            self._buffer.extend(raw)
            if rms >= ENERGY_THRESHOLD:
                self._last_voice_at = now
                self._voiced_samples += int(pcm.size)

            utterance_seconds = len(self._buffer) / (rate * SAMPLE_WIDTH_BYTES)
            silence_seconds = now - self._last_voice_at
            should_finish = (
                silence_seconds >= SILENCE_END_SECONDS
                and self._voiced_samples >= int(rate * MIN_SPEECH_SECONDS)
            ) or utterance_seconds >= MAX_UTTERANCE_SECONDS

            if not should_finish:
                return None

            result = bytes(self._buffer)
            self._pre_roll = bytearray()
            self._buffer = bytearray()
            self._speech_active = False
            self._voiced_samples = 0
            self._status = "Speech captured — preparing a reply"
            return result

    def begin_processing(self) -> bool:
        with self.lock:
            if self._processing:
                self._dropped_turns += 1
                self._status = "Assistant is busy — pause and ask again after the reply"
                return False
            self._processing = True
            self._error = ""
            self._status = "Transcribing and thinking…"
            return True

    def finish_turn(
        self,
        transcript: str,
        reply: str,
        used_live_camera: bool = False,
    ) -> None:
        with self.lock:
            self._latest_transcript = str(transcript or "")[:1200]
            self._latest_reply = str(reply or "")[:1800]
            self._latest_used_live_camera = bool(used_live_camera)
            self._latest_reply_sequence += 1
            self._latest_reply_at = time.monotonic()
            self._history.append({
                "question": self._latest_transcript,
                "reply": self._latest_reply,
            })
            self._history = self._history[-10:]
            self._processing = False
            self._error = ""
            self._status = "Reply ready — speak again after it finishes"

    def fail_turn(self, error: str) -> None:
        with self.lock:
            self._processing = False
            self._error = str(error or "Unknown voice error")[:500]
            self._status = "Voice assistant needs attention"

    def history_snapshot(self) -> list[dict[str, str]]:
        with self.lock:
            return [dict(turn) for turn in self._history[-4:]]

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "processing": self._processing,
                "status": self._status,
                "error": self._error,
                "latest_transcript": self._latest_transcript,
                "latest_reply": self._latest_reply,
                "reply_sequence": self._latest_reply_sequence,
                "reply_age_seconds": (
                    max(0.0, time.monotonic() - self._latest_reply_at)
                    if self._latest_reply_at else None
                ),
                "used_live_camera": self._latest_used_live_camera,
                "history": [dict(turn) for turn in self._history],
                "dropped_turns": self._dropped_turns,
                "speech_active": self._speech_active,
            }


def pcm16_to_wav(pcm_bytes: bytes, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Wrap mono signed 16-bit PCM bytes as a WAV clip for ASR."""
    if not pcm_bytes:
        raise ValueError("Cannot create a WAV file from empty audio.")
    if len(pcm_bytes) % SAMPLE_WIDTH_BYTES:
        raise ValueError("PCM clip is not aligned to 16-bit samples.")
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(SAMPLE_WIDTH_BYTES)
        wav.setframerate(int(sample_rate))
        wav.writeframes(pcm_bytes)
    return output.getvalue()
