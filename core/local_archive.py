"""Local-only archive for Smriti/VisionQuest observations and consented media.

No archive files are uploaded. Configure SMRITI_ARCHIVE_DIR to choose an external
folder (for example D:\\Smriti_AI_Archive). If that drive is unavailable, use a
project-local data/archive folder and inspect the reported path in the app.
"""
from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import re
import threading
import uuid
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
_MAX_MEDIA_BYTES = 60 * 1024 * 1024
_ARCHIVE_LOCK = threading.RLock()


def get_archive_dir() -> Path:
    """Return a writable archive root, preferring the configured external drive."""
    configured = os.getenv("SMRITI_ARCHIVE_DIR", "").strip()
    candidates: list[Path] = []
    if configured:
        candidates.append(Path(configured).expanduser())
    elif os.name == "nt" and Path("D:/").exists():
        candidates.append(Path("D:/Smriti_AI_Archive"))
    candidates.append(ROOT / "data" / "archive")
    last_error: Exception | None = None
    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            probe = candidate / ".write_test"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
            return candidate
        except OSError as exc:
            last_error = exc
    raise OSError(f"Could not find a writable Smriti archive directory: {last_error}")


def _clean_label(value: Any, limit: int = 80) -> str:
    """Keep human-entered tags readable while preventing control characters/path use."""
    return re.sub(r"[\x00-\x1f\x7f]", " ", str(value or "")).strip()[:limit]


def clean_tags(tags: dict[str, Any] | None = None) -> dict[str, str]:
    """Normalize manual people/place/context labels for activity records."""
    tags = tags or {}
    return {
        key: value
        for key, raw in (
            ("people", tags.get("people", "")),
            ("place", tags.get("place", "")),
            ("activity", tags.get("activity", "")),
        )
        if (value := _clean_label(raw, 160))
    }


def archive_event(event_type: str, payload: dict[str, Any], tags: dict[str, Any] | None = None) -> Path:
    """Append one JSONL event to the local activity archive."""
    root = get_archive_dir()
    target = root / "activity.jsonl"
    entry = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "type": _clean_label(event_type, 40) or "activity",
        "tags": clean_tags(tags),
        **payload,
    }
    encoded = json.dumps(entry, ensure_ascii=False, separators=(",", ":"), default=str)
    with _ARCHIVE_LOCK:
        with target.open("a", encoding="utf-8", newline="") as handle:
            handle.write(encoded + "\n")
    return target


def save_media(
    kind: str,
    data: bytes,
    extension: str,
    mime_type: str = "application/octet-stream",
    tags: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Atomically save an explicitly captured media clip/snapshot under a local archive."""
    media = bytes(data or b"")
    if not media:
        raise ValueError("The media recording is empty.")
    if len(media) > _MAX_MEDIA_BYTES:
        raise ValueError("This media file exceeds the 60 MB archive limit.")
    safe_kind = _clean_label(kind, 30).lower()
    if safe_kind not in {"audio", "video", "frames", "exports"}:
        raise ValueError("Unsupported archive media category.")
    safe_ext = _clean_label(extension, 10).lower().lstrip(".")
    if safe_ext not in {"webm", "wav", "mp3", "m4a", "mp4", "jpg", "jpeg", "png", "json", "csv"}:
        raise ValueError("Unsupported archive file type.")
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    name = f"{stamp}_{uuid.uuid4().hex[:8]}.{safe_ext}"
    folder = get_archive_dir() / safe_kind
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / name
    temp = folder / (name + ".part")
    with _ARCHIVE_LOCK:
        committed = False
        try:
            with temp.open("wb") as handle:
                handle.write(media)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp, target)
            committed = True
            log = get_archive_dir() / "activity.jsonl"
            entry = {
                "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
                "type": "media_saved",
                "media_kind": safe_kind,
                "file": str(target),
                "mime_type": _clean_label(mime_type, 100),
                "bytes": len(media),
                "tags": clean_tags(tags),
                "metadata": metadata or {},
            }
            with log.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        except Exception:
            # Do not leave an apparently saved media file behind if its index entry
            # cannot be written; callers can safely retry without duplicate clips.
            if committed:
                target.unlink(missing_ok=True)
            raise
        finally:
            temp.unlink(missing_ok=True)
    return target


def write_text_export(name: str, content: str) -> Path:
    """Write a bounded plain-text/JSONL export into the archive exports folder."""
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name or "export"))[:100]
    if not safe_name:
        safe_name = "export.txt"
    if len(content.encode("utf-8")) > 10 * 1024 * 1024:
        raise ValueError("Text export exceeds the 10 MB limit.")
    folder = get_archive_dir() / "exports"
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / safe_name
    with _ARCHIVE_LOCK:
        target.write_text(content, encoding="utf-8")
    return target
