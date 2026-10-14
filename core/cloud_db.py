"""Long-term memory store using managed PostgreSQL or development SQLite."""
from __future__ import annotations

import csv
import io
import os
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = Path(os.getenv("SMRITI_DB_PATH", str(ROOT / "data" / "visionquest_cloud.db")))


def get_database_url() -> str:
    """Read an optional managed PostgreSQL URL without printing or exposing it."""
    value = os.getenv("DATABASE_URL", "").strip()
    if not value:
        try:
            import streamlit as st
            value = str(st.secrets.get("DATABASE_URL", "")).strip()
        except Exception:
            value = ""
    if value.startswith("postgres://"):
        value = "postgresql://" + value[len("postgres://"):]
    return value


def has_durable_database() -> bool:
    """Managed PostgreSQL survives ordinary app-host filesystem resets."""
    return bool(get_database_url())


class _PostgresAdapter:
    """Tiny adapter so the existing parameterized SQL works with psycopg 3."""
    def __init__(self, connection):
        self._connection = connection

    def execute(self, query: str, params=()):
        return self._connection.execute(query.replace("?", "%s"), params or ())

    def commit(self):
        self._connection.commit()

    def rollback(self):
        self._connection.rollback()

    def close(self):
        self._connection.close()


@contextmanager
def connect():
    """Connect to managed PostgreSQL when configured, otherwise local SQLite."""
    database_url = get_database_url()
    if database_url:
        try:
            import psycopg
            from psycopg.rows import dict_row
        except ImportError as exc:
            raise RuntimeError(
                "DATABASE_URL is configured, but psycopg is missing. Install requirements.txt again."
            ) from exc
        raw = psycopg.connect(database_url, row_factory=dict_row, connect_timeout=10)
        conn = _PostgresAdapter(raw)
    else:
        raw = sqlite3.connect(DB_PATH, timeout=10)
        raw.row_factory = sqlite3.Row
        raw.execute("PRAGMA foreign_keys=ON")
        raw.execute("PRAGMA busy_timeout=10000")
        conn = raw
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def init_db():
    """Create all memory and activity tables, compatible with SQLite and PostgreSQL."""
    if get_database_url():
        people_sql = (
            "CREATE TABLE IF NOT EXISTS people ("
            "id BIGSERIAL PRIMARY KEY, "
            "name TEXT NOT NULL UNIQUE, "
            "relationship TEXT NOT NULL, notes TEXT NOT NULL DEFAULT '', "
            "created_at TEXT NOT NULL)"
        )
        memories_sql = (
            "CREATE TABLE IF NOT EXISTS memories ("
            "id BIGSERIAL PRIMARY KEY, person_id BIGINT NOT NULL, "
            "created_at TEXT NOT NULL, note TEXT NOT NULL, "
            "source TEXT NOT NULL DEFAULT 'Manual', "
            "FOREIGN KEY(person_id) REFERENCES people(id) ON DELETE CASCADE)"
        )
        portrait_type = "BYTEA"
        id_type = "BIGSERIAL PRIMARY KEY"
        bool_type = "BOOLEAN NOT NULL DEFAULT FALSE"
    else:
        people_sql = (
            "CREATE TABLE IF NOT EXISTS people ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "name TEXT NOT NULL COLLATE NOCASE UNIQUE, "
            "relationship TEXT NOT NULL, notes TEXT NOT NULL DEFAULT '', "
            "created_at TEXT NOT NULL)"
        )
        memories_sql = (
            "CREATE TABLE IF NOT EXISTS memories ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, person_id INTEGER NOT NULL, "
            "created_at TEXT NOT NULL, note TEXT NOT NULL, "
            "source TEXT NOT NULL DEFAULT 'Manual', "
            "FOREIGN KEY(person_id) REFERENCES people(id) ON DELETE CASCADE)"
        )
        portrait_type = "BLOB"
        id_type = "INTEGER PRIMARY KEY AUTOINCREMENT"
        bool_type = "INTEGER NOT NULL DEFAULT 0"

    with connect() as conn:
        conn.execute(people_sql)
        conn.execute(memories_sql)
        conn.execute(
            "CREATE TABLE IF NOT EXISTS conversation_turns ("
            f"id {id_type}, created_at TEXT NOT NULL, source TEXT NOT NULL, "
            "user_text TEXT NOT NULL, assistant_text TEXT NOT NULL, "
            f"used_live_camera {bool_type})"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS observations ("
            f"id {id_type}, created_at TEXT NOT NULL, observed_at TEXT NOT NULL, "
            "source TEXT NOT NULL, model TEXT NOT NULL, note TEXT NOT NULL, "
            "frame_hash TEXT NOT NULL DEFAULT '')"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS person_portraits ("
            + ("person_id BIGINT PRIMARY KEY, " if get_database_url() else "person_id INTEGER PRIMARY KEY, ")
            + f"content_type TEXT NOT NULL, image_data {portrait_type} NOT NULL, updated_at TEXT NOT NULL, "
            + "FOREIGN KEY(person_id) REFERENCES people(id) ON DELETE CASCADE)"
        )

def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def save_person(name: str, relationship: str, notes: str = "") -> None:
    name = (name or "").strip()
    relationship = (relationship or "").strip()
    notes = (notes or "").strip()
    if not name or not relationship:
        raise ValueError("Name and relationship/label are required.")
    if len(name) > 100 or len(relationship) > 100 or len(notes) > 1000:
        raise ValueError("Name/label must be at most 100 characters and notes at most 1,000 characters.")
    try:
        with connect() as conn:
            conn.execute(
                "INSERT INTO people(name,relationship,notes,created_at) VALUES(?,?,?,?)",
                (name, relationship, notes, now()),
            )
    except sqlite3.IntegrityError as exc:
        raise ValueError("That display name already exists in this shared demo vault.") from exc


def list_people() -> list[dict]:
    with connect() as conn:
        rows = conn.execute("SELECT * FROM people ORDER BY name COLLATE NOCASE").fetchall()
        return [dict(row) for row in rows]


def save_memory(person_id: int, note: str, source: str = "Manual") -> None:
    note = (note or "").strip()
    source = (source or "Manual").strip()[:300]
    if not note:
        raise ValueError("Memory note cannot be empty.")
    if len(note) > 4000:
        raise ValueError("Memory note must be 4,000 characters or fewer.")
    with connect() as conn:
        exists = conn.execute("SELECT 1 FROM people WHERE id=?", (int(person_id),)).fetchone()
        if not exists:
            raise ValueError("Selected person no longer exists. Refresh the page and try again.")
        conn.execute(
            "INSERT INTO memories(person_id,created_at,note,source) VALUES(?,?,?,?)",
            (int(person_id), now(), note, source),
        )


def list_memories(limit: int = 500) -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT m.*,p.name AS person_name,p.relationship "
            "FROM memories m JOIN people p ON p.id=m.person_id "
            "ORDER BY m.id DESC LIMIT ?",
            (max(1, min(int(limit), 1000)),),
        ).fetchall()
        return [dict(row) for row in rows]



def save_conversation_turn(
    user_text: str,
    assistant_text: str,
    source: str = "Live JARVIS",
    used_live_camera: bool = False,
) -> None:
    """Persist a text conversation turn; raw microphone audio is never stored."""
    question = (user_text or "").strip()[:4000]
    answer = (assistant_text or "").strip()[:6000]
    if not question or not answer:
        raise ValueError("Both the user turn and assistant reply are required.")
    with connect() as conn:
        conn.execute(
            "INSERT INTO conversation_turns(created_at,source,user_text,assistant_text,used_live_camera) "
            "VALUES(?,?,?,?,?)",
            (now(), (source or "Live JARVIS")[:100], question, answer, bool(used_live_camera)),
        )


def list_conversation_turns(limit: int = 100) -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM conversation_turns ORDER BY id DESC LIMIT ?",
            (max(1, min(int(limit), 1000)),),
        ).fetchall()
        return [dict(row) for row in rows]


def save_observation(
    note: str,
    source: str = "Live camera",
    model: str = "",
    observed_at: str = "",
    frame_hash: str = "",
) -> None:
    """Persist a completed text observation, not a continuous video stream."""
    value = (note or "").strip()[:8000]
    if not value:
        return
    with connect() as conn:
        conn.execute(
            "INSERT INTO observations(created_at,observed_at,source,model,note,frame_hash) "
            "VALUES(?,?,?,?,?,?)",
            (
                now(),
                (observed_at or now())[:100],
                (source or "Live camera")[:100],
                (model or "")[:200],
                value,
                (frame_hash or "")[:128],
            ),
        )


def list_observations(limit: int = 100) -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM observations ORDER BY id DESC LIMIT ?",
            (max(1, min(int(limit), 1000)),),
        ).fetchall()
        return [dict(row) for row in rows]


def save_person_portrait(person_id: int, image_bytes: bytes, content_type: str) -> None:
    """Store a user-selected, consented reference photo for the named profile only."""
    raw = bytes(image_bytes or b"")
    mime = (content_type or "").lower().strip()
    if not raw or len(raw) > 3 * 1024 * 1024:
        raise ValueError("Choose a photo no larger than 3 MB.")
    if mime not in {"image/jpeg", "image/png", "image/webp"}:
        raise ValueError("Portrait must be a JPEG, PNG, or WEBP image.")
    try:
        image = Image.open(io.BytesIO(raw))
        fmt = (image.format or "").upper()
        if fmt not in {"JPEG", "PNG", "WEBP"}:
            raise ValueError("Portrait must be a JPEG, PNG, or WEBP image.")
        image.verify()
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("The selected file is not a valid image.") from exc

    person_id = int(person_id)
    with connect() as conn:
        exists = conn.execute("SELECT 1 FROM people WHERE id=?", (person_id,)).fetchone()
        if not exists:
            raise ValueError("Selected person does not exist.")
        if get_database_url():
            conn.execute(
                "INSERT INTO person_portraits(person_id,content_type,image_data,updated_at) "
                "VALUES(?,?,?,?) ON CONFLICT(person_id) DO UPDATE SET "
                "content_type=EXCLUDED.content_type,image_data=EXCLUDED.image_data,updated_at=EXCLUDED.updated_at",
                (person_id, mime, raw, now()),
            )
        else:
            conn.execute(
                "INSERT INTO person_portraits(person_id,content_type,image_data,updated_at) "
                "VALUES(?,?,?,?) ON CONFLICT(person_id) DO UPDATE SET "
                "content_type=excluded.content_type,image_data=excluded.image_data,updated_at=excluded.updated_at",
                (person_id, mime, sqlite3.Binary(raw), now()),
            )


def get_person_portrait(person_id: int) -> dict | None:
    with connect() as conn:
        row = conn.execute(
            "SELECT person_id,content_type,image_data,updated_at FROM person_portraits WHERE person_id=?",
            (int(person_id),),
        ).fetchone()
        return dict(row) if row else None


def delete_person_portrait(person_id: int) -> None:
    with connect() as conn:
        conn.execute("DELETE FROM person_portraits WHERE person_id=?", (int(person_id),))


def search_observations(query: str, limit: int = 50) -> list[dict]:
    """Search the complete observation history rather than only the newest page."""
    value = (query or "").strip()
    if not value:
        return []
    tokens = [
        token for token in re.findall(r"[\w]+", value.casefold(), flags=re.UNICODE)
        if len(token) > 1 and token not in _MEMORY_QUERY_STOPWORDS
    ][:8]
    if not tokens:
        return list_observations(limit) if _MEMORY_RECALL_CUES.search(value) else []

    terms = []
    params: list[str | int] = []
    for token in tokens:
        terms.append("(LOWER(note) LIKE LOWER(?) OR LOWER(source) LIKE LOWER(?) OR LOWER(model) LIKE LOWER(?))")
        pattern = "%" + token + "%"
        params.extend([pattern, pattern, pattern])
    params.append(max(1, min(int(limit), 200)))
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM observations WHERE " + " OR ".join(terms) + " ORDER BY id DESC LIMIT ?",
            tuple(params),
        ).fetchall()
    results = [dict(row) for row in rows]
    if not results and _MEMORY_RECALL_CUES.search(value):
        return list_observations(limit)
    return results


def search_conversation_history(query: str, limit: int = 50) -> list[dict]:
    """Search conversations across the full stored history."""
    value = (query or "").strip()
    if not value:
        return []
    tokens = [
        token for token in re.findall(r"[\w]+", value.casefold(), flags=re.UNICODE)
        if len(token) > 1 and token not in _MEMORY_QUERY_STOPWORDS
    ][:8]
    if not tokens:
        return list_conversation_turns(limit) if _MEMORY_RECALL_CUES.search(value) else []

    terms = []
    params: list[str | int] = []
    for token in tokens:
        terms.append("(LOWER(user_text) LIKE LOWER(?) OR LOWER(assistant_text) LIKE LOWER(?) OR LOWER(source) LIKE LOWER(?))")
        pattern = "%" + token + "%"
        params.extend([pattern, pattern, pattern])
    params.append(max(1, min(int(limit), 200)))
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM conversation_turns WHERE " + " OR ".join(terms) + " ORDER BY id DESC LIMIT ?",
            tuple(params),
        ).fetchall()
    results = [dict(row) for row in rows]
    if not results and _MEMORY_RECALL_CUES.search(value):
        return list_conversation_turns(limit)
    return results


def search_activity_for_voice(query: str, limit: int = 10) -> list[dict]:
    """Retrieve curated notes plus relevant prior conversations and observations."""
    value = (query or "").strip()
    bounded_limit = max(1, min(int(limit), 10))
    ranked: list[tuple[int, int, dict]] = []
    seen: set[tuple[str, str]] = set()

    for index, row in enumerate(search_memories_for_voice(value, limit=min(5, bounded_limit))):
        normalized = dict(row)
        normalized.setdefault("created_at", "")
        normalized.setdefault("source", "Saved memory")
        ranked.append((1000 - index, index, normalized))
        seen.add(("memory", str(row.get("id", ""))))

    tokens = {
        token for token in re.findall(r"[\w]+", value.casefold(), flags=re.UNICODE)
        if len(token) > 1 and token not in _MEMORY_QUERY_STOPWORDS
    }
    recall_cue = bool(_MEMORY_RECALL_CUES.search(value))

    candidates: list[tuple[int, int, dict]] = []
    for row in search_observations(value, limit=100):
        haystack = (str(row.get("note", "")) + " " + str(row.get("source", ""))).casefold()
        score = 5 if value.casefold() and value.casefold() in haystack else 0
        score += sum(1 for token in tokens if token in haystack)
        candidates.append((score, int(row.get("id", 0)), {
            "id": "observation-" + str(row.get("id", "")),
            "person_name": "Past observation",
            "relationship": str(row.get("source", "Camera")),
            "created_at": str(row.get("observed_at", row.get("created_at", ""))),
            "note": "Previously observed: " + str(row.get("note", ""))[:1200],
            "source": str(row.get("model", "Vision analysis")),
        }))

    for row in search_conversation_history(value, limit=100):
        haystack = (
            str(row.get("user_text", "")) + " " + str(row.get("assistant_text", ""))
        ).casefold()
        score = 5 if value.casefold() and value.casefold() in haystack else 0
        score += sum(1 for token in tokens if token in haystack)
        candidates.append((score, int(row.get("id", 0)), {
            "id": "conversation-" + str(row.get("id", "")),
            "person_name": "Past conversation",
            "relationship": str(row.get("source", "Voice")),
            "created_at": str(row.get("created_at", "")),
            "note": (
                "You said: " + str(row.get("user_text", ""))[:450]
                + " Smriti replied: " + str(row.get("assistant_text", ""))[:650]
            ),
            "source": "Conversation log",
        }))

    # Prefer explicit textual matches; for broad recall, newer episodic items still help.
    candidates.sort(key=lambda item: (-item[0], -item[1]))
    result = [row for _, _, row in ranked[:bounded_limit]]
    for _, _, row in candidates:
        if len(result) >= bounded_limit:
            break
        key = (str(row.get("person_name", "")), str(row.get("id", "")))
        if key not in seen:
            result.append(row)
            seen.add(key)
    return result


def export_activity_csv() -> bytes:
    """Export recent activity as a portable UTF-8 CSV backup."""
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(["record_type", "timestamp", "source", "user_text", "assistant_text", "observation", "model"])
    for row in list_conversation_turns(limit=1000):
        writer.writerow(["conversation", row.get("created_at", ""), row.get("source", ""), row.get("user_text", ""), row.get("assistant_text", ""), "", ""])
    for row in list_observations(limit=1000):
        writer.writerow(["observation", row.get("observed_at", row.get("created_at", "")), row.get("source", ""), "", "", row.get("note", ""), row.get("model", "")])
    return output.getvalue().encode("utf-8-sig")


_MEMORY_QUERY_STOPWORDS = {
    "a", "an", "and", "are", "about", "am", "any", "at", "be", "been",
    "can", "did", "do", "does", "for", "from", "had", "has", "have", "i",
    "in", "is", "it", "me", "my", "of", "on", "or", "please", "tell",
    "that", "the", "this", "to", "was", "were", "what", "when", "where",
    "which", "who", "why", "with", "you", "your", "remember", "recall",
    "smriti", "memory", "memories",
}


def search_memories(query: str, limit: int = 100) -> list[dict]:
    """Rank saved notes by meaningful query-token overlap, not only exact phrases.

    This is a lightweight local retrieval method (not semantic/embedding search).
    Person labels receive extra weight so questions like "what do I know about
    my aunt?" can find notes even when the sentence is phrased differently.
    """
    needle = (query or "").strip().casefold()
    if not needle:
        return []

    tokens = {
        token for token in re.findall(r"[\w]+", needle, flags=re.UNICODE)
        if len(token) > 1 and token not in _MEMORY_QUERY_STOPWORDS
    }
    rows = list_memories(limit=1000)
    ranked: list[tuple[int, int, dict]] = []
    for index, row in enumerate(rows):
        person_text = " ".join(
            str(row.get(key, "")) for key in ("person_name", "relationship")
        ).casefold()
        note_text = " ".join(
            str(row.get(key, "")) for key in ("note", "source")
        ).casefold()
        searchable = person_text + " " + note_text
        score = 5 if needle in searchable else 0
        for token in tokens:
            # Count both the person-label match and the note-content match. A
            # note that names the person and answers the query should outrank an
            # unrelated note attached to that same person.
            if token in person_text:
                score += 3
            if token in note_text:
                score += 1
        if score:
            ranked.append((score, index, row))

    ranked.sort(key=lambda item: (-item[0], item[1]))
    return [row for _, _, row in ranked[:max(1, min(int(limit), 500))]]



_MEMORY_RECALL_CUES = re.compile(
    r"\b(?:remember|recall|forgot|forget|memories|memory|remind|reminder)\b"
    r"|\bwhere\s+(?:did|have)\s+i\s+(?:put|leave|left|keep|kept|place)\b"
    r"|\bwhat\s+(?:did|have)\s+i\s+(?:see|look\s+at|hear|say|do|save|put|leave|plan|talk\s+about)\b"
    r"|\b(?:earlier|before|previously|last\s+time|what\s+happened)\b",
    flags=re.IGNORECASE,
)


def search_memories_for_voice(query: str, limit: int = 10) -> list[dict]:
    """Retrieve relevant notes and, for explicit recall requests, recent fallback notes.

    A broad phrase such as "I forgot" may not contain any words from a saved
    note. In that case, offer a small set of recent notes as retrieval context.
    This does not assert that any note is what the user forgot, and is not used
    for ordinary live-scene questions.
    """
    value = (query or "").strip()
    bounded_limit = max(1, min(int(limit), 10))
    relevant = search_memories(value, limit=bounded_limit)
    if not value or not _MEMORY_RECALL_CUES.search(value) or len(relevant) >= bounded_limit:
        return relevant

    seen_ids = {row.get("id") for row in relevant}
    for row in list_memories(limit=1000):
        row_id = row.get("id")
        if row_id not in seen_ids:
            relevant.append(row)
            seen_ids.add(row_id)
        if len(relevant) >= bounded_limit:
            break
    return relevant

def export_memories_csv() -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(["person_name", "relationship", "created_at", "note", "source"])
    for row in list_memories(limit=1000):
        values = [row["person_name"], row["relationship"], row["created_at"], row["note"], row["source"]]
        safe = [
            ("'" + value)
            if isinstance(value, str) and value.startswith(("=", "+", "-", "@", "\t", "\r"))
            else value
            for value in values
        ]
        writer.writerow(safe)
    return output.getvalue().encode("utf-8-sig")
