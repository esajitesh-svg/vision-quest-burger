"""SQLite demo vault shared by the cloud app's authenticated viewers."""
from __future__ import annotations

import csv
import io
import os
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = Path(os.getenv("SMRITI_DB_PATH", str(ROOT / "data" / "visionquest_cloud.db")))


@contextmanager
def connect():
    """Open a connection and guarantee commit/rollback plus close."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=10000")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    with connect() as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS people ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "name TEXT NOT NULL COLLATE NOCASE UNIQUE, "
            "relationship TEXT NOT NULL, notes TEXT NOT NULL DEFAULT '', "
            "created_at TEXT NOT NULL)"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS memories ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, person_id INTEGER NOT NULL, "
            "created_at TEXT NOT NULL, note TEXT NOT NULL, "
            "source TEXT NOT NULL DEFAULT 'Manual', "
            "FOREIGN KEY(person_id) REFERENCES people(id) ON DELETE CASCADE)"
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
        token for token in re.findall(r"[\\w]+", needle, flags=re.UNICODE)
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
            if token in person_text:
                score += 3
            elif token in note_text:
                score += 1
        if score:
            ranked.append((score, index, row))

    ranked.sort(key=lambda item: (-item[0], item[1]))
    return [row for _, _, row in ranked[:max(1, min(int(limit), 500))]]


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
