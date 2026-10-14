"""Focused tests for ranked saved-memory retrieval."""
from __future__ import annotations

import core.cloud_db as db


def test_search_memories_ranks_person_and_note_matches(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "memories.sqlite3")
    db.init_db()

    db.save_person("Maya", "Aunt", "Lives in Chennai")
    maya = next(person for person in db.list_people() if person["name"] == "Maya")
    db.save_memory(maya["id"], "Maya enjoys gardening and grows jasmine plants.")
    db.save_memory(maya["id"], "Her birthday is in March.")

    db.save_person("Arun", "Friend", "School robotics teammate")
    arun = next(person for person in db.list_people() if person["name"] == "Arun")
    db.save_memory(arun["id"], "We built a small rover for the science fair.")

    results = db.search_memories("what do you remember about my aunt Maya?", limit=10)
    assert results
    assert all(row["person_name"] == "Maya" for row in results)
    assert results[0]["note"] == "Maya enjoys gardening and grows jasmine plants."


def test_search_memories_returns_empty_for_blank_or_unmatched_query(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "memories.sqlite3")
    db.init_db()
    db.save_person("Maya", "Aunt")
    person = db.list_people()[0]
    db.save_memory(person["id"], "Likes gardening.")

    assert db.search_memories("   ") == []
    assert db.search_memories("spaceship on Mars") == []


def test_search_memories_is_case_insensitive_and_respects_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "memories.sqlite3")
    db.init_db()
    db.save_person("Maya", "Aunt")
    person = db.list_people()[0]
    db.save_memory(person["id"], "Maya enjoys gardening.")
    db.save_memory(person["id"], "Maya grows roses.")

    results = db.search_memories("GARDENING maya", limit=1)
    assert len(results) == 1
    assert results[0]["person_name"] == "Maya"
    assert "gardening" in results[0]["note"].casefold()
