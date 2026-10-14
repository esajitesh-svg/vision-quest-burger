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

def test_voice_memory_recall_uses_recent_notes_for_broad_forget_prompt(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "memories.sqlite3")
    db.init_db()

    db.save_person("Maya", "Aunt")
    maya = next(person for person in db.list_people() if person["name"] == "Maya")
    db.save_memory(maya["id"], "Maya enjoys gardening and grows jasmine plants.")

    db.save_person("Arun", "Friend")
    arun = next(person for person in db.list_people() if person["name"] == "Arun")
    db.save_memory(arun["id"], "The robotics meeting is on Friday after school.")

    results = db.search_memories_for_voice("I forgot, please help me remember", limit=5)
    assert len(results) == 2
    assert {row["note"] for row in results} == {
        "Maya enjoys gardening and grows jasmine plants.",
        "The robotics meeting is on Friday after school.",
    }


def test_voice_memory_recall_keeps_relevant_matches_first_then_fills_recent(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "memories.sqlite3")
    db.init_db()

    db.save_person("Maya", "Aunt")
    maya = next(person for person in db.list_people() if person["name"] == "Maya")
    db.save_memory(maya["id"], "Maya's birthday is in March.")

    db.save_person("Arun", "Friend")
    arun = next(person for person in db.list_people() if person["name"] == "Arun")
    db.save_memory(arun["id"], "Robotics meeting is on Friday.")

    results = db.search_memories_for_voice("remember Maya's birthday", limit=2)
    assert len(results) == 2
    assert results[0]["note"] == "Maya's birthday is in March."
    assert len({row["id"] for row in results}) == len(results)


def test_voice_memory_recall_does_not_fallback_for_camera_questions(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "memories.sqlite3")
    db.init_db()

    db.save_person("Maya", "Aunt")
    maya = next(person for person in db.list_people() if person["name"] == "Maya")
    db.save_memory(maya["id"], "Maya enjoys gardening.")

    assert db.search_memories_for_voice("What can you see in front of me?", limit=10) == []

