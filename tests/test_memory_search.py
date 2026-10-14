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

def test_activity_logs_persist_conversations_and_observations(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "activity.sqlite3")
    db.init_db()

    db.save_conversation_turn(
        "What did I just see?",
        "You looked at a red notebook on the desk.",
        source="Live JARVIS",
        used_live_camera=True,
    )
    db.save_observation(
        "A red notebook sits on the desk.",
        source="Live camera",
        model="test-vision-model",
        observed_at="2026-10-14 12:00 UTC",
        frame_hash="abc123",
    )

    turns = db.list_conversation_turns(limit=10)
    observations = db.list_observations(limit=10)
    assert turns[0]["user_text"] == "What did I just see?"
    assert turns[0]["assistant_text"] == "You looked at a red notebook on the desk."
    assert turns[0]["used_live_camera"] in (1, True)
    assert observations[0]["note"] == "A red notebook sits on the desk."
    assert observations[0]["frame_hash"] == "abc123"


def test_voice_activity_retrieval_finds_past_observations_and_conversation(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "activity.sqlite3")
    db.init_db()
    db.save_conversation_turn(
        "Where did I leave my notebook?",
        "Your saved observation says the red notebook was on the desk.",
    )
    db.save_observation(
        "A red notebook sits on the desk.",
        source="Live camera",
        model="test-model",
    )

    results = db.search_activity_for_voice("remember the red notebook", limit=10)
    rendered = " ".join(row["note"] for row in results).casefold()
    assert "red notebook" in rendered
    assert "desk" in rendered
    assert any(row["person_name"] == "Past conversation" for row in results)
    assert any(row["person_name"] == "Past observation" for row in results)


def test_portrait_store_requires_valid_image_and_respects_profile(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "activity.sqlite3")
    db.init_db()
    db.save_person("Maya", "Aunt")
    person = db.list_people()[0]

    try:
        db.save_person_portrait(person["id"], b"not-an-image", "image/jpeg")
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid portrait data should be rejected")

    assert db.get_person_portrait(person["id"]) is None
\n