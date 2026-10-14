"""Unit checks for the optional managed-PostgreSQL adapter."""
from __future__ import annotations

import core.cloud_db as db


class FakeConnection:
    def __init__(self):
        self.query = None
        self.params = None

    def execute(self, query, params):
        self.query = query
        self.params = params
        return "cursor"


def test_postgres_adapter_translates_placeholders():
    fake = FakeConnection()
    adapter = db._PostgresAdapter(fake)

    result = adapter.execute(
        "INSERT INTO memories(person_id, note, source) VALUES(?,?,?)",
        (7, "test note", "test"),
    )

    assert result == "cursor"
    assert fake.query == "INSERT INTO memories(person_id, note, source) VALUES(%s,%s,%s)"
    assert fake.params == (7, "test note", "test")


def test_database_url_supports_postgres_scheme_alias(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgres://example:secret@db.invalid:5432/smriti")
    assert db.get_database_url() == "postgresql://example:secret@db.invalid:5432/smriti"
    assert db.has_durable_database() is True


def test_database_url_is_not_required_for_local_sqlite(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert db.get_database_url() == ""
    assert db.has_durable_database() is False
