"""Regression tests for live-observation context used by conversational Q&A."""
from core.observer_context import latest_observation_context


def test_latest_observation_context_returns_completed_live_observation():
    result = latest_observation_context(
        "A blue notebook is on the table beside a closed laptop.",
        "21:42:03",
    )
    assert result == {
        "text": "A blue notebook is on the table beside a closed laptop.",
        "timestamp": "21:42:03",
        "mode": "auto_observe",
        "source": "local-live-observer",
    }


def test_latest_observation_context_ignores_startup_placeholder():
    assert latest_observation_context("Waiting for a live camera frame.", "Not run") is None
    assert latest_observation_context("No scene analysis yet. Start the camera.", "Not run") is None
    assert latest_observation_context("", "Not run") is None
    assert latest_observation_context(None, None) is None


def test_latest_observation_context_never_treats_demo_sample_as_live_memory():
    assert latest_observation_context("DEMO SAMPLE MEMORY: a fictional notebook.", "Now") is None
