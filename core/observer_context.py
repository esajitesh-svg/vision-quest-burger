"""Small, testable helpers for passing real live observations into Q&A."""
from __future__ import annotations


def latest_observation_context(
    text: str | None,
    timestamp: str | None = None,
) -> dict[str, str] | None:
    """Return only a real completed observation, never a waiting/demo placeholder."""
    observation = str(text or "").strip()
    if not observation:
        return None
    folded = observation.casefold()
    if folded.startswith((
        "waiting for a live camera frame",
        "no scene analysis yet",
        "demo sample",
    )):
        return None
    return {
        "text": observation,
        "timestamp": str(timestamp or "Unknown"),
        "mode": "auto_observe",
        "source": "local-live-observer",
    }
