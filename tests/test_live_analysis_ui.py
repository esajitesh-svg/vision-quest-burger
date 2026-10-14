"""Regression checks for the live scene-analysis controls."""
from pathlib import Path


def test_live_ai_analysis_is_visible_and_enabled_by_default():
    source = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")
    control = source.index('live_auto_analyze = st.checkbox(')
    section = source[control:control + 900]

    assert '"Enable automatic AI scene analysis"' in section
    assert "value=True" in section
    assert 'key="live_auto_analysis_enabled_v2"' in section
    assert "Automatic analysis is enabled by default" in source


def test_manual_analysis_button_does_not_require_preview_jpeg():
    source = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")
    assert 'disabled=(not is_playing or frame_count <= 0 or live_state.get_status()["analysis_busy"])' in source
    assert "if live_auto_analyze" in source
