"""Browser-native text-to-speech helpers for VisionQuest Cloud."""
from __future__ import annotations

import html
import json
import re

MAX_SPEECH_CHARS = 4000
SUPPORTED_LANGUAGES: dict[str, str] = {
    "en-IN": "English (India)",
    "en-US": "English (United States)",
    "en-GB": "English (United Kingdom)",
    "hi-IN": "Hindi",
    "ta-IN": "Tamil",
}


def normalize_speech_text(text: str, max_chars: int = MAX_SPEECH_CHARS) -> str:
    """Remove common Markdown/link formatting and cap text sent to speech."""
    value = html.unescape(str(text or ""))
    value = re.sub(r"\x60{3}.*?\x60{3}", " code block omitted. ", value, flags=re.S)
    value = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", value)
    value = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", value)
    value = re.sub(r"https?://\S+", " link omitted. ", value)
    value = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", value)
    value = re.sub(r"(?m)^\s*[-*+]\s+", "", value)
    value = re.sub(r"(?m)^\s*\d+[.)]\s+", "", value)
    value = re.sub(r"[*_\x60~]+", "", value)
    value = re.sub(r"\s+", " ", value).strip()
    if not value:
        raise ValueError("Enter some text before using read aloud.")
    if max_chars < 1:
        raise ValueError("Maximum speech length must be positive.")
    if len(value) > max_chars:
        value = value[:max_chars].rsplit(" ", 1)[0].rstrip(" ,;:") + "… Text shortened."
    return value


def _safe_js_string(value: str) -> str:
    """JSON-encode text and escape HTML-sensitive characters for inline scripts."""
    return (
        json.dumps(value, ensure_ascii=False)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def build_speech_html(
    text: str,
    language: str = "en-IN",
    rate: float = 0.92,
    widget_label: str = "Read aloud",
) -> str:
    """Return a self-contained safe HTML player using browser speech synthesis."""
    spoken_text = normalize_speech_text(text)
    selected_language = language if language in SUPPORTED_LANGUAGES else "en-IN"
    selected_rate = min(1.25, max(0.65, float(rate)))
    text_js = _safe_js_string(spoken_text)
    lang_js = _safe_js_string(selected_language)
    label_html = html.escape(widget_label[:80] or "Read aloud")
    return f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root {{ color-scheme: light; font-family: system-ui, -apple-system, Segoe UI, sans-serif; }}
body {{ margin:0; padding:4px 2px 6px; background:transparent; color:#26314e; }}
.controls {{ display:flex; align-items:center; gap:8px; flex-wrap:wrap; }}
button {{ border:1px solid #d7ddef; border-radius:10px; padding:9px 13px; min-height:39px; cursor:pointer; font-weight:650; font-size:14px; }}
#speak {{ background:#594bd6; color:#fff; border-color:#594bd6; }}
#stop {{ background:#fff; color:#26314e; }}
button:focus-visible,input:focus-visible {{ outline:3px solid #9389ff; outline-offset:2px; }}
.rate {{ font-size:12px; color:#52617d; display:flex; align-items:center; gap:6px; }}
input[type=range] {{ width:95px; accent-color:#594bd6; }}
#status {{ margin:6px 2px 0; font-size:12px; color:#596780; min-height:16px; }}
</style></head>
<body>
<div class="controls" aria-label="Voice controls">
  <button id="speak" type="button" aria-label="{label_html}">🔊 {label_html}</button>
  <button id="stop" type="button" aria-label="Stop speaking">■ Stop</button>
  <label class="rate" for="rate">Speed <input id="rate" type="range" min="0.65" max="1.25" step="0.05" value="{selected_rate:.2f}"><span id="rateValue">{selected_rate:.2f}×</span></label>
</div>
<div id="status" role="status" aria-live="polite">Tap Read aloud to hear this text on your device.</div>
<script>
(() => {{
  const message = {text_js};
  const preferredLanguage = {lang_js};
  const speakButton = document.getElementById('speak');
  const stopButton = document.getElementById('stop');
  const rateInput = document.getElementById('rate');
  const rateValue = document.getElementById('rateValue');
  const status = document.getElementById('status');
  const synth = window.speechSynthesis;
  rateInput.addEventListener('input', () => {{ rateValue.textContent = Number(rateInput.value).toFixed(2) + '×'; }});
  function speak() {{
    if (!synth || typeof window.SpeechSynthesisUtterance === 'undefined') {{
      status.textContent = 'Read aloud is not supported in this browser. Try current Chrome or Edge.';
      return;
    }}
    synth.cancel();
    const utterance = new SpeechSynthesisUtterance(message);
    utterance.lang = preferredLanguage;
    utterance.rate = Number(rateInput.value);
    utterance.pitch = 1.0;
    const voices = synth.getVoices();
    const exactVoice = voices.find(v => v.lang && v.lang.toLowerCase() === preferredLanguage.toLowerCase());
    const languageVoice = voices.find(v => v.lang && v.lang.toLowerCase().startsWith(preferredLanguage.split('-')[0].toLowerCase()));
    if (exactVoice || languageVoice) utterance.voice = exactVoice || languageVoice;
    utterance.onstart = () => {{ status.textContent = 'Speaking on this device…'; }};
    utterance.onend = () => {{ status.textContent = 'Finished reading.'; }};
    utterance.onerror = (event) => {{ status.textContent = event.error === 'canceled' ? 'Speech stopped.' : 'Could not play speech. Try another browser voice or device.'; }};
    status.textContent = 'Preparing voice…';
    synth.speak(utterance);
  }}
  speakButton.addEventListener('click', speak);
  stopButton.addEventListener('click', () => {{
    if (synth) synth.cancel();
    status.textContent = 'Speech stopped.';
  }});
}})();
</script>
</body></html>"""
