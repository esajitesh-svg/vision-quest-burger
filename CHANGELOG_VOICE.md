# Voice module changelog

## Original browser text-to-speech module

The first voice module added:

- Browser-native speech synthesis in `core/voice.py` and the **Voice** tab for reading current observations, selected saved memory notes, or entered text.
- Language choices for English (India), English (United States), English (United Kingdom), Hindi, and Tamil, plus speaking-speed controls.
- Text normalization for common Markdown and links.
- No server-side text-to-speech dependency, separate TTS API key, or generated audio file.
- Protection against accidentally saving an earlier image result as the current image's observation.

## Current live JARVIS companion

The current cloud application extends the original read-aloud feature with:

- Browser-native `SpeechRecognition` opt-in after **ACTIVATE LISTENING**, with visible microphone readiness and recognition error states.
- Asynchronous AI replies grounded in recent camera observations, a fresh frame when available, recent conversation history, and relevant saved memory/activity notes.
- Automatic reply speech with recovery retry, plus **REPLAY LAST** and **SPEAK ANSWER** actions.
- A typed-question fallback (**Or type a question for Smriti** / **SEND QUESTION TO SMRITI**) that uses the same AI response worker when speech recognition is unsupported or not working.
- Visible AI/provider errors, including missing/invalid Hugging Face permissions, provider credit/rate limits, and model availability failures where the provider returns a useful error.
- Text-only conversation and observation logging; raw live microphone audio and continuous camera frames are not stored by the live JARVIS feature.

## Limitations

- Speech recognition support varies by browser and the browser's speech service may process audio externally; current Chrome/Edge are the intended targets.
- Speech output uses voices provided by the visitor's browser/device, so availability, pronunciation, and latency vary.
- AI replies and scene explanations require a working `HF_TOKEN` with Inference Providers permission and depend on model/provider/network availability and usage credits.
- The camera is optional for general questions; live visual answers require an active camera frame. A typed answer can still fail if the hosted AI provider or configuration is unavailable.
- This is a shared-password cloud prototype, not individually isolated storage or a clinically validated medical system. Do not enter sensitive medical information or store another person's portrait without permission.
