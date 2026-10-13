# Voice module update

## Added

- New `core/voice.py` module for browser-native speech synthesis.
- New **Voice** tab to load the latest visual observation, select a saved memory note, or enter text directly.
- Read-aloud controls on the latest current-image observation.
- Language choices: English (India), English (United States), English (United Kingdom), Hindi, and Tamil.
- Speaking-speed control plus Speak and Stop controls.
- Speech text normalization to remove common Markdown formatting and links.
- Safer inline script encoding to prevent text from escaping the speech widget's script context.
- No new server-side TTS dependency, extra API key, or generated audio file.

## Fixed

- The Vision tab now hides the save form for a previous image result when a different image is selected, preventing an old observation from being accidentally saved as the current image's result.
- Memory loading controls are placed together in the Voice tab.

## Limitations

- Uses voices installed/provided by the visitor's browser and device; voice availability and pronunciation vary.
- This version implements text-to-speech, not speech-to-text microphone recognition.
- This remains a shared-password cloud demo and is not validated for medical use.
