# Simple Smriti AI interface

This branch adds a smaller single-screen UI in `simple_app.py`. It uses the existing VisionQuest voice, vision, and memory services.

## Run on Windows

1. Install Python 3.10 or newer and Git.
2. Clone the repository and check out this branch:
   ```powershell
   git clone https://github.com/esajitesh-svg/vision-quest-burger.git
   cd vision-quest-burger
   git switch simple-smriti-ui
   ```
3. Configure `HF_TOKEN` in the environment or in Streamlit Secrets. Do not paste a token into source code or commit it.
4. Run `run_simple_smriti.bat`, or run `python -m streamlit run simple_app.py` from the activated virtual environment.
5. Open the local URL shown in the terminal. Allow camera and microphone permissions.

## Included behavior

- Animated companion orb and a compact single-screen layout.
- Live browser camera feed through WebRTC.
- Button-triggered visual analysis using the configured vision model.
- Browser speech recognition after pressing **ACTIVATE LISTENING**, with typed questions as fallback.
- Replies use current camera-frame context when available, recent saved activity memories, and recent conversation turns.
- Browser speech synthesis reads the reply; the component pauses recognition while speaking and resumes the listening cycle.
- Status/errors are shown in the interface; completed observations and conversations are saved when database storage is available.

## Important limits

This is a simplified prototype, not a medical device or a guaranteed always-on background listener. The browser must remain open, permissions must be granted, and the configured Hugging Face services/network must work. Speech recognition availability varies by browser. Camera feed and analysis are separate: analysis runs when requested. The analysis panel shows actual status and observations, not hidden private model reasoning.

The app has not been verified on every browser/device. Test camera, voice recognition, AI response, speech playback, and persistence in the deployment environment before relying on it.
