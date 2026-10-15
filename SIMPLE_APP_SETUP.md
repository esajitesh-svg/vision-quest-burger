# Simple Smriti AI interface

This branch adds a smaller single-screen UI in `simple_app.py`. Live AI uses the local Ollama Granite vision model; browser speech recognition captures spoken turns and browser speech synthesis reads replies aloud.

## Run on Windows

1. Install Python 3.10 or newer and Git.
2. Clone the repository and check out this branch:
   ```powershell
   git clone https://github.com/esajitesh-svg/vision-quest-burger.git
   cd vision-quest-burger
   git switch simple-smriti-ui
   ```
3. Install Ollama for Windows from https://ollama.com/download/windows. Keep the pendrive containing your models connected, then set the model store for this PowerShell session and verify Granite is listed:
   ```powershell
   $env:OLLAMA_MODELS = "D:\.ollama\models"
   ollama list
   ```
   The `granite3.2-vision:2b` tag must be present.
4. Run `run_simple_smriti.bat`, or run `python -m streamlit run simple_app.py` from the activated virtual environment.
5. Open the local URL shown in the terminal. Allow camera and microphone permissions.

## Included behavior

- Animated companion orb and a compact single-screen layout.
- A global DEMO MODE / LIVE MODE selector. Demo Mode uses scripted replies and simulated scene text without making cloud AI requests; it is marked as sample content.
- Live browser camera feed through WebRTC.
- Live OpenCV analyser with selectable display operators, camera-quality metrics, and optional local YOLOX object detection. The Live Observer defaults to FAST 384 px, uses a concise low-token prompt, works in a background worker, and never queues stale/overlapping frames. Balanced 512 px and Higher detail 640 px are available.
- Consent-based local face enrollment using a chosen reference photo, plus comparison against current camera faces. Unmatched or uncertain faces are labelled Unknown.
- Browser speech recognition after pressing **ACTIVATE LISTENING**, with typed questions as fallback.
- Replies use current camera-frame context when available, recent saved activity memories, and recent conversation turns.
- Browser speech synthesis reads the reply; the component pauses recognition while speaking and resumes the listening cycle.
- Local activity archive: observations, manually tagged people/place/activity context, object detections, and conversation text are automatically appended to `activity.jsonl` when automatic saving is enabled. Completed observations can also save a JPEG snapshot.
- Media archive: user-started microphone clips (maximum 30 seconds / 8 MB) are saved as audio files; user-started MP4 clips stop after 60 seconds. Recording is visibly indicated and is never started automatically.
- People / places / activity fields tag the current records. They are manual labels; they do not imply that the AI identified a person or obtained GPS location.
- The archive defaults to `D:\\Smriti_AI_Archive` when the D: drive is available, or can be changed with `SMRITI_ARCHIVE_DIR`. It contains `activity.jsonl`, `frames`, `audio`, and `video`. Download the JSONL index from the app's archive settings.
- The OpenCV DNN object detector tries OpenCL when supported and falls back to CPU if the device/runtime fails. The UI reports OpenCL availability, browser WebGL/WebGPU capability, and the output of `ollama ps` where available.

## Important limits

This is a simplified prototype, not a medical device or a guaranteed always-on background listener. The browser and local Ollama service must remain available, and camera/microphone permissions must be granted. Browser speech recognition support and network requirements vary by browser. Voice operates as quick turn-by-turn speech recognition → local Granite response → browser speech playback, rather than simultaneous streaming audio-to-audio. Live Observer checks for a fresh frame every second, but local Granite inference may still take several seconds on an 8 GB CPU PC. FAST mode uses 384 px images and short replies to reduce delay; the WebRTC camera preview continues independently while scene text updates after each model response. It does not run vision inference on every camera frame or guarantee 30-FPS narration. The Live Mode connection to `127.0.0.1:11434` works when Streamlit is running on the same PC; it will not reach a user's computer from a cloud-hosted web server. The analysis panel shows actual status and observations, not hidden private model reasoning.

Face comparison uses an experimental local OpenCV LBPH matcher, is sensitive to lighting and pose, and can misidentify or fail to match people; confirm identity rather than relying on it for consequential decisions. Reference portraits should only be registered with permission. Demo replies and sample observations are not real AI results and should never be treated as stored personal memories.

WebGL/WebGPU are browser graphics/compute APIs; detecting them does not automatically move Python/OpenCV or Ollama inference onto the GPU. The local YOLOX detector's OpenCL path is attempted only when available, and it falls back to CPU. Ollama acceleration depends on its installed version, drivers, and device; verify the actual processor status with `ollama ps`. Do not assume GPU acceleration is active merely because WebGPU is available in the browser.

The app has not been verified on every browser/device. Test camera, voice recognition, AI response, speech playback, face matching, and persistence in the deployment environment before relying on it.
