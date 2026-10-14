# Smriti AI / VisionQuest — Stark-inspired Cloud HUD

The primary **LIVE HUD** workspace opens on the live camera panel with a short animated VisionQuest boot splash. The interface uses a dark, cyan-accented command-console style. OpenCV runs on every incoming live frame by default, while **Auto-observe** sends sampled frames to the hosted vision model to summarize the scene, visible objects/locations, actions, and readable text. The target AI request interval defaults to **1 second**, with 2-, 5-, and 10-second options. Requests are serialized; if remote inference takes longer than the interval, the next request waits for the previous one to finish. **Analyze Current Frame Now** is also available. A browser speech button can pronounce the latest observation.

This is a **cloud-deployment edition** of VisionQuest. The Streamlit app does not load PyTorch/Transformers locally (which can exceed Streamlit Community Cloud's memory limit); instead it sends sampled images to a hosted Hugging Face Inference Provider using `Qwen/Qwen2.5-VL-3B-Instruct:fastest`. Users can open the HTTPS URL from a phone, tablet, or computer.

## Important differences from the local edition

- Still photos use browser camera capture (`st.camera_input`) or image upload. The **Live camera & recording** tab uses WebRTC to stream the visitor's camera to the cloud server; OpenCV processes frames on the server. It does not use `cv2.VideoCapture(0)` to access a remote visitor's webcam.
- OpenCV runs continuously on incoming camera frames while the WebRTC stream is active (with an option to toggle the OpenCV path). Auto-observe is the default vision mode: at the selected 15/30/60-second interval, the remote vision model reports the scene, visible objects and approximate positions, visible activities, and readable text. A single frame can also be analyzed manually. The model is not run on every frame because that would be slow and can consume inference credits. Live mode also supports silent MP4 recording (maximum 60 seconds or 25 MB). The user must explicitly start and stop the stream/recording. A recording is held temporarily for download; it is not a permanent server-side archive.
- WebRTC requires HTTPS and network access to STUN/TURN. The app configures a public STUN server by default. If a device/network cannot connect, configure `TURN_SERVER_URL`, `TURN_USERNAME`, and `TURN_CREDENTIAL` in Streamlit Secrets with credentials from a TURN provider. Audio is not recorded.
- Add both `APP_PASSWORD` and `HF_TOKEN` in Streamlit's Secrets. `HF_TOKEN` must have Inference Providers permission. Keep all tokens out of the source and repository. The hosted inference provider may consume your Hugging Face monthly credits and can be subject to rate limits or paid usage; check the current pricing/usage page before using it.
- The demo vault is a single shared SQLite database on the app host and is protected by one app password. Anyone with that password can see and export the same demo records.
- The default Streamlit Community Cloud filesystem should not be treated as durable storage. A restart/redeploy can cause database loss. Do not store real medical or sensitive personal information. Use fictional demo records only.
- This is a prototype, not a medical device or validated dementia-support system.

## Deploy to a public HTTPS URL (Streamlit Community Cloud)

### 1. Create a repository

1. Download and extract this ZIP. The ZIP is arranged so `app.py`, `requirements.txt`, `core/`, `tests/`, and `.streamlit/` are directly at the extracted root; there is no extra nested project folder.
2. Create a GitHub repository. Keep in mind that a public repository exposes the source code; **never upload passwords, secrets, personal photos, or database files**.
3. Upload the extracted project files so `app.py`, `requirements.txt`, `README.md`, `core/`, and `.streamlit/` are at the repository root. Use GitHub Desktop if you want an easier way to publish a full folder.

### 2. Deploy the app

1. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/) using the account workflow it requires.
2. Choose **Create app** and select the repository and branch you created.
3. Set the main file path to `app.py`.
4. Open **Advanced settings** and select Python 3.12 if prompted.
5. In the **Secrets** box, set a long, unique password like this (replace the example value):

   ```toml
   APP_PASSWORD = "replace-with-a-long-unique-passphrase"
   ```

   Never commit this value to GitHub. Streamlit provides a secrets panel for this purpose: <https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management>.

6. Deploy. The first build downloads dependencies. The first image analysis also downloads the model and may take several minutes.
7. Open the generated `https://<your-app-name>.streamlit.app` URL on your phone, tablet, or any computer. Enter your app password.

Official deployment guide: <https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy>.

## How to use the online camera

1. Open the deployed app using its HTTPS URL.
2. Select **Vision** → **Use this device's camera**.
3. Approve browser camera permission when prompted and capture a still image. Upload a file instead if the camera option isn't available.
4. Select a task and click **Analyze image**.
5. If needed, manually select a fictional person to save the reviewed description as a memory.

The phone or laptop camera is captured **in the browser** and the image bytes are sent to the hosted app for inference. The image is not intentionally written to disk by this app, but the hosting platform processes it. Do not use this for photos of people without their permission.

## Model and resource notes

- Model: `HuggingFaceTB/SmolVLM-256M-Instruct` (Apache-2.0 model card: <https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct>).
- It runs on cloud CPU by default, no local Ollama install or API token is required for downloading this public model.
- The first inference may be slow due to model download and cold loading. Small images and a short output limit reduce resource use but don't guarantee a particular response time.
- Streamlit Community Cloud has resource limits, so if the instance runs out of memory, use a hosted model-inference service or a hosting platform with more compute. Limits can change: <https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app>.

## Live camera, OpenCV, and recording

1. Open the **LIVE HUD** main tab, then **LIVE SENSING // WEBCAM**.
2. Allow camera permission and click **START** inside the WebRTC player. If the status stays idle, check browser permission and network connectivity.
3. Choose a live analysis mode: automatic scene + objects + activities, scene overview, objects and approximate positions, visible-text reading, short spoken description, or custom visual question.
4. Use **Analyze Current Frame Now** for an immediate answer, or enable auto-observe with a 1-second target interval (2/5/10 seconds are also available). The model is not run concurrently on every frame; if a request takes longer than the interval, VisionQuest waits for it to finish. OpenCV resizes frames, reports rough brightness/focus metrics, and can apply optional contrast enhancement before sampled frames are analyzed by SmolVLM.
5. Use **Pronounce Observation** to read a result aloud. Browser voice availability varies by device and language.
6. Click **Start MP4 recording** to record a silent clip. Stop it manually or let the selected 15/30/60-second limit end it; the final clip is offered as a download. Temporary files are removed after finalization. Each browser session has its own camera and recording state.
5. Stop the WebRTC stream when finished. The stream sends live images to the cloud host while it is running; do not use it without permission from anyone in view.

This implementation is live WebRTC video plus periodic single-frame model inference, not model inference on every frame. CPU inference can be slow. Public STUN may not work on every network; if required, configure TURN server credentials through host secrets. Never place TURN credentials in source control.

## Voice module (browser text-to-speech)

The **Voice** tab adds read-aloud controls for the latest visual observation, saved memory notes, or text typed by the user. A small inline browser player provides **Speak**, **Stop**, and a speaking-speed slider. The user can choose English (India), English (US), English (UK), Hindi, or Tamil. The browser selects the closest installed device voice; supported voices vary by browser and operating system.

- Speech is generated on the visitor's device using the browser Web Speech API. No new Python package or external TTS API key is required.
- The text to be spoken remains in the already-loaded page; the voice module does not send it to a separate TTS provider or save an audio file.
- The user must tap **Speak**; the app does not autoplay speech. Use **Stop** to cancel speech.
- Current Chrome and Edge generally support browser speech synthesis. If it is unavailable, check the browser/device's speech capabilities and volume.
- This feature is text-to-speech only. Speech recognition/transcription from a microphone is not included in this cloud build.

## Security and data

This is a shared-passphrase demo, **not individual authentication**. Everyone who knows the passphrase can access the shared vault. The app currently has no roles, account-specific databases, encrypted clinical storage, or guaranteed durable backing database. Do not use identifiable health records. For real cross-device memory persistence, add user accounts and a managed database with per-user access policies before deployment.
