# Smriti AI / VisionQuest — Stark-inspired Cloud HUD

The primary **CV WORKSPACE** is a camera-first computer-vision console tuned for a smoother low-load live feed. It defaults to **Standard 640 × 360 at about 10 FPS**, with selectable HD 1280 × 720 and Full HD 1920 × 1080. Automatic scene explanation starts **enabled** by default, while local object detection starts **off** so heavier inference runs only when requested. Brightness/focus metrics and the UI snapshot JPEG are updated about once per second instead of on every frame. When enabled, local **OpenCV DNN + YOLOX-S INT8** runs asynchronously on a copied recent frame at a limited cadence (about once every 1.2 seconds at most); detection boxes expire after 1.5 seconds, so they can trail or disappear between scans. AI observations default to **Fast · 512 px**; Balanced 768 px and Maximum detail 1280 px remain available. The Analyze Current Frame action performs scene inference on demand. Smoothness still depends on cloud CPU, network, camera, and provider latency.

This is a **cloud-deployment edition** of VisionQuest. The compact ONNX object detector is loaded with OpenCV DNN on the app server (no PyTorch/Ultralytics installation and no per-frame object-detection API calls). The first enable downloads the model from [OpenCV Zoo](https://huggingface.co/opencv/opencv_zoo/tree/main/models/object_detection_yolox) and caches it temporarily. The separate scene-explanation function sends sampled images to the configured Hugging Face Inference Provider using `Qwen/Qwen2.5-VL-3B-Instruct:fastest`. Users can open the HTTPS URL on a phone, tablet, or computer; browser camera permission and an explicit **START** click remain required.

## Important differences from the local edition

- Still photos use browser camera capture (`st.camera_input`) or image upload. The **Live camera & recording** tab uses WebRTC to stream the visitor's camera to the cloud server; OpenCV processes frames on the server. It does not use `cv2.VideoCapture(0)` to access a remote visitor's webcam.
- The **Live camera** tab uses WebRTC to send the browser camera stream to the app server; it does not use `cv2.VideoCapture(0)` to access the visitor's camera. The selectable capture targets are Standard 640 × 360 (default), HD 1280 × 720, and Full HD 1920 × 1080. These are requested/processing targets, not a guarantee that the physical camera will negotiate the same resolution; use the visible frame dimensions to see what is being processed. After **START** and browser permission, the local OpenCV DNN YOLOX model is initially off and can be enabled in camera controls. When enabled it runs in a background worker on a recent copied frame instead of blocking each video callback. Results are periodically refreshed and boxes expire after 1.5 seconds; this trades exact frame-by-frame alignment for smoother video. It recognises the 80 common COCO categories, not every possible object or a person's identity. Actual throughput is shown as measured FPS/inference time and depends on the server CPU; it may be slower than the camera capture rate. Scene explanation is a separate, sampled task using the hosted vision-language model.
- WebRTC requires HTTPS and network access to STUN/TURN. The app configures a public STUN server by default. If a device/network cannot connect, configure `TURN_SERVER_URL`, `TURN_USERNAME`, and `TURN_CREDENTIAL` in Streamlit Secrets with credentials from a TURN provider. The camera stream itself does not capture a microphone track. Live JARVIS uses browser-native `SpeechRecognition` after the user presses **ACTIVATE LISTENING**; browser speech services may process audio according to that browser/provider's terms. The app stores the recognized text and reply, not raw microphone audio.
- Add both `APP_PASSWORD` and `HF_TOKEN` in Streamlit's Secrets. `HF_TOKEN` must have Inference Providers permission. Keep all tokens out of the source and repository. The hosted inference provider may consume your Hugging Face monthly credits and can be subject to rate limits or paid usage; check the current pricing/usage page before using it.
- The demo vault supports local SQLite for development or a managed PostgreSQL database via `DATABASE_URL`. It is still a single shared vault protected by one app password; anyone with that password can access the same profiles, portraits, conversations and activity logs. This is not per-user private storage.
- The default Streamlit Community Cloud filesystem should not be treated as durable storage. A restart/redeploy can cause local SQLite data loss; durable multi-session memory requires an external database and backups. See the long-term memory setup below. Do not store real medical or highly sensitive personal information.
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
   HF_TOKEN = "your-hugging-face-token"
   DATABASE_URL = "postgresql://USER:PASSWORD@HOST:5432/DATABASE?sslmode=require"
   ```

   `HF_TOKEN` must have Inference Providers permission. `DATABASE_URL` must point to a managed PostgreSQL database that you have created; it is not a database hosted by Streamlit. Replace the example values with your provider's connection string. Do not paste real credentials into chat or commit them to GitHub. Streamlit provides a secrets panel for this purpose: <https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management>.

6. Deploy. The first build downloads dependencies. The first image analysis also downloads the model and may take several minutes.
7. Open the generated `https://<your-app-name>.streamlit.app` URL on your phone, tablet, or any computer. Enter your app password.

Official deployment guide: <https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy>.

## Long-term memory, activity logs, and portraits

The app now stores **text conversation turns** (recognized transcript + Smriti reply) and **completed text observations/analyses** in database tables. Live camera observations are logged when **Remember completed camera observations in the long-term Activity Log** is enabled. The one-shot Image Lab also logs the resulting text summary. The voice companion searches saved notes plus relevant/recent prior conversations and observations when answering recall questions. Open **ACTIVITY LOG** to review the records and export a CSV backup.

For persistence across redeploys, create a remote PostgreSQL database and set the top-level `DATABASE_URL` secret shown in the deployment section. Without it, the app clearly reports that it is using local SQLite, which can be reset by Streamlit hosting. PostgreSQL does not make data literally permanent on its own: keep independent backups and follow the database provider's retention and backup rules. The built-in CSV export is one portable backup option.

**Portrait memory is user-controlled.** In **PERSON NODES**, upload a person's photo only after confirming you have permission to keep it. The photo is stored against the named profile in the configured database and can be removed there. Smriti does not identify people by matching faces or claim that a face proves an identity; a person/name connection must be confirmed by the user. A portrait is a reference picture, not a biometric recognition feature.

The Activity Log stores text summaries rather than every camera frame; raw live microphone audio is not stored. This is intentional to limit data volume and privacy risk. Turning the long-term observation checkbox off prevents new scene summaries from being logged. Remove a portrait from **PERSON NODES** if you no longer want it kept, and use database-provider controls/backup policies to manage older logs.

## How to use the live CV workspace

1. Open the deployed app using its HTTPS URL and go to **CV WORKSPACE**.
2. Keep **Enable continuous local object detection + bounding boxes** switched on in **PERCEPTION / CAMERA CONTROLS**. First use may take longer while the compact ONNX model downloads and initialises.
3. Click **START** in the live camera player and approve the browser's camera permission. Once started, the feed processes each frame delivered to the app server until you stop it.
4. Use the video's fullscreen control to make the camera fill the screen. Expand the controls panel only when you need to change detector confidence, scene-analysis interval, or OpenCV display mode.
5. Watch boxes from actual local inference, and check the object list plus measured inference throughput. Enable scene explanation or click **Analyze Current Frame Now** for a separate natural-language description. Click **Freeze / refresh report for spoken playback**, then **SPEAK FULL LIVE REPORT** to hear the current report.

The webpage cannot silently start the camera without a user gesture and browser permission. The live video stream is processed on the app host. Local object detection does not send each frame to a hosted detector API, but scene-analysis samples are sent to the configured external vision-language provider. Do not use images of anyone without permission; this prototype is not a validated dementia-support system.

## Model and resource notes

- **Live object detector:** OpenCV Zoo YOLOX-S INT8 ONNX, 80 common COCO classes; loaded by OpenCV DNN on the server CPU. The model directory documents an Apache-2.0 license: [OpenCV Zoo YOLOX model card](https://huggingface.co/opencv/opencv_zoo/tree/main/models/object_detection_yolox).
- **Scene-explanation model:** `Qwen/Qwen2.5-VL-3B-Instruct:fastest` through Hugging Face Inference Providers. Configure `HF_TOKEN` in Streamlit Secrets for this separate feature. Remote latency/credits affect written scene descriptions but not the local detector's ability to draw boxes.
- **Live observation detail:** the smooth-first defaults use Standard 640 × 360 capture and Fast 512 px AI input. When you choose Balanced or Maximum detail, the app resizes a clean color frame and encodes it only when an observation is requested, not on every video frame. Automatic scene explanation is enabled by default; its target interval is normally 5 seconds and changes to 2 seconds when live JARVIS is enabled. Higher detail and remote inference can still add latency or consume more provider credits.
- The first object-detector initialisation downloads an approximately 9 MB quantized ONNX model into a temporary cache on the host. Streamlit Community Cloud can clear temporary files when the app restarts or redeploys; the model will then need to download again.
- The video is received by the hosted app, so object-detection speed depends on available cloud CPU. The app displays measured throughput rather than claiming a fixed FPS. It does not guarantee detection of all visible objects; smaller, blocked, blurry, unusual, or out-of-frame objects can be missed or mislabeled.
- Streamlit Community Cloud has resource and storage limits. Do not store real medical or sensitive personal information. Use fictional demo records only.
## Live camera, OpenCV, and recording

- The camera feed requires user interaction and browser permission: press **START** and allow access. The video player exposes native controls, including fullscreen where supported. Use **STOP** when finished; the site does not turn on the camera secretly.
- OpenCV reports rough brightness/focus metrics and renders six diagnostic modes. Local YOLOX object detection is run for each frame delivered to the camera callback, and results are overlaid on that same frame. CPU inference may reduce the resulting display FPS compared with camera capture.
- The separate vision-language model runs sampled automatic scene explanation when the visible control is enabled (enabled by default). The target interval options are 1, 2, 5, and 10 seconds; live JARVIS uses a 2-second target by default, otherwise 5 seconds. This is not guaranteed to complete at that cadence because remote inference has variable latency. Sampled frames are sent to the configured provider.
- Silent MP4 recording captures the displayed CV view, up to 60 seconds or 25 MB. Recordings are temporary and offered for download after stop.

## Live JARVIS-style voice + camera

At the top of **CV WORKSPACE → LIVE SENSING**, the **JARVIS VOICE LINK · LIVE COMPANION** controls are visible without opening the settings panel. Turn on **Enable LIVE JARVIS listening + replies**, then press **ACTIVATE LISTENING** inside the browser voice panel and grant microphone permission. The camera is optional for general questions; start it when you want answers grounded in the live view. Keep **Speak new camera observations aloud** enabled for spoken scene summaries. Press **STOP** in the voice panel to stop listening; stop the camera to end visual sensing.

- The live path uses the browser's built-in `SpeechRecognition` interface to avoid a separate hosted Whisper transcription round-trip before every live answer. It listens continuously while activated, submits final recognized text to the app, pauses recognition while Smriti speaks to reduce feedback, and resumes afterwards. Browser recognition support varies; current Chrome/Edge browsers are the intended target. The browser's speech-recognition service may process audio externally. The app server stores the recognized text/reply, not raw microphone audio.
- The transcript is answered by the hosted vision-language model using up to four recent conversation turns, relevant saved notes, and a freshly encoded camera frame when one arrived within about four seconds. For explicit broad recall prompts such as “I forgot” or “help me remember,” the app can also supply a few recent saved notes when keyword search finds no match. It presents these as saved notes, not a guarantee of what the user forgot.
- **Speak fresh camera descriptions aloud** is on by default when live JARVIS mode is enabled. It asks for concise camera observations at the selected scene-analysis interval and speaks new descriptions using the same browser-native speech helper already used elsewhere in the site. Scene description is periodic, not guaranteed frame-by-frame; provider latency, network, and host capacity still apply.
- Replies use the selected language and speaking speed and use a voice supplied by the visitor's browser/device. The voice panel includes **REPLAY LAST** and **SPEAK ANSWER** controls for retrying playback. If the microphone service is unavailable, or you want to isolate AI response generation from speech recognition, use the typed-question field and **SEND QUESTION TO SMRITI** in the live JARVIS panel. The browser microphone status and any AI/provider error are shown in the app.
- Live voice requires a working `HF_TOKEN` with appropriate Inference Providers permissions. Each recognized utterance and its context is sent to the configured hosted services. Microphone/camera privacy is important: only enable live mode knowingly, and stop the WebRTC player when finished.
- This is still not a fully local, end-to-end streaming AI model: the browser recognizes speech, but answer generation and visual analysis use the hosted Hugging Face model. The AI cannot reply faster than that service/network responds, and speech results vary by browser. Replies are automatically spoken through the persistent browser voice component; its visible voice panel also shows microphone and playback status.

## Live auto-storing of summaries

For per-person curated memory notes, use **PERCEPTION / CAMERA CONTROLS** to enable **automatic scene explanation** and **Auto-save new scene summaries to memory**; choose the profile and minimum interval (30, 60, or 120 seconds). Separately, the **Remember completed camera observations in the long-term Activity Log** checkbox saves every completed text analysis to the activity history, while the live voice opt-in saves each recognized conversation and reply. Neither feature stores every frame. Keep camera/microphone active only with appropriate consent.

## Security and data

This is a shared-passphrase demo, **not individual authentication**. Everyone who knows the passphrase can access the shared vault. It has no roles or account-specific data isolation, and must not be used for clinical records or highly sensitive information. A configured external PostgreSQL database improves durability across app redeploys, but backups and database-provider retention remain essential.
