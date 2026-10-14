# Smriti AI / VisionQuest — Stark-inspired Cloud HUD

The primary **CV WORKSPACE** is a camera-first computer-vision console tuned for a smoother low-load live feed. It defaults to **Standard 640 × 360 at about 10 FPS**, with selectable HD 1280 × 720 and Full HD 1920 × 1080. Automatic scene explanation and local object detection start **off** so remote requests and neural-network inference do not compete with video unless enabled. Brightness/focus metrics and the UI snapshot JPEG are updated about once per second instead of on every frame. When enabled, local **OpenCV DNN + YOLOX-S INT8** runs asynchronously on a copied recent frame at a limited cadence (about once every 1.2 seconds at most); detection boxes expire after 1.5 seconds, so they can trail or disappear between scans. AI observations default to **Fast · 512 px**; Balanced 768 px and Maximum detail 1280 px remain available. The Analyze Current Frame action performs scene inference on demand. Smoothness still depends on cloud CPU, network, camera, and provider latency.

This is a **cloud-deployment edition** of VisionQuest. The compact ONNX object detector is loaded with OpenCV DNN on the app server (no PyTorch/Ultralytics installation and no per-frame object-detection API calls). The first enable downloads the model from [OpenCV Zoo](https://huggingface.co/opencv/opencv_zoo/tree/main/models/object_detection_yolox) and caches it temporarily. The separate scene-explanation function sends sampled images to the configured Hugging Face Inference Provider using `Qwen/Qwen2.5-VL-3B-Instruct:fastest`. Users can open the HTTPS URL on a phone, tablet, or computer; browser camera permission and an explicit **START** click remain required.

## Important differences from the local edition

- Still photos use browser camera capture (`st.camera_input`) or image upload. The **Live camera & recording** tab uses WebRTC to stream the visitor's camera to the cloud server; OpenCV processes frames on the server. It does not use `cv2.VideoCapture(0)` to access a remote visitor's webcam.
- The **Live camera** tab uses WebRTC to send the browser camera stream to the app server; it does not use `cv2.VideoCapture(0)` to access the visitor's camera. The selectable capture targets are Standard 640 × 360 (default), HD 1280 × 720, and Full HD 1920 × 1080. These are requested/processing targets, not a guarantee that the physical camera will negotiate the same resolution; use the visible frame dimensions to see what is being processed. After **START** and browser permission, the local OpenCV DNN YOLOX model is initially off and can be enabled in camera controls. When enabled it runs in a background worker on a recent copied frame instead of blocking each video callback. Results are periodically refreshed and boxes expire after 1.5 seconds; this trades exact frame-by-frame alignment for smoother video. It recognises the 80 common COCO categories, not every possible object or a person's identity. Actual throughput is shown as measured FPS/inference time and depends on the server CPU; it may be slower than the camera capture rate. Scene explanation is a separate, sampled task using the hosted vision-language model.
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
- **Live observation detail:** the app sends a clean, color camera sample resized to a maximum long edge of 1280 pixels by default (rather than the previous 384–512 px). It uses JPEG quality 94 for the live frame and quality 95 when packaging the image for hosted vision analysis. The provider may apply its own processing, and more detail costs additional latency/credits. For supported 16:9 Full HD camera input, the analysis image is typically 1280 × 720.
- The first object-detector initialisation downloads an approximately 9 MB quantized ONNX model into a temporary cache on the host. Streamlit Community Cloud can clear temporary files when the app restarts or redeploys; the model will then need to download again.
- The video is received by the hosted app, so object-detection speed depends on available cloud CPU. The app displays measured throughput rather than claiming a fixed FPS. It does not guarantee detection of all visible objects; smaller, blocked, blurry, unusual, or out-of-frame objects can be missed or mislabeled.
- Streamlit Community Cloud has resource and storage limits. Do not store real medical or sensitive personal information. Use fictional demo records only.
## Live camera, OpenCV, and recording

- The camera feed requires user interaction and browser permission: press **START** and allow access. The video player exposes native controls, including fullscreen where supported. Use **STOP** when finished; the site does not turn on the camera secretly.
- OpenCV reports rough brightness/focus metrics and renders six diagnostic modes. Local YOLOX object detection is run for each frame delivered to the camera callback, and results are overlaid on that same frame. CPU inference may reduce the resulting display FPS compared with camera capture.
- The separate vision-language model can run automatic scene explanation at a 5-second target interval by default (1/2/10-second settings are available when enabled). This is not guaranteed to complete every second because remote inference has variable latency. Its frames are sent to the configured provider.
- Silent MP4 recording captures the displayed CV view, up to 60 seconds or 25 MB. Recordings are temporary and offered for download after stop.

## Voice module (browser text-to-speech)

The **Voice** tab adds read-aloud controls for the latest visual observation, saved memory notes, or text typed by the user. A small inline browser player provides **Speak**, **Stop**, and a speaking-speed slider. The user can choose English (India), English (US), English (UK), Hindi, or Tamil. The browser selects the closest installed device voice; supported voices vary by browser and operating system.

- Speech is generated on the visitor's device using the browser Web Speech API. No new Python package or external TTS API key is required.
- The text to be spoken remains in the already-loaded page; the voice module does not send it to a separate TTS provider or save an audio file.
- The user must tap **Speak**; the app does not autoplay speech. Use **Stop** to cancel speech.
- Current Chrome and Edge generally support browser speech synthesis. If it is unavailable, check the browser/device's speech capabilities and volume.
- This feature is text-to-speech only. Speech recognition/transcription from a microphone is not included in this cloud build.

## Security and data

This is a shared-passphrase demo, **not individual authentication**. Everyone who knows the passphrase can access the shared vault. The app currently has no roles, account-specific databases, encrypted clinical storage, or guaranteed durable backing database. Do not use identifiable health records. For real cross-device memory persistence, add user accounts and a managed database with per-user access policies before deployment.
