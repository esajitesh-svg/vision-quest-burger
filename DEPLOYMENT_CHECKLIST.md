# Smriti AI / VisionQuest — deployment and release checklist

Use this checklist after deploying to Streamlit Community Cloud and whenever voice, camera, or memory code changes.

## Source and hosting configuration

- [ ] Repository root contains `app.py`, `requirements.txt`, `core/`, `components/`, and `tests/`.
- [ ] No API tokens, app passwords, database files, private portraits, or personal data are committed to GitHub.
- [ ] Streamlit Community Cloud points to the intended repository, `main` branch, and `app.py` entry point.
- [ ] Python/runtime build completes and the app starts without a traceback.
- [ ] Streamlit Secrets contains a private `APP_PASSWORD` and a valid `HF_TOKEN` with Inference Providers permission. Never paste either secret into a public issue or source file.
- [ ] If memory must survive restarts/redeploys, `DATABASE_URL` points to the intended managed PostgreSQL database and a backup/export plan exists. Without this, the app may use local SQLite, which is not durable on hosted storage.
- [ ] App password and database are treated as a shared demo vault, not individual/private user accounts.

## First-run functional checks

- [ ] Sign in and confirm the CV Workspace loads without errors.
- [ ] In Image Lab, upload a non-sensitive sample image and verify written analysis returns. If it fails, read the shown provider error; check token permissions, model/provider availability, and credits.
- [ ] Open **LIVE SENSING**, click **START**, allow browser camera permission, and verify the frame counter increases.
- [ ] Click **ANALYZE CURRENT FRAME NOW** and verify a written observation appears.
- [ ] If needed, enable local object detection in camera controls and verify the status reports the real detector result/errors rather than assuming every object must be detected.
- [ ] In **PERSON NODES**, test a fictional profile and confirm portraits are uploaded only when you have permission to store the image.
- [ ] In **MEMORY INDEX** and **ACTIVITY LOG**, create/review a fictional note, confirm the correct text appears, and test export/backup. Do not use real medical or highly sensitive data in this demo.

## JARVIS voice and reply pipeline

- [ ] In **LIVE SENSING**, enable **Enable LIVE JARVIS listening + replies**. The camera is optional for general questions.
- [ ] Click **ACTIVATE LISTENING** and verify the panel reports microphone/listening status. Grant browser permission if prompted.
- [ ] Ask a short general question and pause. Verify the recognized question appears under **You:** and a reply appears under **JARVIS:**.
- [ ] Verify the answer is spoken. If automatic playback fails, use **REPLAY LAST** or **SPEAK ANSWER**.
- [ ] If recognition is unavailable, type the same question in **Or type a question for Smriti** and press **SEND QUESTION TO SMRITI**. Verify a JARVIS text reply appears; this tests the AI/provider path without browser speech recognition.
- [ ] If the typed question also fails, read **Smriti notice** on the page. Treat the provider/runtime error as the next action (for example, check `HF_TOKEN`, Inference Providers permission, credits/rate limits, or the selected model); do not keep adjusting speaker settings.
- [ ] Ask a visual question only after the camera is streaming, and verify the reply notes when it used a fresh camera frame.
- [ ] Confirm **STOP** turns off voice listening. Stop the camera when finished; do not leave microphone/camera permissions active unnecessarily.

## Automated checks and known limits

- [ ] Latest GitHub Actions workflow passes JavaScript syntax, browser-component smoke checks, Python compilation, runtime imports, and `pytest`.
- [ ] Repeat a real browser test after a voice-component change; a mocked browser test cannot prove physical speaker output, microphone permissions, or hosted-provider availability.
- [ ] Test the deployed URL in the intended browser (current Chrome/Edge are the supported target for browser-native speech recognition).
- [ ] Document any failed step and its visible error before announcing a release.

A passing CI run is evidence that the automated checks passed, not proof that every external secret, model provider, browser, camera, or speaker works in every environment. VisionQuest is a prototype, not a validated medical device or a guarantee of zero defects.
