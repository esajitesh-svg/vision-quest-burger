# Deployment note — current cloud edition

This note supersedes an older deployment workaround that discussed changing a PyTorch version. That workaround belongs to a previous build and **must not be applied to the current cloud edition**.

The current VisionQuest cloud repository uses OpenCV DNN for local object detection and the hosted Hugging Face Inference Providers API for scene understanding and conversational replies. The current `requirements.txt` does not pin PyTorch.

## Deploy the current edition

1. In Streamlit Community Cloud, use this repository's `main` branch and set the main file path to `app.py`.
2. Prefer the tested Python 3.12 runtime described in `README.md`.
3. Configure `APP_PASSWORD` and `HF_TOKEN` in Streamlit Secrets. The Hugging Face token needs Inference Providers permission. Add `DATABASE_URL` only if a managed PostgreSQL database has been created and should be used for durable memory.
4. Let Streamlit rebuild from the current `requirements.txt`. Do not manually add the old Torch pin to this build.
5. Verify the deployed application using `DEPLOYMENT_CHECKLIST.md`, especially the typed-question path under LIVE JARVIS. When a spoken question produces no answer, test a typed question and inspect the displayed provider error before changing browser audio settings.

Never commit passwords, API tokens, database credentials, private portraits, or database files to GitHub. The app is a shared-vault prototype, not a production-ready or clinically validated health application.
