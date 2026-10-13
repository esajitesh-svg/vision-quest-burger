# VisionQuest cloud dependency fix

The previous deployment pinned `torch==2.6.0+cpu`, which is not installable in the Python 3.14.8 runtime shown in the Streamlit Cloud logs. This version updates the pin to `torch==2.9.1+cpu`, which has a CPython 3.14 Linux CPU wheel on the official PyTorch CPU wheel index.

## Apply to an existing GitHub repository

1. Replace the repository's `requirements.txt` with this file (or edit only the torch line).
2. Commit the change to the branch used by Streamlit, currently `main`.
3. Wait for Streamlit Community Cloud to rebuild.
4. Check logs: it should process `torch==2.9.1+cpu`, not `torch==2.6.0+cpu`.

If the install still fails, use Streamlit Advanced settings to redeploy with Python 3.12. That is a valid alternative because it matches the original PyTorch 2.6-era stack, but Python 3.14 can use the updated torch pin.

Do not commit secrets or private records to GitHub. This is still a shared-vault prototype, not a production-ready health application.
