# Online deployment quick checklist

- [ ] Source files are at repository root (app.py + requirements.txt + core/).
- [ ] No `.env`, secrets, database, or identifiable images included in GitHub.
- [ ] Streamlit Community Cloud main file is `app.py`, Python 3.12.
- [ ] `APP_PASSWORD` added in hosting Secrets, not in source code.
- [ ] Deployment logs show all dependencies installed.
- [ ] First analysis tested with a non-sensitive sample image.
- [ ] Browser camera permission tested over the HTTPS URL.
- [ ] App password kept private; use fictional sample data only.
