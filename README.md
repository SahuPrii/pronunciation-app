# Pronunciation Coach

Upload a 30-45 second English speech recording and get a pronunciation score
with specific words highlighted where clarity/pronunciation was likely off.

Fully open-source, no paid APIs — uses [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
(CTranslate2 build of OpenAI Whisper) running locally on CPU.

## How it works

1. Audio is uploaded to `POST /analyze`.
2. Duration is validated server-side (must be 30-45s).
3. `faster-whisper` transcribes the audio with word-level timestamps and
   per-word confidence (probability) scores.
4. Each word is classified:
   - confidence ≥ 0.85 → **good** (clear)
   - 0.60 ≤ confidence < 0.85 → **moderate** (somewhat unclear)
   - confidence < 0.60 → **poor** (likely mispronounced/unclear)
   - filler words (um, uh, etc.) → flagged separately
5. Overall score = average word confidence × 100.
6. The temp audio file is deleted immediately after transcription — nothing
   is persisted to disk or a database.

See `ARCHITECTURE.pdf` for the full system design and DPDP compliance notes.

## Run locally

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

Then open http://localhost:8000

Requires `ffmpeg` installed locally (`apt install ffmpeg` / `brew install ffmpeg`).

## Deploy (Render, Docker-based — also works on Railway/Fly.io with the same Dockerfile)

1. Push this repo to GitHub.
2. On [Render](https://render.com): New → Web Service → connect the repo.
   Render will auto-detect `render.yaml` / the `Dockerfile`.
3. Plan: Free. Health check path: `/health`.
4. Deploy — first build takes a few minutes (downloads the Whisper model at
   build time so it's baked into the image).

## Project structure

```
main.py           FastAPI backend (transcription + scoring)
static/index.html Frontend (vanilla HTML/CSS/JS)
requirements.txt  Python deps
Dockerfile        Container build (includes ffmpeg + pre-downloaded model)
render.yaml       Render deployment config
```
