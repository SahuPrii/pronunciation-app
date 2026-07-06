"""
Pronunciation Scoring App — Backend
------------------------------------
Accepts an English speech audio file (30-45s), transcribes it with an
open-source Whisper model (faster-whisper, runs fully locally/offline),
and scores pronunciation using per-word recognition confidence as a proxy
for clarity/correctness. Flags low-confidence words and common filler
disfluencies as "mistakes" with a reason.

No audio is persisted to disk beyond the lifetime of a single request.
See ARCHITECTURE.md / architecture PDF for full data-handling & DPDP notes.
"""

import io
import os
import tempfile
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydub import AudioSegment
from faster_whisper import WhisperModel

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MIN_DURATION_SEC = 30
MAX_DURATION_SEC = 45
DURATION_TOLERANCE = 3  # small buffer so 28-48s doesn't hard-fail on edge cases

CONFIDENCE_GOOD = 0.85
CONFIDENCE_MODERATE = 0.60

FILLER_WORDS = {"um", "uh", "umm", "uhh", "erm", "hmm", "mm"}

MODEL_SIZE = os.environ.get("WHISPER_MODEL_SIZE", "base.en")
COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE_TYPE", "int8")

model: WhisperModel | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    # Loaded once at startup and reused for every request (stateless per-request use).
    model = WhisperModel(MODEL_SIZE, device="cpu", compute_type=COMPUTE_TYPE)
    yield


app = FastAPI(title="Pronunciation Scorer", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_duration_seconds(audio_bytes: bytes) -> float:
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
    return len(audio) / 1000.0


def classify_word(word: str, confidence: float):
    """Return (flag, reason) for a single recognized word."""
    cleaned = word.strip().lower().strip(".,!?")

    if cleaned in FILLER_WORDS:
        return "filler", "Filler word / disfluency"

    if confidence >= CONFIDENCE_GOOD:
        return "good", None
    elif confidence >= CONFIDENCE_MODERATE:
        return "moderate", "Somewhat unclear pronunciation"
    else:
        return "poor", "Likely mispronounced or unclear"


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("audio"):
        # Some browsers send generic content-types for recorded blobs; don't be too strict.
        pass

    raw_bytes = await file.read()
    if len(raw_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    # --- Duration validation (server-side enforcement of 30-45s constraint) ---
    try:
        duration = get_duration_seconds(raw_bytes)
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read audio file. Please upload a valid audio file (wav/mp3/m4a/webm).")

    if duration < (MIN_DURATION_SEC - DURATION_TOLERANCE) or duration > (MAX_DURATION_SEC + DURATION_TOLERANCE):
        raise HTTPException(
            status_code=400,
            detail=f"Audio must be 30-45 seconds long (got {duration:.1f}s).",
        )

    # --- Write to a temp file only for the duration of transcription, then delete ---
    # (faster-whisper needs a file path or file-like object; we use a NamedTemporaryFile
    # that is deleted immediately after processing — no persistent storage of raw audio.)
    tmp_path = None
    try:
        suffix = os.path.splitext(file.filename or "")[1] or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(raw_bytes)
            tmp_path = tmp.name

        segments, info = model.transcribe(
            tmp_path,
            word_timestamps=True,
            language="en",
        )

        words_out = []
        confidences = []

        for segment in segments:
            if not segment.words:
                continue
            for w in segment.words:
                conf = float(w.probability)
                confidences.append(conf)
                flag, reason = classify_word(w.word, conf)
                words_out.append({
                    "word": w.word.strip(),
                    "start": round(w.start, 2),
                    "end": round(w.end, 2),
                    "confidence": round(conf, 3),
                    "flag": flag,       # "good" | "moderate" | "poor" | "filler"
                    "reason": reason,   # None for "good"
                })

        if not confidences:
            raise HTTPException(status_code=422, detail="No speech could be detected in the audio.")

        overall_score = round((sum(confidences) / len(confidences)) * 100, 1)
        flagged_count = sum(1 for w in words_out if w["flag"] in ("poor", "filler"))

        transcript = " ".join(w["word"] for w in words_out)

        return {
            "request_id": str(uuid.uuid4()),
            "duration_seconds": round(duration, 1),
            "overall_score": overall_score,
            "transcript": transcript,
            "words": words_out,
            "flagged_count": flagged_count,
            "total_words": len(words_out),
        }
    finally:
        # Ensure the temp audio file is deleted immediately — nothing is retained.
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_SIZE}


# Serve the frontend
app.mount("/", StaticFiles(directory="static", html=True), name="static")
