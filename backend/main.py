"""FastAPI backend for EmpathAI Coach — serves the custom web frontend.

Run:  uvicorn backend.main:app --reload --port 8000
Endpoints: POST /coach, POST /feedback, GET /stats, GET /health
"""
from __future__ import annotations

import base64

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from core.pipeline import EmpathAI

app = FastAPI(title="EmpathAI Coach API", version="3.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_engine: EmpathAI | None = None


def engine() -> EmpathAI:
    global _engine
    if _engine is None:
        _engine = EmpathAI()
    return _engine


class CoachRequest(BaseModel):
    question: str
    audio_b64: str
    audio_mime: str | None = None
    duration: float | None = None
    frames_b64: list[str] = []      # webcam frames sampled during the answer
    image_b64: str | None = None    # single-snapshot fallback
    face: dict | None = None        # precomputed by the browser (face-api.js)
    role: str = ""
    session_id: str = "web"


class FeedbackRequest(BaseModel):
    attempt_id: int
    value: int


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "face": engine().face_enabled()}


@app.post("/coach")
def coach(req: CoachRequest) -> dict:
    eng = engine()
    audio = base64.b64decode(req.audio_b64)
    ext = "webm" if (req.audio_mime and "webm" in req.audio_mime) else "wav"
    transcript = eng.transcribe(audio, filename=f"speech.{ext}")

    face = req.face  # browser-computed emotion (preferred)
    if face is None and eng.face_enabled():
        from core import vision
        if req.frames_b64:
            samples = [vision.analyze_face(base64.b64decode(f)) for f in req.frames_b64]
            face = vision.aggregate([s for s in samples if s])
        elif req.image_b64:
            face = vision.analyze_face(base64.b64decode(req.image_b64))

    return eng.analyze_answer(req.question, transcript, audio_bytes=audio, face=face,
                              role=req.role, duration=req.duration, session_id=req.session_id)


@app.post("/feedback")
def feedback(req: FeedbackRequest) -> dict:
    engine().feedback(req.attempt_id, req.value)
    return {"ok": True}


@app.get("/stats")
def stats(session_id: str = "web") -> dict:
    return engine().memory.stats(session_id)