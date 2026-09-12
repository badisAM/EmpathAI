"""Deterministic speech metrics derived from a transcript (and audio duration)."""
from __future__ import annotations

import io
import re
import wave

FILLERS = ["um", "uh", "er", "ah", "like", "you know", "basically", "actually",
           "literally", "kinda", "sorta", "i mean", "right", "so"]


def wav_duration(audio_bytes: bytes) -> float | None:
    """Best-effort duration in seconds for a WAV clip; None if it can't be read."""
    try:
        with wave.open(io.BytesIO(audio_bytes)) as w:
            fr = w.getframerate()
            return w.getnframes() / float(fr) if fr else None
    except Exception:
        return None


def analyze_speech(transcript: str, duration: float | None = None) -> dict:
    text = (transcript or "").lower()
    words = re.findall(r"[a-zA-Z']+", text)
    wc = len(words)
    breakdown: dict[str, int] = {}
    fillers = 0
    for f in FILLERS:
        n = len(re.findall(r"\b" + re.escape(f) + r"\b", text))
        if n:
            breakdown[f] = n
            fillers += n
    wpm = round(wc / (duration / 60)) if duration and duration > 0 else None
    return {"words": wc, "fillers": fillers, "filler_breakdown": breakdown,
            "wpm": wpm, "duration": round(duration, 1) if duration else None}