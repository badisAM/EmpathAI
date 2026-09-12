"""Facial-emotion analysis for photos and live video frames (optional modality).

Uses DeepFace when installed (see requirements-face.txt). Degrades gracefully to
None if deps are missing or no face is found.
"""
from __future__ import annotations

import io
from collections import Counter

from core.emotions import FACE_TO_TAXONOMY, clamp


def face_available() -> bool:
    try:
        import deepface  # noqa: F401
        return True
    except Exception:
        return False


def _from_deepface(result) -> dict:
    first = result[0] if isinstance(result, list) else result
    raw = first.get("emotion", {})
    dominant = first.get("dominant_emotion", "neutral")
    return {"emotion": FACE_TO_TAXONOMY.get(dominant, "neutral"),
            "intensity": clamp(float(raw.get(dominant, 0.0)) / 100.0),
            "raw": {k: round(float(v), 1) for k, v in raw.items()}}


def analyze_face(image_bytes: bytes) -> dict | None:
    """Analyze a single still image (photo fallback)."""
    if not image_bytes:
        return None
    try:
        import numpy as np
        from deepface import DeepFace
        from PIL import Image
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        arr = np.array(img)[:, :, ::-1]  # RGB -> BGR
        return _from_deepface(DeepFace.analyze(arr, actions=["emotion"],
                                               enforce_detection=False, silent=True))
    except Exception:
        return None


def analyze_frame(bgr_ndarray) -> dict | None:
    """Analyze one live video frame (BGR ndarray from webrtc)."""
    try:
        from deepface import DeepFace
        return _from_deepface(DeepFace.analyze(bgr_ndarray, actions=["emotion"],
                                               enforce_detection=False, silent=True))
    except Exception:
        return None


def aggregate(samples: list[dict]) -> dict | None:
    """Combine per-frame samples into one face read with a timeline + distribution."""
    samples = [s for s in samples if s]
    if not samples:
        return None
    counts = Counter(s["emotion"] for s in samples)
    total = sum(counts.values())
    dominant = counts.most_common(1)[0][0]
    dom_int = [s["intensity"] for s in samples if s["emotion"] == dominant]
    intensity = sum(dom_int) / len(dom_int) if dom_int else 0.5
    distribution = {e: round(c / total, 2) for e, c in counts.most_common()}
    timeline = [(s["emotion"], round(s["intensity"], 2)) for s in samples]
    return {"emotion": dominant, "intensity": round(intensity, 2),
            "distribution": distribution, "timeline": timeline, "frames": total}