"""Emotion taxonomy, UI colours, adaptation policy, and multimodal fusion.

The fusion step is the project's differentiator: it combines the emotion inferred
from *text* with the emotion inferred from the user's *face* (camera), and flags
when the two disagree ("congruence").
"""
from __future__ import annotations

EMOTIONS = ["joy", "sadness", "anger", "fear", "surprise", "frustration", "neutral"]

# Colour encodes the detected emotion in the UI (information, not decoration).
EMOTION_COLORS = {
    "joy": "#F2B705",
    "sadness": "#5B8DEF",
    "anger": "#E5484D",
    "fear": "#8B6FE8",
    "surprise": "#12B5A5",
    "frustration": "#F2711C",
    "neutral": "#8A94A6",
}

# DeepFace returns these 7 labels; map them onto our taxonomy.
FACE_TO_TAXONOMY = {
    "happy": "joy",
    "sad": "sadness",
    "angry": "anger",
    "fear": "fear",
    "surprise": "surprise",
    "disgust": "frustration",
    "neutral": "neutral",
}

# How the responder shifts style depending on the fused emotion.
ADAPTATION = {
    "joy":         {"tone": "warm and celebratory",      "length": "medium",       "focus": "match their energy and encourage them"},
    "sadness":     {"tone": "gentle and validating",     "length": "short-medium", "focus": "acknowledge the feeling before anything else"},
    "anger":       {"tone": "calm and non-defensive",    "length": "short",        "focus": "de-escalate and address the actual issue"},
    "fear":        {"tone": "reassuring and steady",     "length": "short",        "focus": "ground them and give one concrete next step"},
    "surprise":    {"tone": "clear and informative",     "length": "medium",       "focus": "explain calmly and fill the gap"},
    "frustration": {"tone": "direct and solution-first", "length": "short",        "focus": "fix the problem fast, minimal fluff"},
    "neutral":     {"tone": "friendly and efficient",    "length": "medium",       "focus": "answer the question clearly"},
}


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def fuse(text_emotion: dict, face_emotion: dict | None,
         text_weight: float = 0.6) -> dict:
    """Combine text- and face-derived emotions into a single state.

    Returns: {emotion, intensity, congruent, note, sources}
    - congruent is None when no face signal is available.
    - When the two signals disagree, `note` describes the mismatch so the
      responder can gently check in.
    """
    t_emo = text_emotion.get("emotion", "neutral")
    t_int = clamp(float(text_emotion.get("intensity", 0.5)))

    if not face_emotion:
        return {"emotion": t_emo, "intensity": t_int, "congruent": None,
                "note": "", "sources": {"text": text_emotion, "face": None}}

    f_emo = face_emotion.get("emotion", "neutral")
    f_int = clamp(float(face_emotion.get("intensity", 0.5)))

    if t_emo == f_emo:
        emotion = t_emo
        intensity = clamp(text_weight * t_int + (1 - text_weight) * f_int + 0.05)
        return {"emotion": emotion, "intensity": intensity, "congruent": True,
                "note": "", "sources": {"text": text_emotion, "face": face_emotion}}

    # Disagreement: primary = the stronger signal, but surface the mismatch.
    if f_int > t_int + 0.15:
        emotion, intensity = f_emo, f_int
    else:
        emotion, intensity = t_emo, t_int
    note = (f"Words read as '{t_emo}' but facial expression reads as '{f_emo}'. "
            "Consider gently checking how they really feel.")
    return {"emotion": emotion, "intensity": clamp(intensity), "congruent": False,
            "note": note, "sources": {"text": text_emotion, "face": face_emotion}}
