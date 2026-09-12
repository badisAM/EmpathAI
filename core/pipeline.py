"""High-level facade for the interview/pitch coach (voice + text + photo/video face)."""
from __future__ import annotations

from core.agents import Coachestrator
from core.config import ENABLE_FACE
from core.llm import RotatingGroqClient
from core.memory import MemoryStore
from core.rag import Retriever, load_knowledge
from core.speech import analyze_speech, wav_duration
from core.vision import analyze_face, face_available


class EmpathAI:
    def __init__(self, llm=None, memory: MemoryStore | None = None):
        self.llm = llm or RotatingGroqClient()
        self.memory = memory or MemoryStore()
        self.retriever = Retriever(load_knowledge())
        self.orch = Coachestrator(self.llm, self.retriever, self.memory)

    def face_enabled(self) -> bool:
        return ENABLE_FACE and face_available()

    def transcribe(self, audio_bytes: bytes, filename: str = "speech.wav") -> str:
        return self.llm.transcribe(audio_bytes, filename)

    def analyze_answer(self, question, transcript, audio_bytes=None, image_bytes=None,
                       face=None, role="", session_id="web", duration=None) -> dict:
        if duration is None:
            duration = wav_duration(audio_bytes) if audio_bytes else None
        metrics = analyze_speech(transcript, duration)
        if face is None and image_bytes and self.face_enabled():
            face = analyze_face(image_bytes)   # photo fallback
        return self.orch.run(question, transcript, metrics, face=face, role=role, session_id=session_id)

    def feedback(self, attempt_id: int, value: int) -> None:
        self.memory.add_feedback(attempt_id, value)
