"""Central configuration and feature flags, all driven by environment variables."""
import os

from dotenv import load_dotenv

load_dotenv()

GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_STT_MODEL = os.getenv("GROQ_STT_MODEL", "whisper-large-v3-turbo")  # voice input
DB_PATH = os.getenv("DB_PATH", "empathai.db")
KNOWLEDGE_DIR = os.getenv("KNOWLEDGE_DIR", "knowledge")

# Optional facial-emotion analysis (needs the heavy deps in requirements-face.txt)
ENABLE_FACE = os.getenv("ENABLE_FACE", "0") == "1"

# Retrieval backend: "tfidf" (default, zero-download) or "fastembed" (vector embeddings)
RAG_BACKEND = os.getenv("RAG_BACKEND", "tfidf")
RAG_MIN_SCORE = float(os.getenv("RAG_MIN_SCORE", "0.05"))