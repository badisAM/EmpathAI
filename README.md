# EmpathAI — multi-agent, multimodal, emotion-aware assistant

A support assistant that infers the user's emotional state from **text and face**,
routes each turn through a **team of specialist agents**, grounds its answers in a
**knowledge base (RAG)**, **logs every interaction** to a database, and **adapts its
tone** to how the user actually feels.

Built on the free, OpenAI-compatible **Groq** API, with a FastAPI backend and a
Streamlit frontend that are ready to deploy.

## What makes it more than a single-LLM demo

| Capability | How it shows up |
|---|---|
| Multi-agent orchestration | A supervisor routes each turn across Safety, Emotion-analyst, Retrieval, Wellbeing, and Responder agents, and records the handoff path. |
| Diversified tools | Knowledge search (RAG), grounding techniques, crisis routing — chosen per turn, not always-on. |
| RAG | Retrieval over a markdown knowledge base (TF-IDF by default, swappable to MiniLM vector embeddings). |
| Persistent memory | Every turn is stored in SQLite with emotion labels and feedback; exportable as JSONL for evaluation or fine-tuning. |
| Multimodal input | Optional webcam facial-emotion reading (DeepFace), fused with the text signal. |
| Deployable | FastAPI service (Dockerfile + render.yaml) with a decoupled frontend. |

## The differentiator: multimodal congruence

EmpathAI compares what you **say** with what your **face** shows. When they agree it
sharpens the read; when they disagree — you type "I'm fine" but look sad — it flags the
mismatch and gently checks in instead of taking the words at face value. That check is
the interesting, demo-friendly moment few portfolio projects have.

## Architecture

```
                         ┌─────────────────┐
   message  ───────────► │  Orchestrator   │  (supervisor: routes + records path)
   webcam frame ───┐     └────────┬────────┘
                   │              │
        ┌──────────▼───┐   1 ┌────▼────────┐  crisis?  ┌──────────────┐
        │  Vision      │────►│ SafetyAgent │──────────►│ crisis reply │
        │ (DeepFace)   │     └────┬────────┘           └──────────────┘
        └──────────────┘        2 │
                        ┌──────────▼──────────┐  fuse(text, face)
                        │  EmotionAnalyst     │  → emotion, intensity, congruence
                        └──────────┬──────────┘
                     3 route       │
              ┌──────────┬─────────┴───────┐
        ┌─────▼─────┐ ┌──▼────────┐        │
        │ Retrieval │ │ Wellbeing │        │
        │  (RAG)    │ │ grounding │        │
        └─────┬─────┘ └──┬────────┘        │
              └──────┬────┴─────────────────┘
                4 ┌──▼──────────┐  tone/length adapted to emotion
                  │ Responder   │──► reply
                  └──┬──────────┘
                5    │
              ┌──────▼──────┐
              │ MemoryStore │  SQLite log → JSONL dataset
              └─────────────┘
```

The agents are plain Python classes with clear responsibilities, so the whole flow is
readable end to end. The design maps directly onto LangGraph nodes/edges if you want to
port it to a graph runtime.

## Project structure

```
core/        llm, emotions+fusion, rag, memory, vision, agents, pipeline
backend/     FastAPI app (main.py)
frontend/    Streamlit UI (streamlit_app.py)
knowledge/   markdown docs indexed by the RAG retriever
scripts/     export_dataset.py
tests/       test_offline.py  (runs with a fake LLM, no key needed)
```

## Setup

```bash
git clone <your-repo-url> && cd empathai
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                   # add your free Groq key(s)
```

Get free Groq keys (no credit card) at https://console.groq.com. Add several,
comma-separated, in `GROQ_API_KEYS` — they rotate automatically on rate limits.

> `.env`, `keys.local.txt`, and `*.db` are git-ignored. Never commit secrets.

## Run

```bash
python -m tests.test_offline                          # verify the pipeline (no key)
streamlit run frontend/streamlit_app.py               # web UI  → localhost:8501
uvicorn backend.main:app --reload --port 8000         # API     → localhost:8000/docs
python -m scripts.export_dataset dataset.jsonl        # export logged turns
```

Call the API directly:

```bash
curl -X POST localhost:8000/chat -H "Content-Type: application/json" \
  -d '{"message":"how do I get a refund? this is the third time it failed"}'
```

### Optional camera modality

```bash
pip install -r requirements-face.txt
# set ENABLE_FACE=1 in .env, then restart the frontend
```

A camera panel appears in the sidebar; the captured frame is analysed with DeepFace
and fused with the text emotion. Without these deps the app runs on text alone.

## Data and re-training

Every turn is stored in SQLite (`empathai.db`): message, text/face/fused emotion,
intensity, congruence, agent path, tools, retrieved sources, response, and 👍/👎
feedback. `scripts/export_dataset.py` dumps it as chat-style JSONL — a labeled dataset
you can use to evaluate the classifier or fine-tune a smaller model.

## Deployment

The backend and frontend are decoupled, so you can host them separately.

Backend (FastAPI) — Render, from `render.yaml`:
1. Push the repo to GitHub.
2. Render → New → Blueprint → pick the repo. It reads `render.yaml`.
3. Set `GROQ_API_KEYS` in the Render dashboard (never in git).
4. Deploy — your API is live at `https://empathai-api.onrender.com`.

Frontend (Streamlit): easiest on Streamlit Community Cloud (point it at
`frontend/streamlit_app.py`), or as a second Render service.

Vercel: Vercel is built for JavaScript frontends and light serverless functions, so a
long-running Streamlit/TensorFlow process does not fit it. The clean split is to deploy
the API on Render and, if you want Vercel specifically, build a small Next.js/React
frontend there that calls the Render API (`/chat`). The API already sends permissive
CORS headers for exactly this. The frontend is deliberately thin to keep that door open.

## Push to GitHub

```bash
git init
git add .
git status                         # confirm no .env / *.db is staged
git commit -m "feat: multi-agent multimodal emotion-aware assistant (RAG + memory)"
git branch -M main
git remote add origin https://github.com/<you>/empathai.git
git push -u origin main
```

Commit in stages (core → agents → backend → frontend → docs) rather than one big
commit — reviewers look at the history.

## Responsible AI and limitations

- Not a mental-health tool. Self-harm signals are routed to crisis resources and skip
  the normal flow.
- Emotion inference (text and face) is a heuristic, not a validated instrument; it can
  be wrong, especially on sarcasm, mixed emotions, or poor lighting.
- Facial analysis runs only with explicit opt-in (`ENABLE_FACE=1`) and processes a
  single captured frame; nothing is sent anywhere except your own Groq LLM call.

## Roadmap

- Port the orchestrator to LangGraph for checkpointing and human-in-the-loop.
- Swap TF-IDF for MiniLM embeddings (`RAG_BACKEND=fastembed`).
- Add an evaluation harness measuring classifier accuracy against the exported dataset.
- Weight the text/face fusion from feedback data.

## License

MIT
