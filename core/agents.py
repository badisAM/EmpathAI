"""Reasoning-driven coaching pipeline (agentic, non-robotic).

Flow:
  Diagnostician  - LLM reads the answer + speech metrics + emotion timeline, reasons
                   about what happened, and DECIDES: retrieval queries + strategies.
  Dynamic RAG    - runs the LLM-generated queries (adapts to the user, not the question).
  Strategist     - the chosen strategies become the coaching focus.
  Scorer         - LLM scores dimensions WITH reasoning over metrics + emotion + context.
  Coach          - tailored feedback grounded in the chosen strategies + retrieved chunks.
  Rewrite        - model answer.
"""
from __future__ import annotations

import json
import re

from core.rag import Retriever

LANG_RULE = "Write ALL prose output in the same language as the candidate's answer.\n"

# Menu of coaching strategies the Diagnostician can pick from (dynamic tool selection).
STRATEGIES = {
    "relevance": "Answer the actual question directly in the first sentence.",
    "star": "Restructure with STAR: Situation, Task, Action, Result.",
    "quantify": "Add concrete numbers, outcomes, or names.",
    "concise": "Cut rambling; lead with the main point.",
    "fillers": "Replace filler words with short silent pauses.",
    "composure": "Steady nerves: slow down, breathe, pause before answering.",
    "story": "Use one specific real example instead of generalities.",
}

FACE_CONF = {"joy": .85, "neutral": .70, "surprise": .60,
             "anger": .40, "frustration": .40, "sadness": .30, "fear": .25}


def _json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError("no json")
    return json.loads(m.group(0))


def clamp(x: float, lo: float = 0, hi: float = 100) -> float:
    return max(lo, min(hi, x))


def verdict_for(s: float) -> str:
    return "Needs work" if s < 50 else "Promising" if s < 70 else "Strong" if s < 85 else "Excellent"


def _emotion_text(face: dict | None) -> str:
    if not face or not face.get("timeline"):
        return "No facial data."
    tl = face["timeline"]
    pick = [tl[0], tl[len(tl) // 2], tl[-1]] if len(tl) >= 3 else tl
    seq = " → ".join(f"{e}({i:.1f})" for e, i in pick)
    return f"Facial emotion over time: {seq}. Dominant: {face.get('emotion')}."


class Diagnostician:
    """One reasoning call: understand the answer, decide queries + strategies."""
    def __init__(self, llm):
        self.llm = llm

    def diagnose(self, question, transcript, metrics, face, role) -> dict:
        menu = ", ".join(STRATEGIES)
        sys = (LANG_RULE +
               "You are an expert interview coach REASONING about one answer before scoring it. "
               "Think about whether it addressed the question, its main weaknesses, and how the "
               "candidate's emotion evolved. Then decide what to look up and how to help.\n"
               f"Available strategy keys: {menu}.\n"
               'Respond ONLY with JSON: {"addressed":true/false,'
               '"summary":"1 sentence on what they actually said",'
               '"emotion_reading":"1 sentence interpreting the emotion timeline",'
               '"weaknesses":["short","short"],'
               '"queries":["retrieval query tied to a weakness","another"],'
               '"strategies":["key","key"]}  (2-3 items in lists).')
        u = (f"Question: {question}\nRole: {role or 'unspecified'}\nAnswer: {transcript}\n"
             f"Speech metrics: {metrics}\n{_emotion_text(face)}")
        try:
            d = _json(self.llm.chat([{"role": "system", "content": sys},
                                     {"role": "user", "content": u}], temperature=0.3))
            d["strategies"] = [k for k in d.get("strategies", []) if k in STRATEGIES][:3] or ["relevance"]
            d.setdefault("queries", [question])
            return d
        except Exception:
            return {"addressed": True, "summary": "", "emotion_reading": "",
                    "weaknesses": [], "queries": [question], "strategies": ["relevance", "quantify"]}


class Scorer:
    """LLM scoring WITH reasoning over metrics, emotion, and retrieved context."""
    def __init__(self, llm):
        self.llm = llm

    def score(self, question, transcript, metrics, face, diag, context) -> dict:
        ctx = "\n\n".join(f"[{c['source']}] {c['text']}" for c in context) or "None."
        sys = (LANG_RULE +
               "Score this answer strictly and with reasoning. Off-topic/incoherent → relevance & "
               "specificity below 25. Judge CONFIDENCE from the words AND the emotion timeline "
               "(nervous emotions or hesitant wording lower it). Use the reference material for structure.\n"
               f"Diagnosis: {diag.get('summary')} | Emotion: {diag.get('emotion_reading')}\n"
               f"Reference:\n{ctx}\n"
               'Respond ONLY with JSON: {"relevance":0-100,"structure":0-100,"specificity":0-100,'
               '"confidence":0-100,"energy":0-100}.')
        u = f"Question: {question}\nAnswer: {transcript}\nMetrics: {metrics}\n{_emotion_text(face)}"
        try:
            d = _json(self.llm.chat([{"role": "system", "content": sys},
                                     {"role": "user", "content": u}], temperature=0.0))
            out = {k: round(clamp(float(d.get(k, 50)))) for k in
                   ("relevance", "structure", "specificity", "confidence", "energy")}
        except Exception:
            out = {"relevance": 45, "structure": 45, "specificity": 40, "confidence": 50, "energy": 50}

        if face:  # nudge confidence with the aggregated facial read
            out["confidence"] = round(clamp(0.7 * out["confidence"] +
                                            0.3 * FACE_CONF.get(face.get("emotion", "neutral"), .6) * 100))
        if metrics.get("words", 0) < 15:
            out["confidence"] = min(out["confidence"], 35)
        return out


class Coach:
    def __init__(self, llm):
        self.llm = llm

    def feedback(self, question, transcript, diag, scores, metrics, context) -> str:
        ctx = "\n\n".join(f"[{c['source']}] {c['text']}" for c in context) or "None."
        strat = "\n".join(f"- {STRATEGIES[k]}" for k in diag.get("strategies", []))
        sys = (LANG_RULE +
               "You are a warm, direct coach speaking to the candidate as 'you'. React to what they "
               "ACTUALLY said (quote a short phrase) and to how they seemed (emotion). If they missed "
               "the question, say it kindly first. Then coach using ONLY these chosen strategies, made "
               "specific to them, each with an example phrasing. Sound human, short paragraphs.\n"
               f"Chosen strategies:\n{strat}\nReference:\n{ctx}")
        u = (f"Question: {question}\nAnswer: {transcript}\n"
             f"Diagnosis: {diag}\nScores: {scores}\nMetrics: {metrics}")
        return self.llm.chat([{"role": "system", "content": sys}, {"role": "user", "content": u}], temperature=0.6)

    def model_answer(self, question, transcript, diag, context) -> str:
        ctx = "\n\n".join(f"[{c['source']}] {c['text']}" for c in context) or "None."
        sys = (LANG_RULE +
               "Write a strong MODEL ANSWER the candidate could say aloud, applying the chosen strategies. "
               "Reuse their real facts; if the answer was empty/off-topic, invent a brief realistic example "
               "clearly tied to the question. Concise, natural, well-structured.\n"
               f"Strategies: {[STRATEGIES[k] for k in diag.get('strategies', [])]}\nReference:\n{ctx}")
        u = f"Question: {question}\nTheir answer: {transcript}"
        return self.llm.chat([{"role": "system", "content": sys}, {"role": "user", "content": u}], temperature=0.5)


class Coachestrator:
    def __init__(self, llm, retriever: Retriever, memory):
        self.diag = Diagnostician(llm)
        self.scorer = Scorer(llm)
        self.coach = Coach(llm)
        self.retriever = retriever
        self.memory = memory

    def _dynamic_retrieve(self, queries: list[str]) -> list[dict]:
        seen, out = set(), []
        for q in queries[:3]:
            for c in self.retriever.search(q, k=2):
                if c["id"] not in seen:
                    seen.add(c["id"])
                    out.append(c)
        return out[:4]

    def run(self, question, transcript, metrics, face=None, role="", session_id="web") -> dict:
        path = ["orchestrator", "speech", "diagnostician"]
        diag = self.diag.diagnose(question, transcript, metrics, face, role)

        path.append("retriever")
        context = self._dynamic_retrieve(diag.get("queries", [question]))
        path.append("strategist")  # strategies chosen inside the diagnosis

        path.append("scorer")
        sc = self.scorer.score(question, transcript, metrics, face, diag, context)

        # deterministic delivery facts + reasoned confidence
        fillers = metrics.get("fillers", 0)
        penalty = min(30, fillers * 3)
        wpm, pace = metrics.get("wpm"), "good"
        if wpm:
            if wpm > 170:
                pace, penalty = "a bit fast", penalty + 5
            elif wpm < 110:
                pace, penalty = "a bit slow", penalty + 5
        delivery = {"confidence": sc["confidence"], "energy": sc["energy"], "pace": pace,
                    "pace_score": 100 if pace == "good" else 65,
                    "filler_score": round(clamp(100 - fillers * 8)),
                    "delivery_score": round(clamp(sc["confidence"] - penalty))}

        content_score = round((sc["relevance"] + sc["structure"] + sc["specificity"]) / 3)
        overall = round(0.5 * content_score + 0.5 * delivery["delivery_score"])

        path.append("coach")
        fb = self.coach.feedback(question, transcript, diag, sc, metrics, context)
        path.append("rewrite")
        model = self.coach.model_answer(question, transcript, diag, context)

        dimensions = {"Relevance": sc["relevance"], "Structure": sc["structure"],
                      "Specificity": sc["specificity"], "Confidence": delivery["confidence"],
                      "Pace": delivery["pace_score"], "Filler control": delivery["filler_score"]}

        iid = self.memory.log_attempt(
            session_id=session_id, question=question, transcript=transcript,
            content_score=content_score, delivery_score=delivery["delivery_score"],
            overall_score=overall, fillers=fillers, wpm=wpm,
            confidence=delivery["confidence"], emotion=(face or {}).get("emotion"), feedback=fb)

        return {"id": iid, "question": question, "transcript": transcript,
                "scores": {"content": content_score, "delivery": delivery["delivery_score"], "overall": overall},
                "verdict": verdict_for(overall), "dimensions": dimensions,
                "diagnosis": diag, "content": {"notes": diag.get("summary", "")},
                "delivery": delivery, "metrics": metrics, "face": face,
                "strategies": [STRATEGIES[k] for k in diag.get("strategies", [])],
                "feedback": fb, "model_answer": model, "agent_path": path,
                "sources": [{"id": c["id"], "source": c["source"], "score": c["score"]} for c in context]}