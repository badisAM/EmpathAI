"""Offline tests for the coaching pipeline (fake LLM, no API key).
Run:  python -m tests.test_offline
"""
import json
import os
import tempfile

from core.agents import Coachestrator
from core.memory import MemoryStore
from core.rag import Retriever, load_knowledge
from core.speech import analyze_speech


class FakeLLM:
    def chat(self, messages, temperature=0.4, **kwargs):
        sys = messages[0]["content"]
        if "CONTENT of an interview" in sys:
            return json.dumps({"relevance": 80, "structure": 60, "specificity": 70,
                               "notes": "Good relevance, structure could use STAR."})
        if "rate spoken DELIVERY" in sys:
            return json.dumps({"confidence": 72, "energy": 65, "notes": "Steady, clear."})
        return "Strengths: clear and relevant. Improvements: use STAR, cut filler words."


def test_speech():
    m = analyze_speech("Um, so basically I, uh, led the project you know", duration=6.0)
    assert m["fillers"] >= 4 and m["words"] > 0 and m["wpm"] is not None
    print(f"speech .......... ok (fillers={m['fillers']}, wpm={m['wpm']})")


def test_rag():
    r = Retriever(load_knowledge("knowledge"))
    hits = r.search("behavioural question tell me about a time", k=2)
    assert hits and "star" in hits[0]["text"].lower()
    print(f"rag ............. ok (top score {hits[0]['score']})")


def test_pipeline():
    with tempfile.TemporaryDirectory() as d:
        mem = MemoryStore(os.path.join(d, "t.db"))
        orch = Coachestrator(FakeLLM(), Retriever(load_knowledge("knowledge")), mem)
        metrics = analyze_speech("Um so basically I led the team and, uh, we shipped it", 8.0)
        res = orch.run("Tell me about a time you led a project.",
                       "Um so basically I led the team and we shipped it", metrics,
                       face={"emotion": "fear", "intensity": .7}, session_id="s")
        assert "content_analyst" in res["agent_path"] and "coach" in res["agent_path"]
        assert 0 <= res["scores"]["overall"] <= 100
        assert res["delivery"]["confidence"] < 80  # nervous face lowers confidence
        assert res["sources"]  # RAG grounded
        st = mem.stats()
        assert st["attempts"] == 1 and 0 <= st["avg_overall"] <= 100
        out = os.path.join(d, "o.jsonl")
        assert mem.export_jsonl(out) == 1
        rec = json.loads(open(out).readline())
        assert "scores" in rec and rec["question"].startswith("Tell me")
    print(f"pipeline ........ ok (overall={res['scores']['overall']}, "
          f"content={res['scores']['content']}, delivery={res['scores']['delivery']})")


if __name__ == "__main__":
    test_speech()
    test_rag()
    test_pipeline()
    print("\nALL OFFLINE TESTS PASSED")