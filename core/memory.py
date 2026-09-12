"""SQLite store of coaching attempts — enables progress tracking and dataset export."""
from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

from core.config import DB_PATH

_SCHEMA = """
CREATE TABLE IF NOT EXISTS attempts (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    ts             REAL,
    session_id     TEXT,
    question       TEXT,
    transcript     TEXT,
    content_score  INTEGER,
    delivery_score INTEGER,
    overall_score  INTEGER,
    fillers        INTEGER,
    wpm            INTEGER,
    confidence     INTEGER,
    emotion        TEXT,
    feedback       TEXT,
    thumbs         INTEGER
);
"""


class MemoryStore:
    def __init__(self, path: str = DB_PATH):
        self.path = path
        parent = Path(path).parent
        if str(parent) not in ("", "."):
            parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(_SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def log_attempt(self, **row) -> int:
        row.setdefault("ts", time.time())
        row.setdefault("thumbs", None)
        cols = ("ts", "session_id", "question", "transcript", "content_score",
                "delivery_score", "overall_score", "fillers", "wpm", "confidence",
                "emotion", "feedback", "thumbs")
        vals = [row.get(c) for c in cols]
        with self._conn() as conn:
            cur = conn.execute(
                f"INSERT INTO attempts ({','.join(cols)}) VALUES ({','.join('?'*len(cols))})", vals)
            return cur.lastrowid

    def add_feedback(self, attempt_id: int, value: int) -> None:
        with self._conn() as conn:
            conn.execute("UPDATE attempts SET thumbs=? WHERE id=?", (value, attempt_id))

    def recent(self, session_id: str, n: int = 10) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM attempts WHERE session_id=? ORDER BY id DESC LIMIT ?",
                                (session_id, n)).fetchall()
        return [dict(r) for r in reversed(rows)]

    def score_trajectory(self, session_id: str, n: int = 30) -> list[int]:
        return [r["overall_score"] for r in self.recent(session_id, n)]

    def stats(self, session_id: str | None = None) -> dict:
        where, args = ("WHERE session_id=?", (session_id,)) if session_id else ("", ())
        with self._conn() as conn:
            row = conn.execute(
                f"SELECT COUNT(*) c, AVG(overall_score) o, AVG(content_score) ct, "
                f"AVG(delivery_score) d, MAX(overall_score) b FROM attempts {where}", args).fetchone()
        return {"attempts": row["c"] or 0,
                "avg_overall": round(row["o"]) if row["o"] is not None else 0,
                "avg_content": round(row["ct"]) if row["ct"] is not None else 0,
                "avg_delivery": round(row["d"]) if row["d"] is not None else 0,
                "best": row["b"] or 0}

    def export_jsonl(self, out_path: str) -> int:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM attempts ORDER BY id").fetchall()
        with open(out_path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps({
                    "question": r["question"], "transcript": r["transcript"],
                    "scores": {"content": r["content_score"], "delivery": r["delivery_score"],
                               "overall": r["overall_score"]},
                    "feedback": r["feedback"], "thumbs": r["thumbs"]}, ensure_ascii=False) + "\n")
        return len(rows)