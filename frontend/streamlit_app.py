"""EmpathAI Coach — professional, client-ready UI for an AI interview & pitch coach.

Guided 3-step flow, refined product styling, a spoken FACE-STATE read (calm /
nervous / positive…), scorecard, tailored coaching, and a model answer.

Run:  streamlit run frontend/streamlit_app.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.emotions import EMOTION_COLORS  # noqa: E402
from core.pipeline import EmpathAI  # noqa: E402
from core import vision  # noqa: E402

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False

try:
    import av  # noqa: F401
    from streamlit_webrtc import VideoProcessorBase, webrtc_streamer
    HAS_WEBRTC = True
except Exception:
    HAS_WEBRTC = False

st.set_page_config(page_title="EmpathAI Coach", page_icon="◐", layout="centered")

QUESTIONS = [
    "Tell me about yourself.", "Tell me about a time you led a project.",
    "What is your greatest weakness?", "Why do you want to work here?",
    "Give your 60-second startup pitch.", "Describe a conflict you resolved at work.",
    "Custom question…",
]
STEPS = [
    ("orchestrator", "Orchestrator"), ("diagnostician", "Diagnostician"),
    ("retriever", "Dynamic RAG"), ("strategist", "Strategist"),
    ("scorer", "Scorer"), ("coach", "Coach"), ("rewrite", "Rewrite"),
]
FACE_WORDS = {
    "joy": "positive and engaged", "neutral": "calm and composed",
    "surprise": "animated and reactive", "fear": "nervous and tense",
    "sadness": "subdued or unsure", "anger": "tense", "frustration": "tense or impatient",
}
NEG = {"fear", "sadness", "anger", "frustration"}


def score_color(s: float) -> str:
    return "#DC6068" if s < 50 else "#E0913E" if s < 70 else "#C9A227" if s < 85 else "#3DA47C"


def highlight_fillers(t: str, breakdown: dict) -> str:
    for f in sorted(breakdown, key=len, reverse=True):
        t = re.sub(r"(?i)\b" + re.escape(f) + r"\b", f"<mark>{f}</mark>", t)
    return t


def describe_face(face: dict) -> str:
    base = FACE_WORDS.get(face.get("emotion", "neutral"), "steady")
    tl = face.get("timeline", [])
    trend = ""
    if len(tl) >= 4:
        half = len(tl) // 2
        n0 = sum(1 for e, _ in tl[:half] if e in NEG) / max(1, half)
        n1 = sum(1 for e, _ in tl[half:] if e in NEG) / max(1, len(tl) - half)
        if n1 < n0 - 0.2:
            trend = " You settled and grew more comfortable as you went."
        elif n1 > n0 + 0.2:
            trend = " You appeared to tense up toward the end."
        else:
            trend = " Your composure stayed steady throughout."
    return f"On camera you came across as {base}.{trend}"


_ProcBase = VideoProcessorBase if HAS_WEBRTC else object


class FaceProcessor(_ProcBase):
    def __init__(self):
        self.samples, self._i, self.latest = [], 0, "—"

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self._i += 1
        if self._i % 15 == 0:
            s = vision.analyze_frame(img)
            if s:
                self.samples.append(s)
                self.latest = s["emotion"]
        return av.VideoFrame.from_ndarray(img, format="bgr24")


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family:'Inter',sans-serif; }
.stApp { background:#F7F8FB; }
#MainMenu, footer, [data-testid="stToolbar"] { visibility:hidden; }
.block-container { padding-top:1.4rem; max-width:880px; }

.topbar { display:flex; align-items:center; gap:.7rem; padding-bottom:.5rem; }
.logo { width:34px; height:34px; border-radius:10px;
  background:linear-gradient(135deg,#4F46E5,#7C6CF0); box-shadow:0 4px 12px rgba(79,70,229,.28); }
.brandname { font-size:1.15rem; font-weight:700; color:#1B1F27; }
.pill { font-size:.66rem; font-weight:600; color:#4F46E5; background:#EEF0FE; padding:.12rem .5rem; border-radius:999px; }
.hero { font-size:1.05rem; color:#4B5563; margin:.3rem 0 1rem; }
.rule { height:2px; border-radius:2px; margin:.2rem 0 1.4rem;
  background:linear-gradient(90deg,#4F46E5,#7C6CF0,transparent); opacity:.5; }

.section { font-size:.72rem; font-weight:600; letter-spacing:.07em; text-transform:uppercase; color:#9099A5; margin:.2rem 0 .5rem; }
.card { background:#FFF; border:1px solid #ECEEF2; border-radius:16px; padding:1.15rem 1.25rem; margin:.55rem 0;
  box-shadow:0 1px 2px rgba(16,24,40,.04), 0 10px 30px rgba(16,24,40,.03); }
.stepnum { display:inline-flex; align-items:center; justify-content:center; width:22px; height:22px; border-radius:50%;
  background:#4F46E5; color:#FFF; font-size:.72rem; font-weight:700; margin-right:.55rem; }
.steplbl { font-size:1rem; font-weight:600; color:#20242C; }
.hint { color:#8A909C; font-size:.82rem; margin:.15rem 0 .5rem 2rem; }

.scoreflex { display:flex; align-items:center; gap:1.2rem; }
.bignum { font-size:2.4rem; font-weight:700; line-height:1; }
.verdict { display:inline-block; font-size:.78rem; font-weight:600; padding:.18rem .6rem; border-radius:999px; color:#FFF; }
.dimrow { display:flex; justify-content:space-between; font-size:.83rem; color:#3A3F49; margin:.5rem 0 .16rem; }
.bar { height:7px; border-radius:5px; background:#EEF0F3; overflow:hidden; }
.barfill { height:100%; border-radius:5px; }
.faceline { font-size:1rem; font-weight:600; color:#20242C; }
.chip { display:inline-block; font-size:.72rem; padding:.16rem .55rem; border-radius:999px; background:#F4F5F7;
  color:#4B5563; margin:.28rem .3rem 0 0; border:1px solid #EEF0F3; }
.spark { display:flex; align-items:flex-end; gap:3px; height:40px; margin-top:.5rem; }
.spark-bar { flex:1; border-radius:3px 3px 0 0; min-height:3px; }
.feedback { font-size:.93rem; line-height:1.68; color:#2A2F3A; white-space:pre-wrap; }
.model { background:#F6F7FF; border:1px solid #E6E9FB; border-radius:12px; padding:.9rem 1.05rem;
  font-size:.9rem; line-height:1.62; color:#31305F; white-space:pre-wrap; }
.focus { font-size:.9rem; color:#2A2F3A; line-height:1.7; }
.small { font-size:.78rem; color:#8A909C; }
.dot { display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:.3rem; }
mark { background:#FDF3D3; color:#8A6D1B; padding:0 .12rem; border-radius:3px; }
.src { background:#F7F8FA; border:1px solid #ECEEF2; border-radius:10px; padding:.5rem .7rem; font-size:.8rem;
  color:#4B5563; margin:.3rem 0; line-height:1.5; }
.stButton>button { border-radius:10px; font-weight:600; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_engine():
    return EmpathAI()


if "result" not in st.session_state:
    st.session_state.result = None
    st.session_state.error = None
    try:
        st.session_state.engine = get_engine()
    except Exception as exc:
        st.session_state.engine = None
        st.session_state.error = str(exc)

engine = st.session_state.get("engine")

# ---- brand header
st.markdown('<div class="topbar"><div class="logo"></div>'
            '<div class="brandname">EmpathAI&nbsp;Coach</div>'
            '<span class="pill">AI interview coach</span></div>', unsafe_allow_html=True)
st.markdown('<div class="hero">Rehearse an answer out loud and get an objective score, a read on your '
            'delivery and composure, and coaching you can act on — in your own language.</div>',
            unsafe_allow_html=True)
st.markdown('<div class="rule"></div>', unsafe_allow_html=True)

with st.expander("How it works", expanded=st.session_state.result is None):
    st.markdown(
        "1. **Pick a question** and, optionally, the role you're targeting.\n"
        "2. **Record your answer** with the mic. Enable the camera to also get a read on your composure.\n"
        "3. Press **Analyze** — your scorecard, face read, coaching, and a model answer appear below.\n\n"
        "Camera (optional): `pip install -r requirements-face.txt`, then set `ENABLE_FACE=1` in `.env`.")

if st.session_state.error:
    st.warning("No API key detected. Add your Groq key to `.env` (GROQ_API_KEYS), then reload.", icon="🔑")

# ---- input card
with st.container():
    st.markdown('<div><span class="stepnum">1</span><span class="steplbl">Question</span></div>',
                unsafe_allow_html=True)
    pick = st.selectbox("Question", QUESTIONS, label_visibility="collapsed")
    question = st.text_input("Your question", "") if pick == "Custom question…" else pick
    role = st.text_input("Target role (optional)", "", placeholder="e.g. Data Analyst")

    st.markdown('<div style="margin-top:.7rem"><span class="stepnum">2</span>'
                '<span class="steplbl">Record your answer</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="hint">Speak naturally for ~30–60s. Allow the mic when your browser asks.</div>',
                unsafe_allow_html=True)
    audio = st.audio_input("Record", label_visibility="collapsed")

    webrtc_ctx = None
    snapshot_bytes = None
    if engine and engine.face_enabled() and HAS_WEBRTC:
        st.markdown('<div class="hint">📹 Keep the video running while you answer for the composure read.</div>',
                    unsafe_allow_html=True)
        webrtc_ctx = webrtc_streamer(
            key="face", video_processor_factory=FaceProcessor,
            media_stream_constraints={"video": True, "audio": False},
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            async_processing=True)
    elif engine and engine.face_enabled():
        shot = st.camera_input("📷 Optional snapshot", label_visibility="visible")
        if shot is not None:
            snapshot_bytes = shot.getvalue()
    else:
        st.markdown('<div class="hint">💡 Enable the camera to add a facial composure read (see How it works).</div>',
                    unsafe_allow_html=True)

    st.markdown('<div style="margin-top:.7rem"><span class="stepnum">3</span>'
                '<span class="steplbl">Analyze</span></div>', unsafe_allow_html=True)
    if st.button("Analyze my answer", type="primary", disabled=engine is None, use_container_width=True):
        if not question.strip():
            st.warning("Choose or write a question first.")
        elif audio is None:
            st.warning("Record your answer first.")
        else:
            face = None
            if webrtc_ctx and webrtc_ctx.video_processor and webrtc_ctx.video_processor.samples:
                face = vision.aggregate(webrtc_ctx.video_processor.samples)
            data = audio.getvalue()
            with st.spinner("Transcribing and coaching…"):
                try:
                    transcript = engine.transcribe(data)
                    st.session_state.result = engine.analyze_answer(
                        question, transcript, audio_bytes=data, image_bytes=snapshot_bytes,
                        face=face, role=role, session_id="web")
                except Exception as exc:
                    st.error(f"Analysis failed: {exc}")

# ---- results
res = st.session_state.result
if not res:
    st.caption("Your results will appear here after you analyze an answer.")
else:
    st.markdown('<div class="rule" style="margin-top:1.2rem"></div>', unsafe_allow_html=True)
    ov = res["scores"]["overall"]

    st.markdown('<div class="section">Overall</div>', unsafe_allow_html=True)
    cA, cB = st.columns([1, 2])
    with cA:
        if HAS_PLOTLY:
            g = go.Figure(go.Indicator(mode="gauge+number", value=ov,
                          number={"font": {"size": 34, "color": score_color(ov)}},
                          gauge={"axis": {"range": [0, 100], "tickvals": [0, 50, 100]},
                                 "bar": {"color": score_color(ov)}, "bgcolor": "#F1F2F5", "borderwidth": 0}))
            g.update_layout(height=150, margin=dict(l=10, r=10, t=6, b=0),
                            paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter"))
            st.plotly_chart(g, use_container_width=True)
        else:
            st.markdown(f'<div class="bignum" style="color:{score_color(ov)}">{ov}<span style="font-size:1rem;'
                        f'color:#9AA0AB">/100</span></div>', unsafe_allow_html=True)
    with cB:
        st.markdown(f'<div style="margin-top:1.4rem"><span class="verdict" style="background:{score_color(ov)}">'
                    f'{res["verdict"]}</span></div>'
                    '<div class="card" style="margin-top:.6rem;box-shadow:none">', unsafe_allow_html=True)
        for label, val in res["dimensions"].items():
            st.markdown(f'<div class="dimrow"><span>{label}</span><span style="color:#8A909C">{val}</span></div>'
                        f'<div class="bar"><div class="barfill" style="width:{val}%;background:{score_color(val)}">'
                        f'</div></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ---- FACE READ (prominent, spoken)
    st.markdown('<div class="section">Face read</div>', unsafe_allow_html=True)
    face = res.get("face")
    if face:
        dist = face.get("distribution", {})
        tl = face.get("timeline", [])
        chips = "".join(f'<span class="chip"><span class="dot" style="background:'
                        f'{EMOTION_COLORS.get(e,"#8A94A6")}"></span>{e} {int(p*100)}%</span>'
                        for e, p in list(dist.items())[:4])
        spark = "".join(f'<div class="spark-bar" style="height:{max(i,.05)*100:.0f}%;'
                        f'background:{EMOTION_COLORS.get(e,"#8A94A6")}"></div>' for e, i in tl)
        st.markdown(f'<div class="card"><div class="faceline">{describe_face(face)}</div>'
                    f'<div style="margin-top:.5rem">{chips}</div>'
                    f'<div class="small" style="margin-top:.7rem">Composure over time '
                    f'({face.get("frames",0)} frames)</div><div class="spark">{spark}</div></div>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<div class="card"><div class="small">No facial data for this attempt. Enable the camera '
                    '(see “How it works”) to get a spoken read of your composure — calm, nervous, positive, '
                    'and how it changes as you speak.</div></div>', unsafe_allow_html=True)

    m, d = res["metrics"], res["delivery"]
    chips = [f'{m["words"]} words', f'{m["fillers"]} fillers',
             f'{m["wpm"]} wpm' if m["wpm"] else "wpm n/a", f'pace {d["pace"]}']
    st.markdown('<div>' + "".join(f'<span class="chip">{c}</span>' for c in chips) + '</div>',
                unsafe_allow_html=True)

    if res.get("strategies"):
        st.markdown('<div class="section" style="margin-top:1rem">Focus for you</div>', unsafe_allow_html=True)
        st.markdown('<div class="card"><div class="focus">' +
                    "".join(f'• {s}<br>' for s in res["strategies"]) + '</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section" style="margin-top:1rem">Coach feedback</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="card"><div class="feedback">{res["feedback"]}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section" style="margin-top:1rem">Model answer</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="card"><div class="model">{res["model_answer"]}</div></div>', unsafe_allow_html=True)

    b1, b2, _ = st.columns([1, 1, 3])
    if b1.button("👍 Helpful"):
        engine.feedback(res["id"], 1); st.toast("Thanks!")
    if b2.button("👎 Not helpful"):
        engine.feedback(res["id"], 0); st.toast("Noted.")

    with st.expander("Details · how the coach reached this"):
        lit = " ".join(
            f'<span class="dot" style="background:{"#4F46E5" if k in res["agent_path"] else "#D7DAE0"}"></span>{lbl}'
            for k, lbl in STEPS)
        st.markdown(f'<div class="small">Agents used: {lit}</div>', unsafe_allow_html=True)
        srcs = "".join(f'<div class="src">📄 {s["source"]} · score {s["score"]}</div>' for s in res["sources"])
        if srcs:
            st.markdown(srcs, unsafe_allow_html=True)
        st.markdown('<div class="small" style="margin-top:.5rem">Transcript (fillers highlighted)</div>',
                    unsafe_allow_html=True)
        st.markdown(f'<div class="src">{highlight_fillers(res["transcript"], m["filler_breakdown"])}</div>',
                    unsafe_allow_html=True)

    traj = engine.memory.score_trajectory("web", 30) if engine else []
    if traj and len(traj) > 1 and HAS_PLOTLY:
        with st.expander("Your progress"):
            xs = list(range(1, len(traj) + 1))
            fig = go.Figure(go.Scatter(x=xs, y=traj, mode="lines+markers",
                            line=dict(width=2.5, color="#4F46E5"),
                            marker=dict(size=8, color=[score_color(v) for v in traj],
                                        line=dict(width=1.5, color="#FFF"))))
            fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10),
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              font=dict(color="#6B7280", family="Inter"),
                              yaxis=dict(range=[0, 100], gridcolor="#ECEEF1"),
                              xaxis=dict(gridcolor="#ECEEF1", title="attempt"))
            st.plotly_chart(fig, use_container_width=True)