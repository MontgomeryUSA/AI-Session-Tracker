"""
rag_chat.py
===========
chat_session.py's brain with the input() loop taken out, and with the HTTP calls
taken out too. The model now lives in this process (local_llm), so a "turn" is
just function calls -- no server, no socket, no timeout to tune.

The three-step turn is unchanged, and it's still the reason this is more than a
loop around recall():

  1. CONDENSE  "what was in the letter?" has no searchable content on its own --
               the subject lives in the previous turn -- so rewrite it into a
               standalone query first
  2. RETRIEVE  embed that query, KNN over vec_chunks, scoped to one patient
  3. ANSWER    hand the excerpts plus the running history to the chat model

A turn is a generator: it yields the sources first, then tokens, so the window
can paint the evidence panel while the model is still writing.
"""
from __future__ import annotations

from typing import Optional

import config
import local_llm
from session_store import _ts, recall

SYSTEM = (
    "You are a clinical documentation assistant. Answer the user's questions "
    "about this patient using ONLY the session excerpts provided with each "
    "question. Cite the session id and timestamp (mm:ss) for facts you state. "
    "If the excerpts don't contain the answer, say so plainly -- do not invent "
    "details. You may use earlier turns of this conversation for context."
)


# ---- 1. condense ------------------------------------------------------------
def condense(history: list[dict], question: str) -> str:
    """history is lean Q&A only -- never the injected excerpts, or the prompt
    bloats and the rewrite gets worse."""
    if not history:
        return question

    convo = "\n".join(f"{m['role']}: {m['content']}" for m in history[-6:])
    prompt = (
        "Given the conversation below, rewrite the FOLLOW-UP into a single "
        "standalone search query that includes the entities it refers to. "
        "Reply with the query only -- no preamble, no quotes.\n\n"
        f"{convo}\n\nFOLLOW-UP: {question}\n\nSTANDALONE QUERY:"
    )
    try:
        rewritten = local_llm.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.0, max_tokens=96,       # a query, not an essay
        ).strip().strip('"')
    except Exception:
        return question           # a bad rewrite is worse than no rewrite

    if not rewritten or len(rewritten) > 300:
        return question           # the model explained itself instead of answering
    return rewritten


# ---- 2. retrieve ------------------------------------------------------------
def _format_context(hits) -> str:
    return "\n\n".join(
        f"(session {sid}, {_ts(st)}-{_ts(en)})\n{txt}"
        for sid, cid, dist, st, en, txt in hits
    )


def hits_to_sources(hits) -> list[dict]:
    return [
        {
            "session_id": sid, "chunk_id": cid, "distance": float(dist),
            "start": float(st), "end": float(en),
            "start_ts": _ts(st), "end_ts": _ts(en), "text": txt,
        }
        for sid, cid, dist, st, en, txt in hits
    ]


# ---- 3. answer --------------------------------------------------------------
def _messages(history: list[dict], question: str, hits) -> list[dict]:
    current = {
        "role": "user",
        "content": f"Session excerpts:\n{_format_context(hits)}\n\nQuestion: {question}",
    }
    return [{"role": "system", "content": SYSTEM}] + history + [current]


def turn(con, patient_id: int, question: str,
         history: Optional[list[dict]] = None, k: int = config.TOP_K,
         cancelled=lambda: False):
    """Yields ('sources', {...}) once, then ('token', str) repeatedly."""
    history = history or []

    search_q = condense(history, question)
    hits = recall(con, patient_id, search_q, k)
    yield "sources", {"sources": hits_to_sources(hits), "search_query": search_q}

    for piece in local_llm.chat_stream(
        _messages(history, question, hits), cancelled=cancelled
    ):
        yield "token", piece


def ask(con, patient_id: int, question: str,
        history: Optional[list[dict]] = None, k: int = config.TOP_K) -> dict:
    """Blocking convenience wrapper, for scripts and tests."""
    sources, answer = {}, []
    for kind, payload in turn(con, patient_id, question, history, k):
        if kind == "sources":
            sources = payload
        else:
            answer.append(payload)
    return {"answer": "".join(answer), **sources}
