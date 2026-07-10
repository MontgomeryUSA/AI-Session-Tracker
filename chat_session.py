"""
chat_session.py
Conversational chat over one patient's encrypted session history.

Why this is more than a loop around recall():
  A follow-up like "what was in the letter?" has no searchable content on its
  own -- the subject ("the letter to his mother in Italy") lives in the PREVIOUS
  turn. So every turn does:

    1. CONDENSE : rewrite the follow-up into a standalone search query using the
                  conversation so far          ("what was in the letter?"  ->
                  "what was in the letter the man wrote to his mother in Italy?")
    2. RETRIEVE : embed that standalone query, KNN search scoped to the patient
    3. ANSWER   : feed retrieved excerpts + the running conversation to the chat
                  model, so it replies naturally AND stays grounded in the data

Run:
    python chat_session.py
    you> what was said about italy?
    ai > ...
    you> what was in the letter?      <- follow-up resolves via condense
"""

import sys
import requests

from session_store import connect, recall, OLLAMA, CHAT_MODEL, _ts

# ---- config -----------------------------------------------------------------
DB_PATH    = "sql_db/clinic.db"
PASSPHRASE = "password"
PATIENT_ID = 1
K          = 5
SHOW_SOURCES = True

SYSTEM = (
    "You are a clinical documentation assistant. Answer the user's questions "
    "about this patient using ONLY the session excerpts provided with each "
    "question. Cite the session id and timestamp (mm:ss) for facts you state. "
    "If the excerpts don't contain the answer, say so plainly -- do not invent "
    "details. You may use earlier turns of this conversation for context."
)


# ---- 1. condense a follow-up into a standalone search query ------------------
def condense(history, question):
    """history: list of {'role','content'} (lean Q&A, no injected context)."""
    if not history:
        return question
    convo = "\n".join(f"{m['role']}: {m['content']}" for m in history[-6:])
    prompt = (
        "Given the conversation below, rewrite the FOLLOW-UP into a single "
        "standalone search query that includes the entities it refers to. "
        "Output ONLY the rewritten query, nothing else.\n\n"
        f"Conversation:\n{convo}\n\nFollow-up: {question}\n\nStandalone query:"
    )
    r = requests.post(f"{OLLAMA}/api/generate",
                      json={"model": CHAT_MODEL, "prompt": prompt, "stream": False},
                      timeout=120)
    r.raise_for_status()
    return r.json()["response"].strip().strip('"')


# ---- 2. one chat completion over a messages list -----------------------------
def complete(messages):
    r = requests.post(f"{OLLAMA}/api/chat",
                      json={"model": CHAT_MODEL, "messages": messages, "stream": False},
                      timeout=300)
    r.raise_for_status()
    return r.json()["message"]["content"]


def _format_context(hits):
    if not hits:
        return "(no matching session excerpts found)"
    return "\n\n".join(
        f"(session {sid}, {_ts(st)}-{_ts(en)})\n{txt}"
        for sid, cid, dist, st, en, txt in hits)


# ---- 3. the REPL -------------------------------------------------------------
def run():
    con = connect(DB_PATH, PASSPHRASE)
    history = []   # lean conversation: [{'role':'user'/'assistant','content':...}]

    print(f"Chatting about patient {PATIENT_ID}. Type 'exit' to quit.\n")
    while True:
        try:
            question = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() in {"exit", "quit", "q"}:
            break
        if not question:
            continue

        # 1. rewrite follow-up -> standalone query
        search_q = condense(history, question)

        # 2. retrieve, scoped to this patient
        hits = recall(con, PATIENT_ID, search_q, K)

        # 3. answer: system + prior turns + current question-with-evidence
        current = {
            "role": "user",
            "content": f"Session excerpts:\n{_format_context(hits)}\n\nQuestion: {question}",
        }
        messages = [{"role": "system", "content": SYSTEM}] + history + [current]
        answer = complete(messages)

        # keep history LEAN (plain Q&A, no injected excerpts) so it doesn't bloat
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})

        print(f"\nai > {answer}\n")
        if SHOW_SOURCES and hits:
            if search_q != question:
                print(f"     [searched: {search_q}]")
            for sid, cid, dist, st, en, txt in hits:
                print(f"     - session {sid} [{_ts(st)}-{_ts(en)}]  dist={dist:.3f}")
                #print(f"\t-->{txt}")
            print()


if __name__ == "__main__":
    run()