"""
session_store.py
WhisperX JSON  ->  chunk  ->  embed (Ollama bge-m3)  ->  encrypted SQLite + vec index
                                                     ->  per-patient recall + chat

Pipeline:
    input.json {"segments":[{speaker,start,end,text}]}  (+ SOAP note, audio path)
        |  resolve speaker roles
        |  merge segments into ~600-char turn-aware chunks
        |  embed each chunk with bge-m3 (1024-dim)
        v
    clinic.db (SQLCipher)  -- session row, chunk rows, vec_chunks index, audit log
        |
        v
    recall(patient_id, "what did we discuss about sleep?")  -> grounded chunks
    chat(...)  -> Ollama answer cited to timestamps

Deps:
    sudo apt install -y sqlcipher libsqlcipher-dev
    pip install sqlcipher3-binary sqlite-vec requests
    ollama pull bge-m3          # 1024-dim embeddings
    ollama pull llama3.2:3b     # chat model (fits your 3.9GB VRAM)

NOTE: the chunker + vec0 KNN query in this file were validated end-to-end.
The SQLCipher key handling and Ollama calls follow your Phase-04 manual.
"""

import json
import datetime as dt
import sqlcipher3 as sqlite          # encrypted SQLite
import sqlite_vec

EMBED_DIM   = 1024

# ----------------------------------------------------------------------------
# 1. connection (encrypted) + extension + schema
# ----------------------------------------------------------------------------
def connect(path: str, passphrase: str):
    """Open the encrypted DB. PRAGMA key MUST be set before any other query."""
    con = sqlite.connect(path)
    con.execute(f"PRAGMA key = '{passphrase}'")     # parametrize via a vault in prod
    con.execute("PRAGMA cipher_compatibility = 4")
    con.execute("PRAGMA foreign_keys = ON")
    con.enable_load_extension(True)
    sqlite_vec.load(con)
    con.enable_load_extension(False)
    return con


def init_schema(con):
    con.executescript("""
        CREATE TABLE IF NOT EXISTS patient(
            id INTEGER PRIMARY KEY,
            alias TEXT);

        CREATE TABLE IF NOT EXISTS session(
            id INTEGER PRIMARY KEY,
            patient_id INT REFERENCES patient(id),
            ts TEXT,
            transcript TEXT,        -- full role-tagged transcript
            note TEXT,              -- SOAP note from Ollama
            audio_path TEXT);

        -- holds the actual chunk text so a KNN hit can recover the words.
        -- vec_chunks only stores vectors + ids; this is its text twin.
        CREATE TABLE IF NOT EXISTS chunk(
            id INTEGER PRIMARY KEY,
            session_id INT REFERENCES session(id),
            patient_id INT REFERENCES patient(id),
            start REAL, end REAL,
            text TEXT);

        CREATE TABLE IF NOT EXISTS audit(
            id INTEGER PRIMARY KEY, ts TEXT, action TEXT, ref TEXT);
    """)
    # vec0: patient_id as PARTITION KEY -> correct + fast per-patient scoping.
    # chunk_id mirrors chunk.id so we can join back to the text.
    con.execute(f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
            patient_id INTEGER partition key,
            session_id INTEGER,
            chunk_id   INTEGER,
            embedding  FLOAT[{EMBED_DIM}])""")
    con.commit()


def _audit(con, action, ref):
    con.execute("INSERT INTO audit(ts,action,ref) VALUES (?,?,?)",
                (dt.datetime.utcnow().isoformat(), action, str(ref)))


# ----------------------------------------------------------------------------
# 2. embeddings
# ----------------------------------------------------------------------------
def embed(text: str):
    from local_llm import embed as _embed
    return _embed(text)


# ----------------------------------------------------------------------------
# 3. chunking  (156 tiny segments -> ~16 coherent, retrievable chunks)
# ----------------------------------------------------------------------------
def _ts(s: float) -> str:
    return f"{int(s // 60):02d}:{int(s % 60):02d}"


def chunk_segments(segments, role_map, max_chars=600, max_gap=8.0):
    """
    Merge consecutive segments into windows that:
      - stay under max_chars,
      - break on a pause longer than max_gap seconds,
      - never split a segment,
      - carry  [mm:ss] Role: text  so the embedding has speaker + time context.
    """
    chunks, buf = [], []

    def flush():
        if not buf:
            return
        text = "\n".join(
            f"[{_ts(s['start'])}] {role_map.get(s['speaker'], s['speaker'])}: {s['text'].strip()}"
            for s in buf)
        chunks.append({
            "start": buf[0]["start"],
            "end":   buf[-1]["end"],
            "text":  text,
        })
        buf.clear()

    for s in segments:
        if buf and (s["start"] - buf[-1]["end"] > max_gap or
                    sum(len(x["text"]) for x in buf) + len(s["text"]) > max_chars):
            flush()
        buf.append(s)
    flush()
    return chunks


# ----------------------------------------------------------------------------
# 4. ingest one session
# ----------------------------------------------------------------------------
def ingest_session(con, *, patient_id, patient_alias, json_path,
                   soap_note_path, audio_path, role_map):
    """
    role_map: {"SPEAKER_00": "Patient", "SPEAKER_01": "Clinician"}
    Returns the new session id.
    """
    segments = json.load(open(json_path))["segments"]
    
    soap_note = open(soap_note_path, encoding="utf-8").read().strip()

    # full role-tagged transcript (stored whole, for display / re-summary)
    transcript = "\n".join(
        f"[{_ts(s['start'])}] {role_map.get(s['speaker'], s['speaker'])}: {s['text'].strip()}"
        for s in segments)

    con.execute("INSERT OR IGNORE INTO patient(id,alias) VALUES (?,?)",
                (patient_id, patient_alias))

    cur = con.execute(
        "INSERT INTO session(patient_id,ts,transcript,note,audio_path) VALUES (?,?,?,?,?)",
        (patient_id, dt.datetime.utcnow().isoformat(), transcript, soap_note, audio_path))
    session_id = cur.lastrowid

    chunks = chunk_segments(segments, role_map)
    for c in chunks:
        ccur = con.execute(
            "INSERT INTO chunk(session_id,patient_id,start,end,text) VALUES (?,?,?,?,?)",
            (session_id, patient_id, c["start"], c["end"], c["text"]))
        chunk_id = ccur.lastrowid
        vec = embed(c["text"])
        con.execute(
            "INSERT INTO vec_chunks(patient_id,session_id,chunk_id,embedding) VALUES (?,?,?,?)",
            (patient_id, session_id, chunk_id, sqlite_vec.serialize_float32(vec)))

    _audit(con, "ingest_session", f"patient={patient_id} session={session_id} chunks={len(chunks)}")
    con.commit()
    return session_id


# ----------------------------------------------------------------------------
# 5. recall  (scoped to ONE patient — never crosses patients)
# ----------------------------------------------------------------------------
def recall(con, patient_id, question, k=5):
    q = embed(question)
    rows = con.execute("""
        SELECT v.session_id, v.chunk_id, v.distance, c.start, c.end, c.text
        FROM vec_chunks v
        JOIN chunk c ON c.id = v.chunk_id
        WHERE v.patient_id = ? AND v.embedding MATCH ? AND k = ?
        ORDER BY distance""",
        (patient_id, sqlite_vec.serialize_float32(q), k)).fetchall()
    _audit(con, "recall", f"patient={patient_id} q={question!r}")
    con.commit()
    return rows  # (session_id, chunk_id, distance, start, end, text)


# ----------------------------------------------------------------------------
# 6. chat  (RAG: retrieved chunks -> grounded answer)
# ----------------------------------------------------------------------------
def chat(con, patient_id, question, k=5):
    hits = recall(con, patient_id, question, k)
    context = "\n\n".join(
        f"(session {sid}, {_ts(st)}-{_ts(en)})\n{txt}"
        for sid, cid, dist, st, en, txt in hits)

    prompt = (
        "You are a clinical documentation assistant. Answer ONLY from the session "
        "excerpts below. Cite the session id and timestamp for each claim. If the "
        "excerpts don't contain the answer, say so.\n\n"
        f"=== EXCERPTS ===\n{context}\n\n=== QUESTION ===\n{question}")

    print(f"Promting the AI...asking question {question}")
    r = requests.post(f"{OLLAMA}/api/generate",
                      json={"model": CHAT_MODEL, "prompt": prompt, "stream": False},
                      timeout=300)
    r.raise_for_status()
    return r.json()["response"], hits


# ----------------------------------------------------------------------------
# example
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    db_password = "password"
    con = connect("sql_db/clinic.db", db_password)
    init_schema(con)

    sid = ingest_session(
        con,
        patient_id=1,
        patient_alias="PT-0001",
        json_path="./transcript/conversation_10min_16k_proc.json",
        soap_note_path="./summary/conversation_10min_16k_summary_2026-06-24_15-01-41.txt",
        audio_path="./audio/conversation_10min.wav",
        role_map={"SPEAKER_01": "Therapist", "SPEAKER_00": "Patient"},
    )
    print("ingested session", sid)

    
    question = "what was said about italy?"
    print("Test the database by asking a question about the transcript:")
    print(f"{question}")
    answer, hits = chat(con, patient_id=1, question=question)
    print("\nANSWER:\n", answer)
    print("\nSOURCES:")
    for sid, cid, dist, st, en, txt in hits:
        print(f"  session {sid} [{_ts(st)}-{_ts(en)}] dist={dist:.3f} -> {txt}")
    