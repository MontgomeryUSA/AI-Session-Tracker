"""
migrate_reembed.py
==================
Run this ONCE, on any vault that was built while embeddings came from Ollama.

Why it's necessary even though the model is identical:

    Ollama's bge-m3 and llama.cpp's bge-m3 are the same weights, but they don't
    necessarily pool or normalise the output the same way. local_llm.embed()
    L2-normalises; Ollama's /api/embeddings did not. vec_chunks is an L2-distance
    index, so normalising changes every distance in it.

    Mix the two and nothing crashes -- which is the dangerous part. Queries just
    quietly retrieve slightly wrong chunks, and the model dutifully cites them.
    A silent accuracy regression in a clinical tool is worse than a loud crash.

So: rebuild every vector from the chunk text, which is still sitting in the
`chunk` table in plaintext-under-encryption. Nothing is re-transcribed, nothing
is re-summarised, no audio is touched. It's just embeddings.

    python migrate_reembed.py

Takes a minute or two per hour of recorded audio on CPU.
"""
from __future__ import annotations

import sys
from getpass import getpass

import config
config.install_network_guard()

import sqlite_vec

import local_llm
import session_store as store


def main() -> None:
    if not config.DB_PATH.exists():
        print(f"No vault at {config.DB_PATH}. Nothing to migrate.")
        return

    passphrase = getpass("Vault passphrase: ")
    con = store.connect(str(config.DB_PATH), passphrase)
    try:
        con.execute("SELECT count(*) FROM sqlite_master").fetchone()
    except Exception:
        sys.exit("That passphrase doesn't open this vault.")

    rows = con.execute(
        "SELECT id, patient_id, session_id, text FROM chunk ORDER BY id"
    ).fetchall()
    if not rows:
        print("No chunks in the vault. Nothing to do.")
        return

    print(f"Re-embedding {len(rows)} chunks with the in-process model.")
    print("Loading bge-m3…")
    local_llm.warm_up(lambda m: None)

    # Drop every vector and rebuild. Safe: the source of truth is chunk.text,
    # and we're inside a transaction until the commit at the end.
    con.execute("DELETE FROM vec_chunks")

    for i, (chunk_id, patient_id, session_id, text) in enumerate(rows, start=1):
        vec = local_llm.embed(text)
        con.execute(
            "INSERT INTO vec_chunks(patient_id,session_id,chunk_id,embedding) "
            "VALUES (?,?,?,?)",
            (patient_id, session_id, chunk_id, sqlite_vec.serialize_float32(vec)),
        )
        if i % 25 == 0 or i == len(rows):
            print(f"  {i}/{len(rows)}")

    con.execute(
        "INSERT INTO audit(ts,action,ref) VALUES (datetime('now'),?,?)",
        ("reembed", f"chunks={len(rows)} model=local_llm/bge-m3"),
    )
    con.commit()
    con.close()
    print("Done. Every vector in the vault now comes from the in-process model.")


if __name__ == "__main__":
    main()
