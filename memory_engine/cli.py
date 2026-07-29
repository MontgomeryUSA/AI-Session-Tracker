"""
Phase 7: a small local CLI for exercising the MemoryEngine.

Two modes:

  One-shot (good for scripting / automation):
      python -m memory_engine.cli ingest ./transcripts
      python -m memory_engine.cli search "sleep problems" --top-k 5
      python -m memory_engine.cli context "sleep problems"
      python -m memory_engine.cli session session_001
      python -m memory_engine.cli rebuild

  Interactive REPL (good for manual exploration), launched with no
  arguments:
      $ python -m memory_engine.cli
      memory> search sleep problems
      memory> context sleep problems
      memory> session session_001
      memory> ingest ./transcripts
      memory> rebuild
      memory> exit
"""
from __future__ import annotations

import argparse
import json
import logging
import shlex
import sys
from pathlib import Path
from typing import Optional

from .config import EngineConfig
from .engine import MemoryEngine


def _print_search_results(results: list[dict]) -> None:
    if not results:
        print("No results.")
        return
    print("Results:")
    for i, r in enumerate(results, start=1):
        print(
            f'[{i}] Score: {r["score"]:.2f}  Speaker: {r["speaker"]}  '
            f'({r["start_time"]:.1f}s-{r["end_time"]:.1f}s)  '
            f'Session: {r["conversation_id"]}\n'
            f'    "{r["text"]}"'
        )


def _print_context_windows(windows: list[dict]) -> None:
    if not windows:
        print("No results.")
        return
    for i, w in enumerate(windows, start=1):
        print(f'[{i}] Score: {w["score"]:.2f}  Session: {w["conversation_id"]}')
        print(w["formatted_text"])
        print()


def _print_session(session: dict) -> None:
    print(f'Session: {session["session_id"]}  ({session["segment_count"]} segments)')
    print(session["transcript"])


def _run_command(engine: MemoryEngine, command: str, args: list[str]) -> None:
    if command == "ingest":
        if not args:
            print("Usage: ingest <directory>")
            return
        stats = engine.ingest_directory(args[0])
        print(json.dumps(stats, indent=2))
    elif command == "search":
        if not args:
            print("Usage: search <query>")
            return
        top_k = 5
        if "--top-k" in args:
            idx = args.index("--top-k")
            top_k = int(args[idx + 1])
            del args[idx : idx + 2]
        query = " ".join(args)
        _print_search_results(engine.search(query, top_k=top_k))
    elif command == "context":
        if not args:
            print("Usage: context <query>")
            return
        top_k = 5
        if "--top-k" in args:
            idx = args.index("--top-k")
            top_k = int(args[idx + 1])
            del args[idx : idx + 2]
        query = " ".join(args)
        _print_context_windows(engine.retrieve_context(query, top_k=top_k))
    elif command == "session":
        if not args:
            print("Usage: session <session_id>")
            return
        _print_session(engine.get_session(args[0]))
    elif command == "rebuild":
        n = engine.rebuild_index()
        print(f"Rebuilt index with {n} vectors.")
    else:
        print(f"Unknown command: {command}")


def _repl(engine: MemoryEngine) -> None:
    print("Local RAG Memory Engine -- type 'help' for commands, 'exit' to quit.")
    while True:
        try:
            line = input("memory> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line in ("exit", "quit"):
            break
        if line == "help":
            print(
                "Commands:\n"
                "  ingest <directory>\n"
                "  search <query> [--top-k N]\n"
                "  context <query> [--top-k N]\n"
                "  session <session_id>\n"
                "  rebuild\n"
                "  exit"
            )
            continue
        parts = shlex.split(line)
        command, args = parts[0], parts[1:]
        try:
            _run_command(engine, command, args)
        except Exception as exc:  # noqa: BLE001 - CLI top-level guard
            print(f"Error: {exc}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local RAG memory engine CLI")
    parser.add_argument(
        "--data-dir", default="./memory_data", help="Directory for the SQLite DB and FAISS index"
    )
    parser.add_argument(
        "--model", default="BAAI/bge-small-en-v1.5", help="sentence-transformers model name"
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Never contact Hugging Face Hub; model must already be cached",
    )
    sub = parser.add_subparsers(dest="command")

    p_ingest = sub.add_parser("ingest", help="Ingest a directory of transcript JSON files")
    p_ingest.add_argument("directory")

    p_search = sub.add_parser("search", help="Semantic search over indexed segments")
    p_search.add_argument("query")
    p_search.add_argument("--top-k", type=int, default=5)

    p_context = sub.add_parser("context", help="Search and reconstruct surrounding context")
    p_context.add_argument("query")
    p_context.add_argument("--top-k", type=int, default=5)

    p_session = sub.add_parser("session", help="Retrieve a full reconstructed session transcript")
    p_session.add_argument("session_id")

    sub.add_parser("rebuild", help="Rebuild the FAISS index from SQLite")

    return parser


def main(argv: Optional[list[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = build_arg_parser()
    args = parser.parse_args(argv)

    config = EngineConfig(
        data_dir=Path(args.data_dir),
        model_name=args.model,
        local_files_only=args.local_files_only,
    )
    engine = MemoryEngine(config=config)

    try:
        if args.command == "ingest":
            stats = engine.ingest_directory(args.directory)
            print(json.dumps(stats, indent=2))
        elif args.command == "search":
            _print_search_results(engine.search(args.query, top_k=args.top_k))
        elif args.command == "context":
            _print_context_windows(engine.retrieve_context(args.query, top_k=args.top_k))
        elif args.command == "session":
            _print_session(engine.get_session(args.session_id))
        elif args.command == "rebuild":
            n = engine.rebuild_index()
            print(f"Rebuilt index with {n} vectors.")
        else:
            _repl(engine)
    finally:
        engine.close()


if __name__ == "__main__":
    main()
