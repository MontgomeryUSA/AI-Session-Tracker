"""
Configuration for the memory engine.

Keeping all tunables in one dataclass makes it trivial to, e.g., point a
desktop application at a per-user data directory, or swap embedding
models later without touching engine internals.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# bge-small-en-v1.5 produces 384-dimensional embeddings. This is only
# used as a fallback / sanity-check value; the real dimension is always
# read from the loaded embedding model at runtime.
DEFAULT_EMBEDDING_DIM = 384

# BGE models are "asymmetric": passages are embedded as-is, but queries
# should be prefixed with an instruction for best retrieval quality.
DEFAULT_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


@dataclass
class EngineConfig:
    """All paths and settings the engine needs.

    Attributes:
        data_dir: Root directory where the SQLite database and FAISS
            index are persisted. Created automatically if missing.
        model_name: Sentence-transformers model identifier or local path.
        model_cache_dir: Optional explicit cache folder for downloaded
            model weights. Useful for bundling a pre-downloaded model
            with a desktop installer for fully-offline deployment.
        local_files_only: If True, never attempt to contact Hugging Face
            Hub -- the model must already exist in the cache/path. Set
            this to True in production/HIPAA-conscious deployments once
            the model has been pre-downloaded and verified.
        db_filename: SQLite database filename, stored under data_dir.
        index_filename: FAISS index filename, stored under data_dir.
        query_instruction: Instruction prefix prepended to search queries
            (BGE-style asymmetric retrieval). Leave empty for symmetric
            models.
        context_before / context_after: Default number of neighbouring
            segments to include on either side of a match in
            ``retrieve_context``.
    """

    data_dir: Path = field(default_factory=lambda: Path("./memory_data"))
    model_name: str = "BAAI/bge-small-en-v1.5"
    model_cache_dir: Optional[Path] = None
    local_files_only: bool = False
    db_filename: str = "memory.db"
    index_filename: str = "vectors.faiss"
    index_meta_filename: str = "vectors.meta.json"
    query_instruction: str = DEFAULT_QUERY_INSTRUCTION
    context_before: int = 2
    context_after: int = 2

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        if self.model_cache_dir is not None:
            self.model_cache_dir = Path(self.model_cache_dir)

    @property
    def db_path(self) -> Path:
        return self.data_dir / self.db_filename

    @property
    def index_path(self) -> Path:
        return self.data_dir / self.index_filename

    @property
    def index_meta_path(self) -> Path:
        return self.data_dir / self.index_meta_filename

    def ensure_data_dir(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
