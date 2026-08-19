"""Pluggable embedding backends, selected by the EMBEDDING_BACKEND env var.

Why this exists: `.env.example` has always advertised
`EMBEDDING_BACKEND=fastembed | bge-m3 | openai`, setup-docker.sh flips it to
`bge-m3` and prints "bge-m3 embeddings", and the README sells bge-m3 as the
reason to take the Docker path -- but nothing ever read the variable. Every
path silently used BAAI/bge-small-en-v1.5, an ENGLISH model, which is exactly
why NB2 shows weak recall on Vietnamese paraphrases. Students were promised an
upgrade that never happened.

The default is unchanged (fastembed / bge-small / 384-dim), so the lite path
and every rubric threshold behave exactly as before. The other backends are
opt-in via the environment.

    EMBEDDING_BACKEND=fastembed     BAAI/bge-small-en-v1.5     384   (default, lite)
    EMBEDDING_BACKEND=multilingual  intfloat/multilingual-e5-large 1024 (fastembed)
    EMBEDDING_BACKEND=bge-m3        BAAI/bge-m3                1024  (sentence-transformers)
    EMBEDDING_BACKEND=openai        text-embedding-3-small     1536  (needs OPENAI_API_KEY)
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, Iterator

import numpy as np

DEFAULT_BACKEND = "fastembed"

# ONNX Runtime defaults to one intra-op thread per core. For a SINGLE short
# query that is the classic latency anti-pattern: the model is far too small to
# fill 20 cores, so thread fan-out/join dominates the actual matmuls. Measured
# on this 20-core host (50 golden queries x 3 reps, warm):
#
#     threads=4      p50 20.6 ms   p99 29.2 ms      1000-doc index ~128 s
#     threads=8      p50 20.8 ms   p99 28.9 ms      1000-doc index ~129 s
#     unbounded      p50 33.0 ms   p99 54.1 ms      1000-doc index ~199 s
#
# Bounding it is faster on BOTH axes -- there is no throughput/latency trade to
# make here, oversubscription simply loses. The cap is 8 rather than 4 so a
# clean 4-8 core laptop keeps every core; only many-core hosts are held back.
# Override with EMBED_THREADS if you are profiling.
DEFAULT_THREAD_CAP = 8


def embed_threads() -> int:
    raw = os.getenv("EMBED_THREADS")
    if raw:
        return max(1, int(raw))
    return max(1, min(DEFAULT_THREAD_CAP, os.cpu_count() or DEFAULT_THREAD_CAP))


@dataclass(frozen=True)
class BackendSpec:
    model: str
    dim: int
    provider: str       # fastembed | sentence-transformers | openai
    note: str = ""


BACKENDS: dict[str, BackendSpec] = {
    "fastembed": BackendSpec("BAAI/bge-small-en-v1.5", 384, "fastembed",
                             "English-focused; weak on Vietnamese paraphrase (that is the NB2 lesson)"),
    # Added for the NB2 embedding experiment: the cheapest way to test the
    # "is the model or the fusion the problem?" question. Same 384 dims and
    # roughly the same per-query cost as bge-small, but multilingual -- so it
    # isolates *language coverage* from *model size*, which comparing against
    # a 1024-dim 2.2 GB model cannot do.
    "multilingual-small": BackendSpec(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", 384, "fastembed",
        "Multilingual (VN included), 384d, ~0.22 GB — same latency class as bge-small"),
    "multilingual": BackendSpec("intfloat/multilingual-e5-large", 1024, "fastembed",
                                "Multilingual, no extra dependency, ~2.2 GB download"),
    "bge-m3": BackendSpec("BAAI/bge-m3", 1024, "sentence-transformers",
                          "Multilingual; needs sentence-transformers (requirements-full.txt)"),
    "openai": BackendSpec("text-embedding-3-small", 1536, "openai",
                          "Needs OPENAI_API_KEY; costs money"),
}


class Embedder:
    """Uniform `.embed(list[str]) -> Iterator[np.ndarray]`, matching fastembed."""

    def __init__(self, backend: str | None = None) -> None:
        name = (backend or os.getenv("EMBEDDING_BACKEND") or DEFAULT_BACKEND).strip().lower()
        if name not in BACKENDS:
            raise ValueError(
                f"Unknown EMBEDDING_BACKEND={name!r}. "
                f"Valid: {', '.join(sorted(BACKENDS))}"
            )
        self.backend = name
        self.spec = BACKENDS[name]
        self._impl = None

    # dimension is a property of the chosen model, never a hard-coded constant
    @property
    def dim(self) -> int:
        return self.spec.dim

    @property
    def model_name(self) -> str:
        return self.spec.model

    def _load(self):
        if self._impl is not None:
            return self._impl
        p = self.spec.provider
        if p == "fastembed":
            from fastembed import TextEmbedding
            self._impl = TextEmbedding(model_name=self.spec.model,
                                       threads=embed_threads())
        elif p == "sentence-transformers":
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:                       # pragma: no cover
                raise ImportError(
                    f"EMBEDDING_BACKEND={self.backend} needs sentence-transformers.\n"
                    "It ships with the Docker path:  pip install -r requirements-full.txt"
                ) from exc
            self._impl = SentenceTransformer(self.spec.model)
        elif p == "openai":
            try:
                from openai import OpenAI
            except ImportError as exc:                       # pragma: no cover
                raise ImportError(
                    "EMBEDDING_BACKEND=openai needs the openai package "
                    "(requirements-full.txt) and OPENAI_API_KEY."
                ) from exc
            if not os.getenv("OPENAI_API_KEY"):
                raise RuntimeError("EMBEDDING_BACKEND=openai but OPENAI_API_KEY is unset.")
            self._impl = OpenAI()
        return self._impl

    def embed(self, texts: Iterable[str]) -> Iterator[np.ndarray]:
        texts = list(texts)
        impl = self._load()
        p = self.spec.provider
        if p == "fastembed":
            yield from impl.embed(texts)
        elif p == "sentence-transformers":
            for v in impl.encode(texts, normalize_embeddings=True):
                yield np.asarray(v, dtype=np.float32)
        else:  # openai
            resp = impl.embeddings.create(model=self.spec.model, input=texts)
            for item in resp.data:
                yield np.asarray(item.embedding, dtype=np.float32)


def describe() -> str:
    e = Embedder()
    return f"{e.backend} -> {e.model_name} ({e.dim}d) — {e.spec.note}"
