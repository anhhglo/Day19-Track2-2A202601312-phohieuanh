"""HybridMemoryAgent — episodic memory (vector) + stable profile (feature store).

Bonus challenge POC for Lab 19. Every mechanism here is one of the lab's
lessons applied rather than restated:

  NB2  RRF hybrid          — BM25 catches the literal token (`Kubernetes`,
                             `bge-m3`, `Nghị định 13`) that an English-trained
                             embedder drops; vector catches the paraphrase.
  NB5  filtered-ANN        — the `user_id` filter goes INSIDE the Qdrant query,
                             never as a post-filter. Post-filtering a per-user
                             memory store is both a recall bug and, at k=10 with
                             many users, an empty-answer bug.
  NB7  namespaced cache    — the semantic cache keys on `user_id`. An
                             un-namespaced cache over personal memory is a
                             cross-user data leak, not a cache miss.
  NB4  online feature store — the profile is read at request time with a
                             single online lookup, not recomputed from history.

Run the demo:  python bonus/demo.py
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qdrant_client import QdrantClient, models  # noqa: E402
from qdrant_client.models import Distance, PointStruct, VectorParams  # noqa: E402
from rank_bm25 import BM25Okapi  # noqa: E402

from app.cache import SemanticCache  # noqa: E402
from app.embeddings import Embedder  # noqa: E402

MEMORY_COLLECTION = "bonus_episodic_memory"
RRF_K = 60


@dataclass
class Memory:
    memory_id: int
    user_id: str
    text: str
    kind: str          # note | read | query
    ts: float


@dataclass
class RecallResult:
    query: str
    user_id: str
    profile: dict[str, Any]
    memories: list[Memory]
    cached: bool
    context: str


class HybridMemoryAgent:
    """Assembles LLM context from two stores that answer two different questions.

    Vector store  -> "what is relevant to this query"   (episodic, grows hourly)
    Feature store -> "who is this user"                 (profile, refreshed daily)

    The split is deliberate; `bonus/ARCHITECTURE.md` argues why they are not one
    store.
    """

    def __init__(
        self,
        feature_store: Any | None = None,
        client: QdrantClient | None = None,
        cache_threshold: float = 0.85,
        cache_ttl_s: float | None = 900.0,
    ) -> None:
        self.embedder = Embedder()
        self.client = client or QdrantClient(":memory:")
        if MEMORY_COLLECTION in {c.name for c in self.client.get_collections().collections}:
            self.client.delete_collection(MEMORY_COLLECTION)
        self.client.create_collection(
            collection_name=MEMORY_COLLECTION,
            # dimension follows the model, never a literal -- switching
            # EMBEDDING_BACKEND to bge-m3 moves it 384 -> 1024.
            vectors_config=VectorParams(size=self.embedder.dim, distance=Distance.COSINE),
        )
        self.memories: list[Memory] = []
        self._bm25: BM25Okapi | None = None
        self._bm25_dirty = True
        self.feature_store = feature_store
        # 0.85, not the 0.75 AWS headline: NB7's sweep showed 0.75 still serves
        # a measurable share of wrong answers on this corpus. Personal memory
        # is exactly where a wrong-but-fluent answer is most damaging.
        self.cache = SemanticCache(
            client=self.client,
            embedder=self.embedder,
            dim=self.embedder.dim,
            threshold=cache_threshold,
            ttl_s=cache_ttl_s,
            namespaced=True,
        )

    # ── write path ──────────────────────────────────────────────────────
    def remember(self, text: str, user_id: str = "u_001", kind: str = "note") -> None:
        """Add one piece of episodic memory for this user."""
        mem = Memory(
            memory_id=len(self.memories),
            user_id=user_id,
            text=text,
            kind=kind,
            ts=time.time(),
        )
        vector = next(self.embedder.embed([text])).tolist()
        self.client.upsert(
            collection_name=MEMORY_COLLECTION,
            points=[PointStruct(
                id=mem.memory_id,
                vector=vector,
                payload={"user_id": user_id, "text": text, "kind": kind, "ts": mem.ts},
            )],
        )
        self.memories.append(mem)
        self._bm25_dirty = True

    # ── retrieval ───────────────────────────────────────────────────────
    def _ensure_bm25(self) -> None:
        if self._bm25_dirty:
            # Whitespace tokenisation, same baseline as app/search.py. For a
            # real deployment this is where pyvi/underthesea goes -- see
            # ARCHITECTURE.md, decision 1.
            self._bm25 = BM25Okapi([m.text.lower().split() for m in self.memories])
            self._bm25_dirty = False

    def _keyword_ids(self, query: str, user_id: str, depth: int) -> list[int]:
        self._ensure_bm25()
        if self._bm25 is None or not self.memories:
            return []
        scores = self._bm25.get_scores(query.lower().split())
        mine = [m.memory_id for m in self.memories if m.user_id == user_id]
        return sorted(mine, key=lambda i: -scores[i])[:depth]

    def _vector_ids(self, query: str, user_id: str, depth: int) -> list[int]:
        qv = next(self.embedder.embed([query])).tolist()
        pts = self.client.query_points(
            collection_name=MEMORY_COLLECTION,
            query=qv,
            limit=depth,
            # filtered-ANN: the tenant predicate is evaluated inside the index
            # traversal. Fetching top-k globally and dropping other users after
            # the fact is the NB5 recall cliff, with a privacy failure attached.
            query_filter=models.Filter(must=[models.FieldCondition(
                key="user_id", match=models.MatchValue(value=user_id))]),
        ).points
        return [int(p.id) for p in pts]

    def search_memories(self, query: str, user_id: str, top_k: int = 5,
                        affinity: str | None = None) -> list[Memory]:
        """Hybrid RRF over this user's memories only.

        Two retrievers always (BM25 + vector). A THIRD leg joins the fusion when
        the feature store supplied a `topic_affinity`: the same query re-embedded
        with the user's dominant topic appended. This is why the profile is a
        feature-store lookup and not a constant -- "gợi ý tôi nên đọc gì tiếp
        theo" carries almost no topical signal on its own, so without the
        profile leg the ranking is close to arbitrary.

        RRF is what makes a third leg cheap: it fuses *ranks*, so a new
        retriever needs no score calibration against the other two.
        """
        depth = max(top_k * 5, 25)
        legs = [self._keyword_ids(query, user_id, depth),
                self._vector_ids(query, user_id, depth)]
        if affinity:
            legs.append(self._vector_ids(f"{query} {affinity}", user_id, depth))
        rrf: dict[int, float] = {}
        for ids in legs:
            for rank, mid in enumerate(ids, start=1):     # rank is 1-based
                rrf[mid] = rrf.get(mid, 0.0) + 1.0 / (RRF_K + rank)
        best = sorted(rrf.items(), key=lambda kv: -kv[1])[:top_k]
        return [self.memories[mid] for mid, _ in best]

    # ── profile ─────────────────────────────────────────────────────────
    def profile(self, user_id: str) -> dict[str, Any]:
        """One online lookup. Degrades to {} when Feast has not been applied."""
        if self.feature_store is None:
            return {}
        try:
            raw = self.feature_store.get_online_features(
                features=[
                    "user_profile_features:preferred_language",
                    "user_profile_features:reading_speed_wpm",
                    "user_profile_features:topic_affinity",
                    "query_velocity_features:queries_last_hour",
                    "query_velocity_features:distinct_topics_24h",
                ],
                entity_rows=[{"user_id": user_id}],
            ).to_dict()
        except Exception as exc:                          # noqa: BLE001
            return {"_error": str(exc)}
        return {k: v[0] for k, v in raw.items() if k != "user_id"}

    # ── read path ───────────────────────────────────────────────────────
    def recall(self, query: str, user_id: str = "u_001", top_k: int = 5) -> str:
        """Top-K memories + profile features -> assembled context string."""
        return self.recall_detailed(query, user_id, top_k).context

    def recall_detailed(self, query: str, user_id: str = "u_001",
                        top_k: int = 5) -> RecallResult:
        hit = self.cache.get(user_id, query)
        if hit is not None:
            return RecallResult(query, user_id, {}, [], True, hit.answer)

        prof = self.profile(user_id)
        affinity = prof.get("topic_affinity") if "_error" not in prof else None
        mems = self.search_memories(query, user_id, top_k, affinity=affinity)
        context = self._assemble(query, user_id, prof, mems, affinity)
        self.cache.put(user_id, query, context)
        return RecallResult(query, user_id, prof, mems, False, context)

    @staticmethod
    def _assemble(query: str, user_id: str, prof: dict[str, Any],
                  mems: list[Memory], affinity: str | None = None) -> str:
        lines = [f"### Người dùng: {user_id}"]
        if prof and "_error" not in prof:
            lines += [
                f"- ngôn ngữ ưa dùng : {prof.get('preferred_language')}",
                f"- tốc độ đọc       : {prof.get('reading_speed_wpm')} wpm",
                f"- chủ đề quan tâm  : {prof.get('topic_affinity')}",
                f"- query 1 giờ qua  : {prof.get('queries_last_hour')}",
                f"- chủ đề 24h qua   : {prof.get('distinct_topics_24h')}",
            ]
        else:
            lines.append("- (chưa có profile — chạy NB4 `feast apply` + materialize trước)")
        legs = "BM25 + vector" + (f" + profile[{affinity}]" if affinity else "")
        lines += ["", f"### Ký ức liên quan tới: {query!r}", f"    (RRF legs: {legs})"]
        if not mems:
            lines.append("- (chưa có ký ức nào)")
        for i, m in enumerate(mems, 1):
            lines.append(f"{i}. [{m.kind}] {m.text}")
        return "\n".join(lines)


def load_feature_store(repo: Path | None = None) -> Any | None:
    """Open the Lab-19 Feast repo if NB4 has already been run; else None."""
    repo = repo or (ROOT / "app" / "feast_repo")
    try:
        from feast import FeatureStore
    except ImportError:
        return None
    if not (repo / "registry.db").exists():
        return None
    try:
        return FeatureStore(repo_path=str(repo))
    except Exception:                                     # noqa: BLE001
        return None
