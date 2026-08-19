"""Does the embedding model or the fusion explain NB2's paraphrase result?

NB2 on the default backend shows vector search LOSING to BM25 on the
`paraphrase` slice — the opposite of what the textbook says should happen. Two
explanations are possible and they lead to opposite fixes:

  (a) the RRF fusion is wrong                -> fix the code
  (b) the embedder does not speak Vietnamese -> fix the model

BM25 is identical across every run here, so it acts as a control: if the
keyword column moves, the harness is wrong. Only `EMBEDDING_BACKEND` changes.

    python scripts/embedding_ablation.py                       # both backends
    python scripts/embedding_ablation.py fastembed             # just one

Latency is measured per mode so the quality gain can be priced against the
rubric's 50 ms P99 budget — a model that answers better but misses the budget
is not a production upgrade.
"""
from __future__ import annotations

import json
import os
import statistics as stats
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TOP_K = 10
RRF_K = 60
LATENCY_REPS = 3         # × 50 golden queries = 150 calls per mode
DEFAULT_BACKENDS = ["fastembed", "multilingual-small"]


def precision_at_k(retrieved: list[str], relevant: set[str], k: int = TOP_K) -> float:
    top = retrieved[:k]
    return sum(1 for d in top if d in relevant) / len(top) if top else 0.0


def evaluate(backend: str, golden: list[dict]) -> dict:
    os.environ["EMBEDDING_BACKEND"] = backend
    # Imported inside the function: app.search snapshots the backend at import
    # time, so each run needs a fresh module state.
    for mod in [m for m in list(sys.modules) if m.startswith("app.")]:
        del sys.modules[mod]
    from app.embeddings import Embedder            # noqa: PLC0415
    from app.search import Searcher                # noqa: PLC0415

    emb = Embedder()
    print(f"\n── {backend}: {emb.model_name} ({emb.dim}d) ──", flush=True)
    t0 = time.perf_counter()
    searcher = Searcher.from_corpus(ROOT / "data" / "corpus_vn.jsonl")
    index_s = time.perf_counter() - t0
    print(f"   indexed {searcher.size} docs in {index_s:.1f}s", flush=True)

    scores = {"keyword": [], "semantic": [], "hybrid": []}
    by_slice: dict[str, dict[str, list[float]]] = {}
    for q in golden:
        relevant = set(q["relevant_doc_ids"])
        row = {}
        for mode in scores:
            hits = [h.doc_id for h in searcher.search(q["query"], mode=mode,
                                                      top_k=TOP_K, rrf_k=RRF_K)]
            p = precision_at_k(hits, relevant)
            scores[mode].append(p)
            row[mode] = p
        s = by_slice.setdefault(q["mode_hint"], {m: [] for m in scores})
        for mode, p in row.items():
            s[mode].append(p)

    # Latency, warm. The first call of each mode is dropped: it pays for the
    # lazy ONNX session, which production warms before taking traffic.
    latency = {}
    for mode in scores:
        searcher.search(golden[0]["query"], mode=mode, top_k=TOP_K)
        samples = []
        for _ in range(LATENCY_REPS):
            for q in golden:
                t = time.perf_counter()
                searcher.search(q["query"], mode=mode, top_k=TOP_K, rrf_k=RRF_K)
                samples.append((time.perf_counter() - t) * 1000)
        samples.sort()
        latency[mode] = {
            "p50": samples[len(samples) // 2],
            "p99": samples[int(len(samples) * 0.99)],
        }

    return {
        "backend": backend, "model": emb.model_name, "dim": emb.dim,
        "index_s": index_s,
        "avg": {m: stats.mean(v) for m, v in scores.items()},
        "slice": {k: {m: stats.mean(v) for m, v in d.items()} for k, d in by_slice.items()},
        "latency": latency,
    }


def main() -> int:
    backends = sys.argv[1:] or DEFAULT_BACKENDS
    golden = [json.loads(l) for l in
              (ROOT / "data" / "golden_set.jsonl").open(encoding="utf-8")]
    print(f"Embedding ablation — {len(golden)} golden queries, Precision@{TOP_K}")
    print("=" * 78)

    results = [evaluate(b, golden) for b in backends]

    print("\n\nPrecision@10 — trung bình toàn bộ golden set")
    print(f"{'backend':22}{'dim':>5}{'keyword':>10}{'semantic':>10}{'hybrid':>9}")
    for r in results:
        print(f"{r['backend']:22}{r['dim']:>5}{r['avg']['keyword']:>9.1%}"
              f"{r['avg']['semantic']:>10.1%}{r['avg']['hybrid']:>9.1%}")

    print("\nPrecision@10 — theo loại query (đây mới là chỗ model lộ ra)")
    print(f"{'slice':12}{'backend':22}{'keyword':>10}{'semantic':>10}{'hybrid':>9}")
    for sl in ("exact", "paraphrase", "mixed"):
        for r in results:
            if sl not in r["slice"]:
                continue
            s = r["slice"][sl]
            print(f"{sl:12}{r['backend']:22}{s['keyword']:>9.1%}"
                  f"{s['semantic']:>10.1%}{s['hybrid']:>9.1%}")
        print()

    print("Latency in-process (không qua HTTP) — giá phải trả cho chất lượng")
    print(f"{'backend':22}{'sem P50':>10}{'sem P99':>10}{'hyb P50':>10}{'hyb P99':>10}"
          f"{'index':>9}")
    for r in results:
        print(f"{r['backend']:22}{r['latency']['semantic']['p50']:>9.1f}ms"
              f"{r['latency']['semantic']['p99']:>9.1f}ms"
              f"{r['latency']['hybrid']['p50']:>9.1f}ms"
              f"{r['latency']['hybrid']['p99']:>9.1f}ms"
              f"{r['index_s']:>8.0f}s")

    if len(results) == 2:
        a, b = results
        d_para = (b["slice"].get("paraphrase", {}).get("semantic", 0)
                  - a["slice"].get("paraphrase", {}).get("semantic", 0))
        d_lat = b["latency"]["hybrid"]["p99"] - a["latency"]["hybrid"]["p99"]
        print(f"\nΔ paraphrase / semantic : {d_para:+.1%}")
        print(f"Δ hybrid P99            : {d_lat:+.1f}ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
