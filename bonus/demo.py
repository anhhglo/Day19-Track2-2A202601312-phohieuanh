"""Demo for HybridMemoryAgent — 5 queries, one assembled context each.

    python bonus/demo.py

Exits 0 whether or not NB4 has been run; without the Feast registry the
profile block degrades to a notice and only the episodic half is shown.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bonus.agent import HybridMemoryAgent, load_feature_store  # noqa: E402

USER = "u_001"
OTHER = "u_002"

# What the assistant has seen this user do. `kind` mirrors the three episodic
# sources in the brief: notes the user wrote, documents they read, queries they
# typed.
SEEDS: list[tuple[str, str]] = [
    ("read", "Đã đọc: Kubernetes HPA tự động scale pod theo CPU và custom metric"),
    ("read", "Đã đọc: so sánh Karpenter và Cluster Autoscaler khi node pool co giãn"),
    ("note", "Ghi chú: spot instance rẻ hơn 70% nhưng phải chịu được interruption 2 phút"),
    ("read", "Đã đọc: mTLS giữa các service trong service mesh Istio"),
    ("note", "Ghi chú: Nghị định 13/2023 yêu cầu dữ liệu cá nhân của người dùng VN "
             "phải có cơ sở pháp lý xử lý rõ ràng"),
    ("read", "Đã đọc: pgvector HNSW index và cách chọn ef_search"),
    ("query", "Tìm kiếm: cách giảm chi phí hạ tầng khi traffic tăng đột biến"),
    ("read", "Đã đọc: RRF k=60 hợp nhất BM25 với vector search"),
    ("note", "Ghi chú: bge-m3 đa ngữ tốt cho tiếng Việt nhưng nặng gấp 4 lần bge-small"),
    ("query", "Tìm kiếm: circuit breaker tránh cascading failure giữa microservice"),
]

# A second user, so the isolation check at the end is a real check.
OTHER_SEEDS = [
    ("note", "Ghi chú: doanh thu quý 3 của GLOBEX là 4,2 tỷ VND"),
    ("read", "Đã đọc: hướng dẫn thiết kế bảng lương cho phòng nhân sự"),
]

QUERIES = [
    ("1. Tra cứu đơn giản (vector đủ)", "Tôi đã đọc gì về Kubernetes?"),
    ("2. Cần profile", "Gợi ý tôi nên đọc gì tiếp theo"),
    ("3. Hoạt động gần đây", "Dạo này tôi đang tập trung vào cái gì?"),
    ("4. Diễn đạt lại (vector thắng)", "Tài liệu về mở rộng quy mô hạ tầng?"),
    ("5. Hỗn hợp (hybrid + profile)", "Tóm tắt về bảo mật trên cloud cho tôi"),
]


def main() -> int:
    fs = load_feature_store()
    print(f"feature store : {'Feast (app/feast_repo)' if fs else 'chưa có — chạy NB4 trước'}")

    agent = HybridMemoryAgent(feature_store=fs)
    print(f"embedding     : {agent.embedder.model_name} ({agent.embedder.dim}d)")

    for kind, text in SEEDS:
        agent.remember(text, user_id=USER, kind=kind)
    for kind, text in OTHER_SEEDS:
        agent.remember(text, user_id=OTHER, kind=kind)
    print(f"episodic memory: {len(agent.memories)} mục "
          f"({sum(m.user_id == USER for m in agent.memories)} của {USER})")

    for label, q in QUERIES:
        print("\n" + "═" * 78)
        print(label)
        print("═" * 78)
        res = agent.recall_detailed(q, user_id=USER)
        print(res.context)

    # ── the two failure modes this POC is built to avoid ────────────────
    print("\n" + "═" * 78)
    print("Kiểm tra 1 — semantic cache có namespace (NB7)")
    print("═" * 78)
    repeat = agent.recall_detailed(QUERIES[0][1], user_id=USER)
    print(f"  cùng user hỏi lại        → {'HIT (tiết kiệm)' if repeat.cached else 'MISS'}")
    other = agent.recall_detailed(QUERIES[0][1], user_id=OTHER)
    print(f"  user khác hỏi y hệt      → {'HIT — RÒ RỈ!' if other.cached else 'MISS (đúng)'}")
    agent.cache.advance(3600)           # đồng hồ ảo: quá TTL 900 s
    after_ttl = agent.recall_detailed(QUERIES[0][1], user_id=USER)
    print(f"  sau khi quá TTL 900 s    → {'HIT (sai)' if after_ttl.cached else 'MISS (đúng)'}"
          f"   stale_evictions={agent.cache.stats.stale_evictions}")

    print("\n" + "═" * 78)
    print("Kiểm tra 2 — cô lập ký ức giữa các user (filtered-ANN, NB5)")
    print("═" * 78)
    leaked = [m for m in agent.search_memories("doanh thu quý 3", USER, top_k=5)
              if m.user_id != USER]
    print(f"  {USER} truy xuất 'doanh thu quý 3' → {len(leaked)} ký ức của user khác "
          f"({'ĐẠT' if not leaked else 'HỎNG'})")
    mine = agent.search_memories("doanh thu quý 3", OTHER, top_k=5)
    print(f"  {OTHER} truy xuất cùng câu đó      → {len(mine)} ký ức, "
          f"tất cả của chính mình: {all(m.user_id == OTHER for m in mine)}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
