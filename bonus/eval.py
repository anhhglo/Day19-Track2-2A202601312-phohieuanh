"""Golden set for the memory agent — measure the claims instead of asserting them.

ARCHITECTURE.md makes four falsifiable claims. Every one is cheap to test and
none of them was tested when it was written:

  1. hybrid RRF beats either retriever alone on personal memory;
  2. BM25 is what saves code-switched queries (`Kubernetes`, `pgvector`,
     `Nghị định 13`) that an English embedder mangles in Vietnamese context;
  3. the profile leg earns its keep on queries with no topical signal;
  4. consolidation buys context, because top-K is a fixed budget.

A bonus that argues for measurement while shipping unmeasured claims would be
the same mistake NB6 warns about, one level up.

    python bonus/eval.py
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
DAY = 86400.0
NOW = 1_760_000_000.0          # fixed epoch: the whole eval is reproducible

# (tag, kind, age_days, text)
SEEDS: list[tuple[str, str, float, str]] = [
    ("k8s_hpa",    "read",  40, "Đã đọc: Kubernetes HPA tự động scale pod theo CPU và custom metric"),
    ("karpenter",  "read",  35, "Đã đọc: so sánh Karpenter và Cluster Autoscaler khi node pool co giãn"),
    ("spot",       "note",  30, "Ghi chú: spot instance rẻ hơn 70% nhưng phải chịu được interruption 2 phút"),
    ("rightsize",  "note",  25, "Ghi chú: cắt bớt CPU request thừa giúp giảm hoá đơn hạ tầng hàng tháng"),
    ("mtls",       "read",  20, "Đã đọc: mTLS giữa các service trong service mesh Istio"),
    ("oauth",      "read",  18, "Đã đọc: OAuth2 và JWT cho xác thực giữa các microservice"),
    ("nd13",       "note",  15, "Ghi chú: Nghị định 13/2023 yêu cầu dữ liệu cá nhân của người dùng Việt Nam "
                               "phải có cơ sở pháp lý xử lý rõ ràng"),
    ("pgvector",   "read",  12, "Đã đọc: pgvector HNSW index và cách chọn ef_search"),
    ("btree",      "read",  10, "Đã đọc: chỉ mục B-tree tăng tốc truy vấn lọc theo khoảng giá trị"),
    ("replica",    "read",   9, "Đã đọc: đảm bảo tính nhất quán giữa nhiều bản sao cơ sở dữ liệu"),
    ("rrf",        "read",   8, "Đã đọc: RRF k=60 hợp nhất BM25 với vector search"),
    ("bgem3",      "note",   7, "Ghi chú: bge-m3 đa ngữ tốt cho tiếng Việt nhưng nặng gấp 4 lần bge-small"),
    ("finetune",   "read",   6, "Đã đọc: tinh chỉnh mô hình trên tập dữ liệu của riêng mình"),
    ("lb_region",  "read",   5, "Đã đọc: cân bằng tải giữa nhiều region để giảm độ trễ end-to-end"),
    ("bluegreen",  "read",   4, "Đã đọc: triển khai blue-green giảm rủi ro khi phát hành"),
    ("rollback",   "note",   3, "Ghi chú: tự động rollback khi error rate vượt ngưỡng sau khi phát hành"),
    ("circuit",    "query",  2, "Tìm kiếm: circuit breaker tránh cascading failure giữa microservice"),
    ("apk",        "read",  60, "Đã đọc: tối ưu kích thước APK cho ứng dụng di động"),
    ("lcp",        "read",  55, "Đã đọc: tối ưu LCP và FID cho Core Web Vitals"),
    ("kafka",      "read",  50, "Đã đọc: xử lý sự kiện streaming với Kafka và Flink"),
]

# (query, {tag đúng}, slice)
#   chuyen_ma   = truy vấn chứa token ASCII/đúng-nguyên-văn có trong ký ức
#   dien_dat_lai = không trùng chữ nào, chỉ trùng ý
GOLDEN: list[tuple[str, set[str], str]] = [
    ("Tôi đã đọc gì về Kubernetes?",              {"k8s_hpa", "karpenter"},   "chuyen_ma"),
    ("pgvector cấu hình ef_search thế nào",       {"pgvector"},               "chuyen_ma"),
    ("RRF hoạt động ra sao",                      {"rrf"},                    "chuyen_ma"),
    ("Nghị định 13 nói gì",                       {"nd13"},                   "chuyen_ma"),
    ("Istio thì tôi ghi gì",                      {"mtls"},                   "chuyen_ma"),
    ("Kafka và Flink",                            {"kafka"},                  "chuyen_ma"),
    ("làm sao giảm tiền hạ tầng hàng tháng",      {"spot", "rightsize"},      "dien_dat_lai"),
    ("bảo vệ thông tin riêng tư của khách hàng",  {"nd13"},                   "dien_dat_lai"),
    ("các service tin nhau bằng cách nào",        {"mtls", "oauth"},          "dien_dat_lai"),
    ("cách cho câu truy vấn chạy nhanh hơn",      {"btree", "pgvector"},      "dien_dat_lai"),
    ("phát hành mà ít rủi ro nhất",               {"bluegreen", "rollback"},  "dien_dat_lai"),
    ("chia tải cho người dùng ở xa",              {"lb_region"},              "dien_dat_lai"),
]

STRATEGIES: dict[str, tuple[str, ...]] = {
    "bm25 đơn":        ("bm25",),
    "vector đơn":      ("vector",),
    "hybrid (RRF)":    ("bm25", "vector"),
    "hybrid+profile":  ("bm25", "vector", "profile"),
}
TOP_K = 5


def build(agent: HybridMemoryAgent) -> dict[str, int]:
    tag_of_id: dict[str, int] = {}
    for tag, kind, age, text in SEEDS:
        agent.remember(text, user_id=USER, kind=kind, ts=NOW - age * DAY)
        tag_of_id[tag] = agent.memories[-1].memory_id
    return tag_of_id


def recall_at_k(got_ids: list[int], want: set[int]) -> float:
    return len(set(got_ids) & want) / len(want) if want else 1.0


def main() -> int:
    fs = load_feature_store()
    agent = HybridMemoryAgent(feature_store=fs, half_life_days=30.0)
    tag_of_id = build(agent)
    prof = agent.profile(USER)
    affinity = prof.get("topic_affinity") if "_error" not in prof else None

    print(f"Golden set trí nhớ — {len(SEEDS)} ký ức, {len(GOLDEN)} truy vấn, Recall@{TOP_K}")
    print(f"embedding : {agent.embedder.model_name} ({agent.embedder.dim}d)")
    print(f"affinity  : {affinity or '(không có — chạy NB4 trước)'}")
    print("=" * 78)

    # ── 1. bốn chiến lược, cùng ngân sách top-5 ─────────────────────────
    slices = sorted({s for _, _, s in GOLDEN})
    print(f"\n{'chiến lược':<18}{'tổng':>8}" + "".join(f"{s:>16}" for s in slices))
    results: dict[str, float] = {}
    for name, legs in STRATEGIES.items():
        per_slice: dict[str, list[float]] = {s: [] for s in slices}
        allr: list[float] = []
        for q, want_tags, sl in GOLDEN:
            want = {tag_of_id[t] for t in want_tags}
            mems = agent.search_memories(q, USER, TOP_K, affinity=affinity,
                                         legs=legs, use_decay=False)
            r = recall_at_k([m.memory_id for m in mems], want)
            allr.append(r)
            per_slice[sl].append(r)
        results[name] = sum(allr) / len(allr)
        row = f"{name:<18}{results[name]:>7.1%}"
        for s in slices:
            v = per_slice[s]
            row += f"{sum(v)/len(v):>15.1%} "
        print(row)

    best = max(results, key=results.get)
    print(f"\n  → tốt nhất: {best} ({results[best]:.1%})")

    # ── 2. decay: ký ức cũ có nên thắng ký ức mới không ─────────────────
    print("\n" + "=" * 78)
    print("2. Memory decay — cùng chủ đề, khác tuổi")
    print("=" * 78)
    agent.remember("Ghi chú: giá thuê GPU A100 khoảng 3 USD mỗi giờ",
                   user_id=USER, kind="note", ts=NOW - 400 * DAY)
    old_id = agent.memories[-1].memory_id
    agent.remember("Ghi chú: giá thuê GPU A100 khoảng 2 USD mỗi giờ",
                   user_id=USER, kind="note", ts=NOW - 2 * DAY)
    new_id = agent.memories[-1].memory_id
    q = "giá thuê GPU bây giờ bao nhiêu"
    for label, decay in (("tắt decay", False), ("bật decay (nửa đời 30 ngày)", True)):
        ids = [m.memory_id for m in agent.search_memories(
            q, USER, TOP_K, affinity=affinity, use_decay=decay, now_ts=NOW)]
        pos_old = ids.index(old_id) + 1 if old_id in ids else None
        pos_new = ids.index(new_id) + 1 if new_id in ids else None
        print(f"  {label:<30} ghi chú 2 ngày tuổi ở hạng {pos_new}, "
              f"ghi chú 400 ngày tuổi ở hạng {pos_old}")

    # ── 3. consolidation: dọn trùng để mua lại slot ─────────────────────
    print("\n" + "=" * 78)
    print("3. Consolidation — top-K là ngân sách cố định")
    print("=" * 78)
    dupes = [
        "Ghi chú: spot instance rẻ hơn 70% nhưng phải chịu được interruption 2 phút",
        "Ghi chú: spot instance rẻ hơn 70%, nhưng phải chịu được interruption 2 phút",
        "Ghi chú: spot instance rẻ hơn 70% nhưng phải chịu interruption 2 phút",
    ]
    for i, d in enumerate(dupes):
        agent.remember(d, user_id=USER, kind="note", ts=NOW - (29 - i) * DAY)

    # Truy vấn phải THỰC SỰ kéo được nhóm trùng về, nếu không phép đo vô nghĩa.
    # Lần đo đầu tôi dùng "làm sao giảm tiền hạ tầng hàng tháng" và thấy 5/5 →
    # 5/5: nhóm trùng không hề lọt top-5 nên chẳng có slot nào để mua lại. Đó
    # là kết quả thật, và nó thu hẹp phạm vi của luận điểm — xem ARCHITECTURE.md.
    q = "spot instance"

    def so_su_that(mems) -> int:
        """Đếm SỰ THẬT riêng biệt trong top-K, không đếm chuỗi riêng biệt.

        Ba cách diễn đạt của cùng một ghi chú là ba chuỗi khác nhau nhưng chỉ
        một sự thật — và người dùng chỉ nhận được một.
        """
        return len({"spot" if "spot instance" in m.text else m.text for m in mems})

    before = agent.search_memories(q, USER, TOP_K, affinity=affinity, use_decay=False)
    live_before = sum(1 for m in agent.memories if m.user_id == USER and m.alive)
    n_before = so_su_that(before)

    folded = agent.consolidate(USER, threshold=0.92)

    after = agent.search_memories(q, USER, TOP_K, affinity=affinity, use_decay=False)
    live_after = sum(1 for m in agent.memories if m.user_id == USER and m.alive)
    n_after = so_su_that(after)

    print(f"  ký ức sống       : {live_before} → {live_after}  (gộp {folded})")
    print(f"  top-5 cho {q!r}:")
    print(f"    sự thật riêng biệt TRƯỚC khi gộp : {n_before}/{TOP_K}")
    print(f"    sự thật riêng biệt SAU khi gộp   : {n_after}/{TOP_K}")
    kept = [m for m in agent.memories if m.merged_count > 1 and m.alive]
    for m in kept:
        print(f"    giữ lại (gộp {m.merged_count} bản): {m.text[:62]}…")

    gpu = [m for m in agent.memories if "GPU A100" in m.text]
    print(f"\n  Hàng rào con số — hai ghi chú giá GPU (3 USD / 2 USD, cosine ≈ 0,99):")
    for m in gpu:
        print(f"    {'còn sống' if m.alive else 'ĐÃ BỊ XOÁ'}: {m.text}")
    print(f"    → cả hai còn sống: {all(m.alive for m in gpu)}  "
          f"(trước khi thêm `_cung_so()` thì một cái bị nuốt)")

    print("\nXong.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
