# ---
# jupyter:
#   jupytext:
#     formats: py:percent
# ---

# %% [markdown]
# # NB6 — Agentic Retrieval: truy xuất như một *tool*
#
# **Stack:** `app.agent` (tool schema + planner + reflection) trên `FilteredIndex`.
# Maps to deck §6 "Retrieval Như Một Tool" + "Ghép Ngữ Cảnh".
#
# > Trong RAG cổ điển, retrieval là một *bước* trong pipeline. Với agent, nó là
# > một **tool** mà model tự quyết định gọi — gọi mấy lần, với filter nào. Chênh
# > lệch chất lượng nằm ở chỗ đó, và notebook này đo nó.
#
# Planner ở đây là **rule-based, không gọi LLM** — lab chạy zero-key. Bài học là
# về *chiến lược truy xuất* (tách câu hỏi, suy ra filter, phản tỉnh, thử lại),
# và bài học đó rõ hơn khi planner có thể đọc được từng dòng.

# %%
import _setup  # noqa: F401
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

from app.agent import (SEARCH_TOOL, Agent, RetrievalTool, RuleBasedPlanner,
                       SingleShotPlanner, build_context)
from app.filters import FilteredIndex
from app.search import Searcher

DATA = Path(_setup.__file__).resolve().parent.parent / "data"

# %% [markdown]
# ## 1. Tool definition — thứ agent thực sự nhìn thấy
#
# Đây là toàn bộ thông tin agent có khi quyết định *có gọi retrieval hay không*.
# Nếu `description` mơ hồ, agent gọi sai lúc — và không có log nào nói cho bạn biết.

# %%
print(json.dumps(SEARCH_TOOL, ensure_ascii=False, indent=2)[:900])

# %% [markdown]
# Hai chi tiết đáng chú ý:
#
# * `description` **chính là prompt truy xuất**. Nó không phải comment cho người đọc.
# * `topic` là **`enum`**, không phải string tự do. Agent không thể bịa ra một topic
#   không tồn tại → filter không bao giờ âm thầm khớp 0 document.

# %%
searcher = Searcher.from_corpus(DATA / "corpus_vn.jsonl")
index = FilteredIndex.from_searcher(searcher)
tool = RetrievalTool(index)

# %% [markdown]
# ## 2. Planner: một câu hỏi, mấy ý định?
#
# Câu hỏi thật của người dùng thường có **nhiều hơn một ý định**. Một embedding
# duy nhất của câu ghép sẽ rơi vào *khoảng giữa* hai cụm — gần cả hai, thuộc về
# không cụm nào.

# %%
planner = RuleBasedPlanner(budget=16)
demo_q = "tự động mở rộng theo lưu lượng và cân bằng tải giữa nhiều region"
for i, args in enumerate(planner.plan(demo_q), 1):
    print(f"  call {i}: {args.as_dict()}")

# %% [markdown]
# ## 3. Đo: single-shot vs agentic, **cùng ngân sách truy xuất**
#
# Đây là chỗ dễ đo gian lận nhất. Nếu agent được lấy 32 doc còn single-shot chỉ
# 16, agent thắng vì *ngân sách*, không phải vì *chiến lược*. Ở đây cả hai đều
# lấy đúng **16 document** — chỉ khác cách chia.
#
# Ngoài `recall`, ta đo thêm **`balance`**: trong 16 doc lấy về, hai vế của câu
# hỏi được phủ đều đến đâu (1.00 = đều hoàn hảo, 0.00 = bỏ hẳn một vế).

# %%
queries = [json.loads(l) for l in (DATA / "agent_queries.jsonl").open(encoding="utf-8")]
BUDGET = 16


def evaluate(agent, label):
    rec, bal, calls, ms = [], [], [], []
    for q in queries:
        r = agent.answer(q["question"])
        truth, got = set(q["relevant_doc_ids"]), set(r.doc_ids)
        rec.append(len(truth & got) / len(truth))
        a, b = len(set(q["gold_a"]) & got), len(set(q["gold_b"]) & got)
        bal.append(min(a, b) / max(1, max(a, b)))
        calls.append(r.n_calls)
        ms.append(r.latency_ms)
    n = len(queries)
    print(f"{label:<14}{sum(rec)/n:8.3f}{sum(bal)/n:9.2f}{sum(calls)/n:8.1f}{sum(ms)/n:9.1f}")
    return sum(rec) / n


print(f"{'strategy':<20}{'recall':>8}{'balance':>9}{'calls':>8}{'ms':>9}")
base = evaluate(Agent(tool, SingleShotPlanner(budget=BUDGET)), "single-shot")
split = evaluate(Agent(tool, RuleBasedPlanner(budget=BUDGET, use_filters=False)),
                 "agentic (no filter)")
filt = evaluate(Agent(tool, RuleBasedPlanner(budget=BUDGET, use_filters=True)),
                "agentic (+filter)")
print(f"\nΔ recall vs single-shot:  tách câu {split - base:+.3f}   tách + filter {filt - base:+.3f}")

# %% [markdown]
# **Đọc kết quả.** `balance` của single-shot rất thấp: nó gần như chỉ lấy *một*
# vế của câu hỏi. Đó chính xác là lý do RAG một-lượt trả lời "đúng một nửa" cho
# câu hỏi ghép — và vì câu trả lời nghe vẫn trôi chảy, lỗi này rất khó phát hiện.
#
# Nhưng hãy nhìn cột `calls` và `ms`: agentic tốn **nhiều lần gọi hơn** và chậm
# hơn tương ứng. Ở đây mỗi call chỉ là vector search; trong hệ thật mỗi vòng
# planning là một **LLM call** — tức là tiền và latency thật.
#
# > Quy tắc rút ra: **classic RAG cho tra cứu đơn giản, agentic cho câu hỏi
# > nhiều phần.** Đừng bật agentic cho mọi query.
#
# **Và hãy so hai dòng agentic với nhau** — đây là chỗ rubric hỏi *vì sao*.
#
# `agentic (+filter)` **thấp hơn** `agentic (no filter)` ở cả recall lẫn balance.
# Cơ chế: `RuleBasedPlanner.detect_topic()` đoán `topic` bằng cách dò keyword
# (`TOPIC_HINTS` trong `app/agent.py`), rồi cái đoán đó trở thành một
# `FieldCondition` **cứng** trong Qdrant. Ground truth của NB6 lại là top-8
# brute-force **không filter** cho mỗi vế — nên mọi document đúng nhưng bị
# `seed_corpus.py` gán sang cụm bên cạnh đều bị loại **trước khi** được xếp
# hạng. Filter không hạ điểm chúng, nó xoá chúng khỏi danh sách.
#
# Điểm quan trọng: một keyword đoán sai đúng một lần là mất trọn một vế của câu
# hỏi, và recall của vế đó về 0 — đó là lý do `balance` tụt mạnh hơn `recall`.
#
# **Sửa lại một câu trong bản gốc của lab.** Có thể nghĩ "đổi lại, filter tốn ít
# call hơn". Số đo trên máy này nói ngược: **cả hai đều 2,3 call**, và bản có
# filter **không nhanh hơn** — số call bằng nhau nên nó chỉ thêm điều kiện
# payload vào mỗi lần gọi (lần đo này 60,2 ms so với 51,4 ms; hai lần chạy khác
# nhau cho tỉ lệ khác nhau, nhưng chưa lần nào filter thắng về thời gian). Vậy
# trên corpus này filter suy đoán là **lỗ thuần** —
# mất recall mà không mua lại được gì. Đúng bài học NB5 lặp lại ở tầng agent:
# **filter không miễn phí, phải đo chứ đừng đoán.**
#
# Khi nào filter *đáng* bật? Khi topic đến từ nguồn **chắc chắn** thay vì suy
# đoán — ví dụ `topic_affinity` lấy từ feature store (§5 bên dưới), hay một
# scope do người dùng chọn trên UI — hoặc khi corpus lớn tới mức chi phí quét
# vượt quá phần recall bị mất.

# %% [markdown]
# ## 4. Reflection: filter tồi còn tệ hơn không filter
#
# `Agent` thử lại **một lần** với filter được nới ra khi một call trả về quá ít
# bằng chứng. Không có bước này, một filter đoán sai sẽ âm thầm trả về rỗng.

# %%
from app.agent import ToolArgs  # noqa: E402

Q = "cân bằng tải giữa nhiều region"

starving = ToolArgs(query=Q, topic="networking", since_year=2027, top_k=8)
print("filter quá chặt (since_year=2027) →", len(tool(starving).doc_ids), "kết quả")

sane = ToolArgs(query=Q, topic="networking", top_k=8)
print("filter hợp lý                     →", len(tool(sane).doc_ids), "kết quả")


class StarvingPlanner:
    """Cố tình sinh ra một filter không khớp gì, để xem agent xử lý thế nào."""

    def plan(self, question):
        return [ToolArgs(query=question, topic="networking", since_year=2027, top_k=8)]


res = Agent(tool, StarvingPlanner(), min_evidence=4).answer(Q)
print(f"\nagent phản tỉnh: {res.n_calls} call → {len(res.doc_ids)} doc")
for c in res.trace:
    print("   ", c.args, "→", len(c.doc_ids), "kết quả")

# %% [markdown]
# ## 5. Ghép ngữ cảnh: nơi feature store gặp vector store
#
# `build_context()` hỏi **hai câu hỏi khác nhau**:
#
# * **Feature store** → *"user này là ai"* (cá nhân hoá, online lookup <10 ms)
# * **Vector store** → *"cái gì liên quan"* (grounding)
#
# Chạy được cả khi chưa `feast apply` (NB4) — lúc đó `features` rỗng và agent
# chỉ còn grounding. Chạy NB4 trước rồi quay lại đây để thấy khác biệt.

# %%
store = None
try:
    from feast import FeatureStore
    repo = Path(_setup.__file__).resolve().parent.parent / "app" / "feast_repo"
    if (repo / "registry.db").exists():
        store = FeatureStore(repo_path=str(repo))
except Exception as exc:  # noqa: BLE001
    print("Feast chưa sẵn sàng:", exc)

ctx = build_context("u_001", "làm sao tối ưu chi phí hạ tầng", tool, feature_store=store)
print("features   :", ctx["features"] or "(chưa có — chạy NB4 trước)")
print("affinity   :", ctx["affinity_used"])
print("tool_args  :", ctx["tool_args"])
print("doc_ids    :", ctx["doc_ids"][:5], "…")

# %% [markdown]
# ## Deliverable evidence
#
# 1. §1: tool schema in ra, thấy rõ `description` + `enum` của `topic`.
# 2. §3: bảng single-shot vs agentic — Δrecall và Δbalance dương, kèm chi phí calls/ms.
# 3. §4: agent phục hồi được sau một filter sai.
# 4. §5: `build_context()` trả về cả feature lẫn doc_ids.
#
# ---
#
# ## Vibe-coding callout
#
# **Delegate freely:** JSON schema của tool, vòng lặp đánh giá, code in bảng.
# AI viết nhanh và đúng.
#
# **Think hard yourself:** *ngân sách so sánh*. Khi bạn bảo AI "so sánh agentic
# với single-shot", nó gần như luôn cho agent gọi nhiều lần với `top_k` giữ
# nguyên — nghĩa là agent lấy về gấp đôi số document. Kết quả: agent "thắng"
# một cách vô nghĩa. Bài đo chỉ có giá trị khi **tổng số document lấy về bằng
# nhau**. Hãy tự kiểm tra dòng `per = budget // len(parts)` trước khi tin bất kỳ
# con số nào ở §3.
