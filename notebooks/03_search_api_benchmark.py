# ---
# jupyter:
#   jupytext:
#     formats: py:percent
# ---

# %% [markdown]
# # NB3 — FastAPI `/search` Endpoint + Latency Benchmark
#
# **Stack:** FastAPI + uvicorn + httpx (client). Searcher từ `app/search.py`.
# Maps to slide §7 (Production Patterns) + deliverable bullets 1, 4.
#
# > Mục tiêu: bọc `Searcher` thành REST API, đo P50/P95/P99 latency, đảm bảo
# > P99 < 50 ms cho hybrid mode (rubric threshold).

# %%
import _setup  # noqa: F401
import socket
import statistics
import subprocess
import time
from pathlib import Path

import httpx

# %% [markdown]
# ## 1. Khởi động API server (background)
#
# Trong production thực tế, bạn sẽ chạy `make api` ở terminal riêng. Notebook
# này khởi động uvicorn ở background subprocess và đợi `/healthz` trả ready.
#
# Port được **chọn động**: lấy 8000 nếu còn trống, không thì xin OS một port
# tự do. Hard-code 8000 là một giả định về máy người khác — nếu port đó đã bị
# service khác chiếm, uvicorn chết ngay còn vòng `/healthz` bên dưới lại đi
# hỏi **nhầm service**, và cell chỉ hỏng sau 60 s với thông báo sai chỗ.

# %%
def free_port(preferred: int = 8000) -> int:
    """`preferred` nếu bind được, không thì để OS cấp một port trống."""
    for candidate in (preferred, 0):
        with socket.socket() as s:
            # No SO_REUSEADDR on purpose: it can let this probe bind
            # 127.0.0.1:8000 while another process already listens on
            # 0.0.0.0:8000, so the probe would report "free" and uvicorn would
            # then fail. A TIME_WAIT false negative just falls through to an
            # OS-assigned port, which costs nothing.
            try:
                s.bind(("127.0.0.1", candidate))
            except OSError:
                continue
            return s.getsockname()[1]
    raise RuntimeError("no free port")


ROOT = Path(_setup.__file__).resolve().parent.parent
PORT = free_port()
print(f"API port: {PORT}")
proc = subprocess.Popen(
    ["uvicorn", "app.main:app", "--port", str(PORT), "--log-level", "warning"],
    cwd=str(ROOT),
)

# Đợi server up + warm. Startup = load model + embed và index 1000 doc, nên
# deadline phải tính theo *máy chậm nhất*, không phải máy của người viết lab:
# trên box này (CPU chia sẻ với các container khác) bước đó mất ~10 phút, còn
# 60 s chỉ đủ cho một laptop rảnh. Và nếu uvicorn chết (port bận, import lỗi)
# thì `proc.poll()` cho biết ngay, thay vì đợi hết deadline rồi báo "not ready"
# — một thông báo trỏ sai tầng.
URL = f"http://localhost:{PORT}"
DEADLINE_S = 900
t_start = time.perf_counter()
ready = False
while time.perf_counter() - t_start < DEADLINE_S:
    if proc.poll() is not None:
        raise RuntimeError(f"uvicorn thoát sớm với mã {proc.returncode} — xem log ở trên")
    try:
        r = httpx.get(f"{URL}/healthz", timeout=2.0)
        if r.status_code == 200 and r.json().get("ready"):
            ready = True
            break
    except httpx.HTTPError:
        pass
    time.sleep(2)
if not ready:
    proc.terminate()
    raise RuntimeError(f"API chưa ready sau {DEADLINE_S}s")
print(f"ready sau {time.perf_counter() - t_start:.0f}s")

print(httpx.get(f"{URL}/healthz").json())

# %% [markdown]
# ## 2. Single query — kiểm tra response shape

# %%
r = httpx.get(f"{URL}/search", params={"q": "cloud computing tự động mở rộng", "mode": "hybrid"})
r.raise_for_status()
body = r.json()
print(f"latency_ms: {body['latency_ms']:.1f}")
print(f"top-3 hits:")
for h in body["hits"][:3]:
    print(f"  {h['doc_id']:>14}  score={h['score']:.4f}  {h['title']}")

# %% [markdown]
# ## 3. TODO — Latency benchmark (100 queries × 3 modes)
#
# Dùng 50 golden queries × 2 reps = 100 calls/mode. Ghi nhận latency từ
# `body["latency_ms"]` (server-side, đã trừ network) HOẶC từ wall-clock httpx
# (bao gồm network) — note: rubric assert P99 < 50ms áp dụng cho server-side.
#
# Output: bảng P50/P95/P99 cho 3 mode.
#
# **Warm-up là một phần của phép đo, không phải mẹo làm đẹp số.** Với 100 mẫu,
# `percentile(0.99)` chính là *mẫu tệ nhất*, nên một request lạnh duy nhất
# (ONNX session khởi tạo lazy, allocator chưa ổn định) quyết định toàn bộ con
# số P99. Production đo tail latency ở trạng thái steady-state — instance mới
# được warm trước khi load balancer đưa traffic vào — nên ta warm ở đây cho
# phép đo trả lời đúng câu hỏi "hệ thống này chạy nhanh thế nào", chứ không
# phải "lần gọi đầu tiên chậm thế nào". Cold start được đo riêng ở §2.

# %%
import json

DATA = ROOT / "data"
golden = [json.loads(l) for l in (DATA / "golden_set.jsonl").open(encoding="utf-8")]

WARMUP = 10
for mode in ("keyword", "semantic", "hybrid"):
    for q in golden[:WARMUP]:
        httpx.get(f"{URL}/search", params={"q": q["query"], "mode": mode})
print(f"warm-up xong: {WARMUP} query × 3 mode")


def percentile(values: list[float], p: float) -> float:
    n = len(values)
    if n == 0:
        return 0.0
    return sorted(values)[min(int(n * p), n - 1)]


def benchmark_mode(mode: str, reps: int = 2) -> dict[str, float]:
    server_latencies: list[float] = []
    wall_latencies: list[float] = []
    for _ in range(reps):
        for q in golden:
            t0 = time.perf_counter()
            r = httpx.get(f"{URL}/search", params={"q": q["query"], "mode": mode})
            wall_latencies.append((time.perf_counter() - t0) * 1000)
            server_latencies.append(r.json()["latency_ms"])
    return {
        "p50_server": percentile(server_latencies, 0.50),
        "p95_server": percentile(server_latencies, 0.95),
        "p99_server": percentile(server_latencies, 0.99),
        "p99_wall":   percentile(wall_latencies, 0.99),
    }


print(f"  {'mode':10}  {'P50':>7}  {'P95':>7}  {'P99':>7}  {'P99(wall)':>9}")
results = {}
for mode in ("keyword", "semantic", "hybrid"):
    res = benchmark_mode(mode)
    results[mode] = res
    print(f"  {mode:10}  {res['p50_server']:>5.1f}ms  {res['p95_server']:>5.1f}ms  "
          f"{res['p99_server']:>5.1f}ms  {res['p99_wall']:>7.1f}ms")

# %% [markdown]
# ## 4. Rubric assertion — hybrid P99 server-side < 50ms

# %%
hybrid_p99 = results["hybrid"]["p99_server"]
print(f"Hybrid P99 server-side: {hybrid_p99:.1f}ms")
if hybrid_p99 < 50:
    print(f"PASS — hybrid P99 < 50ms ({hybrid_p99:.1f}ms)")
else:
    print(f"WARN — hybrid P99 >= 50ms ({hybrid_p99:.1f}ms)")
    print("  Possible causes: cold cache, fastembed model not warm yet, or RRF depth=50 is too aggressive")
    print("  Check: re-run benchmark after 10 warm-up queries; or reduce RRF depth")

# %% [markdown]
# ## 5. Cleanup — stop the API server

# %%
proc.terminate()
proc.wait(timeout=5)
print("API server stopped")

# %% [markdown]
# ## Deliverable evidence
#
# 1. Output cell 2: 1 single hybrid query response with `top-3 hits`.
# 2. Output cell 3: latency table P50/P95/P99 for keyword/semantic/hybrid.
# 3. Output cell 4: hybrid P99 < 50ms PASS.
#
# ---
#
# ## Vibe-coding callout
#
# **Delegate freely:** the FastAPI scaffolding (route definition, Pydantic
# response model, lifespan handler). AI generates this perfectly given the
# spec "GET /search?q=str&mode=Literal[...] returning SearchResponse with
# latency_ms field". `app/main.py` is exactly that pattern — review the diff,
# don't write it from scratch.
#
# **Think hard yourself:** *what to measure*. Server-side latency vs wall-clock
# vs client-side. P50 vs P95 vs P99. Cold vs warm. Single user vs concurrent.
# These are *judgement* decisions: nếu rubric chỉ check P99, optimization sẽ
# hướng vào tail latency, không phải mean. Đừng nhờ AI quyết định metric —
# chỉ nhờ implement metric đã chọn.
