# Ghi chú kỹ thuật — những chỗ tôi sửa so với bản template

Lab này ship sẵn code cho cả 4 TODO, nên phần việc thật nằm ở chạy — và chạy
thì lộ ra vài chỗ hỏng. Dưới đây là toàn bộ thay đổi tôi làm ngoài việc điền
notebook, kèm lý do. **Không đụng vào dữ liệu gốc của BTC**
(`scripts/seed_corpus.py`, corpus, golden set, ngưỡng rubric) — mọi con số dưới
đây đo trên đúng bộ dữ liệu do `make seed` sinh ra, seed=42.

---

## 1. `pyproject.toml` — `make test` không chạy test nào

```diff
-testpaths = ["app", "scripts"]
+testpaths = ["tests"]
```

`make test` báo `no tests ran in 0.00s` và **exit 0**, nên nó "xanh" theo nghĩa
tệ nhất: xanh vì không kiểm tra gì. Suite thật nằm ở `tests/`. Sau khi sửa:
**41 passed**. README hứa "34 tests" nên con số đã tăng từ lúc viết README.

## 2. `notebooks/03` — port 8000 hard-code

Máy này đã có một service khác chiếm `:8000` (container `day12-chat-service`).
Hai hệ quả, cái thứ hai nguy hiểm hơn:

* `uvicorn` không bind được;
* vòng chờ `/healthz` vẫn nhận **HTTP 200** — từ *service kia* — nên notebook
  chờ đủ 60 s rồi báo "API didn't become ready", một thông báo trỏ sai tầng.

Sửa: `free_port()` lấy 8000 nếu trống, không thì xin OS một port tự do (máy
sạch vẫn ra đúng 8000); thêm `proc.poll()` để phát hiện uvicorn chết ngay lập
tức thay vì đợi hết deadline.

## 3. `notebooks/03` — deadline 60 s quá ngắn

Startup = load model + embed và index 1000 doc. Trên máy này bước đó mất
217 s. Nâng deadline lên 900 s và in thời gian thực tế. Thêm warm-up 10 query ×
3 mode trước khi đo: với 100 mẫu, `percentile(0.99)` **chính là mẫu tệ nhất**,
nên một request lạnh quyết định toàn bộ con số P99. Production đo tail latency
ở steady state; cold start được đo riêng ở §2.

## 4. `app/embeddings.py` — số thread ONNX (đây là fix có giá trị nhất)

P99 hybrid ban đầu **96,1 ms**, gấp đôi ngưỡng rubric 50 ms. BM25 chỉ tốn
5,7 ms — toàn bộ ngân sách nằm ở một lượt embedding query.

Nguyên nhân không phải model chậm mà là **oversubscription**: ONNX Runtime mặc
định mở một intra-op thread mỗi core, và bge-small (33M tham số) quá nhỏ để lấp
đầy 20 core, nên chi phí fan-out/join lấn át chính phép nhân ma trận.

| threads | query p50 | query p99 | index 1000 doc |
|---|---|---|---|
| 4 | 20,6 ms | 29,2 ms | ~128 s |
| 8 | 20,8 ms | 28,9 ms | ~129 s |
| **mặc định (20)** | **33,0 ms** | **54,1 ms** | **~199 s** |

Không có đánh đổi nào ở đây — chặn thread nhanh hơn **cả hai** chiều. Đặt trần
`min(8, cpu_count)` (override bằng `EMBED_THREADS`) nên laptop 4–8 core không
mất gì, chỉ máy nhiều core bị giữ lại. Hiệu ứng phụ đo được: NB1 chạy từ
**765 s xuống 137 s**.

## 5. `notebooks/04` — PIT join trả 2 dòng thay vì 3

Rubric đòi 3 dòng. `make_user_profile()` ghi feature của `u_001` tại `NOW-1h`,
nhưng `entity_df` hỏi tại `NOW-2h` — **trước khi feature tồn tại** — nên Feast
đúng khi không trả giá trị. Đây không phải bug của Feast mà là entity_df đặt sai
thời điểm.

Sửa: đưa mốc hỏi ra **sau** mốc ghi feature (đúng thứ tự production: feature
tính trước, sự kiện cần chấm điểm xảy ra sau) → 3/3 dòng. Và thêm một cell
*cố tình* hỏi ngược về quá khứ để thấy PIT join từ chối dùng giá trị tương lai —
phần đáng học của cơ chế này.

## 6. `notebooks/06` — sửa một nhận định của lab

Bản gốc viết: bật filter suy đoán làm giảm recall nhưng "đổi lại, nó tốn ít call
hơn". Số đo nói ngược: **cả hai đều 2,3 call**, latency cùng bậc. Trên corpus
này filter suy đoán là **lỗ thuần** (recall 0,906 → 0,823, không mua lại được
gì). Tôi viết lại đoạn đó kèm cơ chế thật và điều kiện để filter *đáng* bật.

## 7. `app/embeddings.py` — thêm backend `multilingual-small`

Để trả lời câu hỏi của NB2 bằng thí nghiệm chứ không bằng phỏng đoán — xem
`scripts/embedding_ablation.py` và `submission/screenshots/embedding_ablation.txt`.
Chọn `paraphrase-multilingual-MiniLM-L12-v2` (384d, 0,22 GB) thay vì `bge-m3`
(1024d, 2,2 GB) mà README gợi ý, vì **cùng số chiều và cùng bậc latency** với
bge-small — nhờ vậy thí nghiệm tách được *độ phủ ngôn ngữ* khỏi *kích thước
model*. So với một model lớn gấp 10 lần thì không tách được.

BM25 là **cột control**: giống hệt nhau ở cả hai lần chạy, nên mọi chênh lệch
đều thuộc về embedding.

| slice | backend | keyword | semantic | hybrid |
|---|---|---|---|---|
| exact | fastembed | 96,7% | 88,7% | 96,7% |
| exact | multilingual-small | 96,7% | 90,0% | 98,0% |
| **paraphrase** | fastembed | 33,3% | **24,0%** | 32,0% |
| **paraphrase** | multilingual-small | 33,3% | **48,0%** | 44,0% |
| mixed | fastembed | 97,0% | 98,5% | 100,0% |
| mixed | multilingual-small | 97,0% | 86,0% | 95,0% |

Tổng thể hybrid 78,6% → **80,6%**, và **rẻ hơn**: hybrid P99 33,8 → 30,7 ms,
index 135 → 120 s. Nhưng không miễn phí — `mixed` tụt 100% → 95%.

**Quyết định:** giữ `fastembed` làm mặc định đã commit (mọi ngưỡng rubric hiệu
chỉnh theo nó, và nó đúng 2/3 kỳ vọng slice), còn `multilingual-small` là
**khuyến nghị cho production phục vụ người dùng Việt** — nhân đôi recall trên
câu diễn đạt lại với latency thấp hơn thì đáng. Đổi bằng một dòng env,
`EMBEDDING_BACKEND=multilingual-small`, không phải sửa code.

---

## Những gì tôi **không** đổi, và vì sao

* **`app/feast_repo/feature_store.yaml` giữ nguyên SQLite + Parquet.** Đường
  Docker được chứng minh riêng bằng `scripts/docker_path_report.py`, sinh ra một
  Feast repo thứ hai (`app/feast_repo_docker/`, đã gitignore). Lý do: tiêu chí
  "reproducible from clean `bash setup-lite.sh`" đòi repo phải chạy được khi
  **không có Docker**. Trỏ config đã commit vào Redis/Postgres là đánh đổi 5 điểm
  reproducibility lấy một dòng cấu hình.
* **`docker-compose.yml` giữ nguyên.** Máy này đã có `redis-server` chiếm
  `127.0.0.1:6379` (pid 210, chạy 9 ngày) nên Redis của lab phải publish sang
  6382 — nhưng qua một overlay **ngoài repo**, không commit. Commit vào là đổi
  port cho cả người chấm, trong khi `scripts/verify_docker.py` hard-code 6379.
  > Bẫy đi kèm: trên máy này `make verify-docker` sẽ báo **PASS giả** — nó ping
  > `localhost:6379` và con Redis *của host* trả lời vui vẻ. Nên tôi kiểm tra
  > bằng `docker_path_report.py` trỏ đúng cổng, không dựa vào script đó.
* **Dữ liệu gốc.** `seed_corpus.py`, `gen_agent_queries.py`, `gen_spend.py`,
  corpus, golden set: không sửa một dòng nào.

## Đường Docker: đã chạy thật, không chỉ "bật container lên"

`submission/screenshots/docker_path_report.txt` — Qdrant server + Redis (online
store) + Postgres (offline store), cùng phép đo như NB3/NB4:

| | lite | docker |
|---|---|---|
| Feast online lookup P99 | 1,35 ms (SQLite) | **1,11 ms** (Redis, qua TCP) |
| hybrid P@10 | 78,6% | **78,6%** |
| hybrid P99 | 32,6 ms | 33,6 ms |

Chất lượng **không đổi** — đúng như phải thế: đổi nơi cất vector không đổi
vector. Cái đổi là độ bền, khả năng nhiều replica dùng chung, và một chặng mạng
thêm vào mỗi lần đọc.

Ba chỗ vấp đáng ghi lại, vì không cái nào xuất hiện trong tài liệu lab:

1. **`sqlalchemy` mặc định dịch `postgresql://` thành psycopg2**, trong khi
   feast dùng psycopg 3 → `ModuleNotFoundError: psycopg2`. Phải ghi rõ driver:
   `postgresql+psycopg://`.
2. **Feast mặc định `sslmode=require`**, còn image `postgres:16-alpine` build
   không có SSL → `materialize` chết với *"server does not support SSL, but SSL
   was required"*. Thêm `sslmode: disable` (chấp nhận được với container
   loopback; production giữ TLS).
3. **`feast` CLI không nằm trên PATH** khi chạy `.venv/bin/python script.py` mà
   không activate venv — đúng vấn đề `notebooks/_setup.py` đã xử lý cho notebook.

## Môi trường có một lỗi thật, ghi lại để người sau đỡ mất thời gian

WSL2 host này có đồng hồ tường **vọt tiến +550 s trong khoảng 2% số lần đọc**
(đo được 280.160 lần vọt trên 13,5 triệu mẫu trong 6 giây). `time.perf_counter()`
và `time.monotonic()` không bị ảnh hưởng, mà lab chỉ đọc wall clock ở hai chỗ
tự-nhất-quán, nên **không có phép đo nào trong bài này bị sai**. Dấu vết duy
nhất: script chạy notebook của tôi (dùng `date +%s`) từng log NB2 chạy hết
**-327 giây**.
