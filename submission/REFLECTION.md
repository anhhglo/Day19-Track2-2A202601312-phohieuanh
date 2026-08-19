# Reflection — Lab 19

**Path đã chạy:** lite (nộp bài) **và** docker (Qdrant server + Redis + Postgres,
xem `screenshots/docker_path_report.txt`)

---

## Mode nào thắng ở loại query nào, và khi nào **không** dùng hybrid

**exact (n=15).** BM25 96,7% = hybrid 96,7% > vector 88,7%. Query trùng chữ với
title; vector đi tìm hàng xóm ngữ nghĩa nên kéo thêm nhiễu.

**mixed (n=20).** hybrid **100%** > vector 98,5% > BM25 97,0%. Hai retriever sai
ở hai chỗ khác nhau; RRF chỉ cần doc đúng lọt top của *một* bên.

**paraphrase (n=15).** BM25 33,3% > hybrid 32,0% > **vector 24,0%** — ngược sách
vở. Không phải lỗi RRF: cột `keyword` giữ nguyên qua mọi lần chạy nên nó là
control. Ablation chỉ đổi `EMBEDDING_BACKEND` sang
`paraphrase-multilingual-MiniLM-L12-v2` (cùng 384 chiều): vector trên
paraphrase **24,0% → 48,0%**, tổng thể 78,6% → 80,6%, P99 còn giảm 33,8 → 30,7 ms.
Thủ phạm là **model tiếng Anh đọc corpus tiếng Việt**, không phải fusion. Giá
phải trả: `mixed` tụt 100% → 95%.

**Không dùng hybrid khi:** (1) `exact` — BM25 đã 96,7%, hybrid không thêm gì;
(2) tail latency chặt — hybrid P99 32,6 ms so với BM25 0,9 ms, **36×** để đổi
lấy +0,8 điểm P@10; (3) chưa biết phân bố query thật — hybrid thắng tổng thể
phần lớn nhờ `mixed` chiếm 40% golden set.

---

## Điều ngạc nhiên nhất

Ngưỡng semantic cache 0,75 mà AWS công bố để lọt **36%** câu trả lời sai trên
corpus này; phải lên 0,85 mới về 0% mà vẫn giữ trọn phần tiết kiệm. Hằng số của
người khác là **điểm bắt đầu để đo**, không phải giá trị để copy.

Á quân: P99 của hybrid ban đầu là 96,1 ms, gấp đôi ngưỡng rubric — và thứ sửa nó
không phải thuật toán mà là **số thread ONNX**. Mặc định mở một thread mỗi core;
chặn ở 8 làm P99 còn 35,1 ms *và* index nhanh hơn 5,6×. Oversubscription thua ở
cả hai chiều.

---

## Bonus challenge

- [x] Đã làm bonus (xem [`bonus/ARCHITECTURE.md`](../bonus/ARCHITECTURE.md))
- [ ] Pair work với: — (làm một mình)

`bonus/eval.py` là phần tôi thấy đáng nhất: golden set 20 ký ức × 12 truy vấn
**bác bỏ hai luận điểm** tôi đã viết trong ARCHITECTURE.md trước khi đo — hybrid
chỉ *hoà* với BM25 đơn ở tổng thể (87,5%), và leg profile không đổi được con số
nào. Nó cũng lòi ra một lỗi mất dữ liệu: `consolidate()` gộp
*"giá GPU 3 USD/giờ"* với *"giá GPU 2 USD/giờ"* vì cosine ≈ 0,99, tức **xoá mất
một sự thật**. Nâng ngưỡng không cứu được — phải chặn bằng so khớp chữ số
chính xác. Đúng bài học của cả lab, chỉ là cắn ở tầng khác.

---

> Chi tiết mọi thay đổi so với template BTC (và hai chỗ tôi cố ý **không** đổi):
> [`NOTES.md`](NOTES.md).
