# Probe evaluation — diagnosis and fix

Tóm tắt vấn đề
- Triệu chứng: khi chạy `system/probe_main.py` thu được giá trị `acc_tprime` rất khác nhau giữa các cặp task (ví dụ 44%, 11%, 0%), hoặc kết quả thiếu/không ổn định.

Nguyên nhân gốc rễ
- Trong `system/utilities_probe/evaluation.py` hàm `PredictionBasedEvaluator.eval_all_tasks` đang lặp chỉ trên `data_stream.tasks[:2]` (chỉ 2 task đầu tiên). Kết quả:
  - Khi target task là `0` hoặc `1` thì evaluation trả về accuracy như mong đợi.
  - Khi target task là >1 thì evaluator không tính metric cho task đó, nên probe đôi khi không trả về `task_{t}` hoặc trả kết quả không mong muốn.

Đề xuất sửa lỗi
1. Sửa dòng sau trong `system/utilities_probe/evaluation.py`:

   - Thay `for task in tqdm(data_stream.tasks[:2], desc=f'Evaluating tasks...'):`
   - Bằng  `for task in tqdm(data_stream.tasks, desc=f'Evaluating tasks...'):`

   Điều này đảm bảo evaluator chạy trên tất cả các task trong `data_stream` (không chỉ 2 task đầu).

2. (Tùy chọn) Cách khác hợp lý: thay vì để `PredictionBasedEvaluator` tự chọn tasks, cho `ModelCoach` truyền riêng `tasks_to_run` (hoặc truyền danh sách task muốn evaluate) vào `eval_all_tasks` để chỉ evaluate những task cần thiết.

Các kiểm tra và bước reproduce
- Trước khi sửa: chạy

```bash
python -m system.probe_main --num_tasks 5 --num_clients 1
```

Quan sát xem baseline chỉ được tạo cho task 0,1 (hoặc task trả về) và các cặp khác bị skip.

- Sau khi sửa: chạy lại lệnh trên, kiểm tra file `checkpoint/` và `probe_cache_head/` để thấy `baseline` cho mọi task và `probe_forgetting_results.csv` có các cặp task đầy đủ.

Lưu ý khác có thể gây sai số
- `probe_main.py` hiện đang dùng checkpoint baseline cố định: `ckpt_t_path = ckpt(client_id, t, round_idx=24)`. Hãy đảm bảo round 24 tồn tại cho mọi task, hoặc thay bằng giá trị phù hợp / tham số hóa.
- Cache key hiện là `client{client_id}_t{t}_block{block_name}` — kiểm tra kỹ nếu cùng cache được dùng cho nhiều cấu hình khác nhau (ví dụ khác `classes_per_task`) thì có thể gây kết quả sai.

Patch gợi ý
- Minimal patch: sửa `eval_all_tasks` slice `[:2]` thành không cắt.

Ví dụ sửa trong `system/utilities_probe/evaluation.py`:

```diff
-for task in tqdm(data_stream.tasks[:2], desc=f'Evaluating tasks...'):
+for task in tqdm(data_stream.tasks, desc=f'Evaluating tasks...'):
```

Nếu bạn muốn, tôi có thể áp patch thay đổi này trực tiếp và chạy một kiểm tra nhanh để xác nhận.

Người liên hệ
- File chính: [system/probe_main.py](system/probe_main.py)
- Evaluator: [system/utilities_probe/evaluation.py](system/utilities_probe/evaluation.py)
- Probe trainer: [system/utilities_probe/trainer.py](system/utilities_probe/trainer.py)
- Dataloader: [system/utils/data_utils.py](system/utils/data_utils.py)

Kết luận ngắn gọn
- Nguyên nhân chính: slicing `[:2]` trong `PredictionBasedEvaluator.eval_all_tasks` khiến evaluator chỉ đánh giá 2 task đầu. Sửa thành lặp trên `data_stream.tasks` sẽ khắc phục kết quả bị thiếu/không nhất quán.
