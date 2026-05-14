# ViVQA Fine-tune Loop v2 — Pipeline

Tự động lặp: eval → phân tích lỗi → sinh ảnh augmented → merge dataset → fine-tune.  
Mỗi loop nhắm vào những câu hỏi/answer mà model đang sai nhiều nhất.

---

## Tổng quan

```
               ┌──────────────────────────────────────────┐
               │               main.py                    │
               │                                          │
               │  Loop 0: Baseline training               │
               │       ↓                                  │
               │  Loop 1…N:                               │
               │    A: Eval checkpoint                    │
               │    B: Stop check                         │
               │    C: Generation plan                    │
               │    D: Generate images (FLUX + verify)    │
               │    E: Merge dataset                      │
               │    F: Fine-tune (resume best)            │
               └──────────────────────────────────────────┘
```

---

## Files

| File | Vai trò |
|---|---|
| `main.py` | Orchestrator — chạy toàn bộ fine-tune loop |
| `config.py` | Config trung tâm — mọi hyperparameter và path |
| `train.py` | Script training một lần (được gọi bởi main.py qua subprocess) |
| `eval.py` | Script eval một lần — xuất EM/F1/ROUGE + result.csv |
| `evaluator.py` | Wrapper gọi eval.py, parse output, kiểm tra stop condition |
| `generation.py` | Sinh ảnh FLUX.1-dev + verify Qwen2-VL/YOLO, budget planner |
| `pipeline_utils.py` | Merge dataset, rebuild image dir, recompute answer weights |
| `dataset.py` | VQAGenDataset, PKSampler, detect_question_type |
| `model.py` | DeterministicVQA (SigLIP + BARTpho + Flamingo + TCVG) |

---

## Các bước trong mỗi loop

### Loop 0 — Baseline

```
recompute_answer_weights(train_csv)
rebuild_combined_image_dir()          # symlink ảnh gốc
run_train(..., epochs=initial_epochs) # 40 epochs mặc định
```

Không có augmentation. Tạo ra checkpoint baseline để các loop sau fine-tune từ đó.

---

### Bước A — Eval

```python
overall, per_type = run_eval(current_ckpt, result_csv)
# overall  = {'F1': float, 'EM': float}   (thang 0-100)
# per_type = {'COUNT': {'F1': ..., 'EM': ...}, ...}
```

`evaluator.run_eval()` gọi `eval.py` qua subprocess và parse stdout bằng regex.  
Output: `result_loop{N}.csv` — mỗi dòng có `ground_truth`, `prediction`, `exact_match`, `question_type`.

---

### Bước B — Stop check

`evaluator.should_stop()` dừng loop khi **cả ba** điều kiện đúng:

| Điều kiện | Config key | Ý nghĩa |
|---|---|---|
| EM variance < ngưỡng | `stop_em_variance=8.0` | Các type đã hội tụ |
| EM tăng >= delta | `stop_em_delta=0.3` | Vẫn còn cải thiện |
| F1 − EM < max gap | `stop_gap_max=10.0` | Model predict chính xác (không chỉ đúng từng phần) |

---

### Bước C — Generation plan

`generation.compute_generation_plan(result_csv, train_csv, loop_idx)` phân tích `result.csv` để tìm answer hay sai nhất.

**Thuật toán:**

```
với mỗi type (COUNT/LOCATION/OBJECT/COLOR):
  1. Tính EM type → nếu >= gen_skip_em (72%) → bỏ qua
  2. Nhóm wrong samples theo ground_truth
  3. Chỉ target answers có error_rate >= gen_min_error_rate (0.3)
  4. type_budget ∝ tổng số lần sai × focus_weight × gap_scale
  5. Phân budget xuống từng answer ∝ error_count của answer đó
```

**Dynamic focus weight** (nếu `gen_focus_dynamic=True`):
```
weight = focus_max - (em_type - em_min) / em_range × (focus_max - focus_min)
```
Type EM thấp nhất → weight cao nhất (1.4), type EM cao nhất → 0.7.

**Budget decay theo loop:** `budget × loop_budget_decay^(loop-1)` — loop sau gen ít hơn (noise control).

---

### Bước D — Sinh ảnh (FLUX.1-dev)

`generation.generate_images_for_type()` sinh ảnh cho từng type với prompt được xây theo template:

| Type | Template chính |
|---|---|
| COUNT | `exactly {N} {object} {scene}, each clearly countable` |
| COLOR | `a {color_en} {object} {scene}, color clearly distinguishable` |
| LOCATION | `a {subject} {prep_en} a {anchor} {scene}, spatial positioning visible` |
| OBJECT | Generic COCO-style scene với object đó |

**FLUX img2img:** nếu `flux_use_img2img=True`, lấy ảnh tham chiếu từ train set (cùng answer) để FLUX giữ cấu trúc kiểu COCO.  
`flux_img2img_strength=0.35` — thấp = giữ cấu trúc ảnh gốc nhiều, cao = tự do hơn.

**Verification sau khi sinh:**
- **Qwen2-VL-2B**: trả lời câu hỏi gốc từ ảnh sinh → so khớp với target answer
- **YOLOv8m**: kiểm tra COUNT — đếm object bbox → EM với count đúng
- Ảnh nào fail verify → loại bỏ

**VRAM tự động:**
```
>= 60 GB : batch=32, no offload, torch.compile
>= 35 GB : batch=16, no offload, torch.compile
< 35 GB  : batch=4,  cpu_offload=True
```

---

### Bước E — Merge dataset

```python
loop_train_csv = merge_dataset(base_train_csv, new_rows, loop_idx)
```

- Nếu `loop_accumulate_aug=True`: base = train CSV đã aug từ loop trước (tích lũy)
- Nếu `loop_accumulate_aug=False` (mặc định): base = train CSV gốc (chỉ aug loop này)
- Image dir mới được tạo bằng symlink ảnh gốc + copy ảnh aug loop này
- `recompute_answer_weights()` chạy lại để cân bằng class

---

### Bước F — Fine-tune

```python
run_train(
    train_csv=loop_train_csv,
    image_dir=loop_combined,
    epochs=finetune_epochs,      # 30
    lr=finetune_lr,              # 2e-5
    warmup_epochs=finetune_warmup_epochs,  # 2
    resume=best_ckpt,            # resume từ best (nếu loop_resume_from_best=True)
    early_stopping=True,
    early_stopping_patience=finetune_es_patience,  # 8
)
```

Mặc định resume từ `best_ckpt` (checkpoint tốt nhất xuyên tất cả loop theo EM tổng thể), không phải checkpoint loop trước — giúp tránh drift khi loop trước bị degraded.

---

## Config chính (config.py)

### Loop behavior

| Key | Default | Ý nghĩa |
|---|---|---|
| `max_loops` | 5 | Số loop tối đa |
| `initial_epochs` | 40 | Epochs cho loop 0 (baseline) |
| `finetune_epochs` | 30 | Epochs mỗi loop fine-tune |
| `finetune_lr` | 2e-5 | LR cho fine-tune (thấp hơn baseline 7e-5) |
| `finetune_warmup_epochs` | 2 | Warmup ngắn cho fine-tune |
| `finetune_es_patience` | 8 | Early stopping patience |
| `loop_accumulate_aug` | True | Giữ aug của loop trước hay không |
| `loop_resume_from_best` | False | Resume từ best ckpt hay ckpt loop trước |
| `loop_budget_decay` | 0.8 | Budget giảm 20% mỗi loop |

### Generation budget

| Key | Default | Ý nghĩa |
|---|---|---|
| `gen_budget_total` | 800 | Tổng budget tham chiếu |
| `gen_budget_min` | 80 | Budget tối thiểu mỗi type |
| `gen_budget_max` | 400 | Budget tối đa mỗi type |
| `gen_skip_em` | 72.0 | Bỏ qua type nếu EM >= ngưỡng này |
| `gen_min_error_rate` | 0.3 | Chỉ gen answer sai >= 30% |
| `gen_max_type_share` | 0.25 | Type không quá 25% tổng dataset sau aug |
| `gen_gap_budget_scale` | 1.0 | Scale budget theo khoảng cách EM → skip threshold |
| `gen_focus_dynamic` | True | Tự điều chỉnh weight theo EM từng type |
| `gen_focus_weights` | COUNT:1.2, LOC:1.1, OBJ:0.9, CLR:0.9 | Static weight trước dynamic adjust |
| `loop_min_error_rate_delta` | 0.1 | Siết ngưỡng lỗi thêm 10% mỗi loop |

### Stop condition

| Key | Default | Ý nghĩa |
|---|---|---|
| `stop_em_variance` | 8.0 | Max(EM) - Min(EM) qua các type |
| `stop_em_delta` | 0.3 | EM tổng phải tăng >= giá trị này |
| `stop_gap_max` | 10.0 | F1 - EM < ngưỡng này cho mọi type |

---

## Chạy pipeline

### Cách đơn giản nhất

```bash
python main.py
```

Config đọc từ `config.py`. Chỉnh sửa `CFG` trong `config.py` trước khi chạy.

### Đầu ra

```
checkpoints_pk/
├── loop0/
│   ├── best_model.pt
│   ├── last_model.pt
│   ├── training_curves.png
│   └── metrics.csv
├── loop1/
│   └── ...
└── loop{N}/
    └── ...

aug_images/
└── result_{loop}/        # ảnh sinh bởi FLUX mỗi loop

result_loop{N}.csv        # eval output mỗi loop
train_loop{N}.csv         # dataset sau merge mỗi loop
pipeline_history.json     # F1/EM toàn bộ loop
finetune_loop_v2.png      # plot tổng thể
result_final.csv          # eval final trên best checkpoint
```

### Plots

`finetune_loop_v2.png` có 4 panel:
1. Overall F1 vs EM theo loop
2. Per-type F1 theo loop
3. Per-type EM theo loop (stop condition reference)
4. F1 − EM gap theo loop (generation style reference)

---

## Luồng dữ liệu

```
archive/train_split.csv        ← dataset gốc
        │
        ▼
   Loop 0 training             → checkpoints_pk/loop0/best_model.pt
        │
        ▼
   Eval on val_split.csv       → result_loop0.csv
        │
        ▼
   compute_generation_plan()   ← đọc result_loop0.csv + train_split.csv
        │
        ▼
   generate_images_for_type()  → aug_images/result_1/*.jpg
        │
        ▼
   merge_dataset()             → train_loop1.csv
        │
        ▼
   Loop 1 fine-tune            → checkpoints_pk/loop1/best_model.pt
        │
       ...
```

---

## Monitoring

Pipeline in log ra stdout mỗi bước. Theo dõi:

```
[1A] Evaluating...
  Overall  F1=73.21%  EM=68.45%
  Type         F1      EM     Gap
  COUNT       50.12   45.30    4.82
  LOCATION    31.44   25.11    6.33
  OBJECT      80.55   78.90    1.65
  COLOR       67.34   62.18    5.16

  Stop check:
    EM variance  = 53.79  (cần < 8.0) → ❌
    EM delta     = +3.21  (cần >= 0.3) → ✅
    Max EM-F1 gap= 6.33   (cần < 10.0) → ✅

[1C] Computing generation plan...
  Type         EM    Budget  Note
  COUNT        45.3      80  3 answers cần gen | 120 lần sai | min_err=0.30
  LOCATION     25.1     400  5 answers cần gen | 280 lần sai | min_err=0.30
  OBJECT       78.9       0  EM=78.9% >= 72.0 → skip
  COLOR        62.2     144  4 answers cần gen | 95 lần sai | min_err=0.30
```

---

## Troubleshooting

| Triệu chứng | Nguyên nhân | Fix |
|---|---|---|
| `per_type rỗng` | eval.py crash hoặc output format thay đổi | Chạy thử `python eval.py` riêng, xem stderr |
| Budget = 0 cho mọi type | Tất cả EM >= `gen_skip_em` | Giảm `gen_skip_em` trong config |
| Checkpoint không tồn tại sau loop 0 | train.py crash sớm | Chạy thử `python train.py ...` riêng |
| FLUX OOM | VRAM không đủ | Giảm `flux_batch_size`, bật `flux_cpu_offload=True` |
| EM giảm sau loop fine-tune | Augmented data có noise | Tăng `gen_min_error_rate`, giảm `flux_img2img_strength` |
| Loop không dừng | stop condition quá chặt | Giảm `stop_em_variance` hoặc tăng `stop_gap_max` |
