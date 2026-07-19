# ViGCT-VQA — Vietnamese Visual Question Answering

Mã nguồn mô hình **ViGCT-VQA** cho bài toán Visual Question Answering tiếng Việt trên bộ dữ
liệu **ViVQA**. Mô hình kết hợp bộ mã hoá thị giác **SigLIP** với mô hình ngôn ngữ **BARTpho**
qua cơ chế gated cross-attention, sinh câu trả lời theo kiểu seq2seq.

Repo này là phần nộp **mã nguồn** cho luận văn; **không** đính kèm dataset, ảnh hay checkpoint.

---

## Cài đặt

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Mô hình nền tải tự động từ Hugging Face khi chạy lần đầu:
`google/siglip-base-patch16-224` và `vinai/bartpho-syllable`.

---

## Chuẩn bị dữ liệu

Bộ dữ liệu **ViVQA** là công khai — tải từ nguồn gốc rồi tạo split cục bộ. CSV cần các cột
`question`, `answer`, `filename` (tên file ảnh); ảnh để phẳng trong một thư mục.

```
question,answer,filename
Đây là con gì?,con mèo,img_001.jpg
Có bao nhiêu người?,3,img_002.jpg
```

Tạo tập validation và answer weights:

```bash
python create_val_split.py \
    --input_csv train.csv --train_out train_split.csv --val_out val_split.csv \
    --val_ratio 0.1 --seed 42

python compute_answer_weights.py \
    --train_csv train_split.csv --output answer_weights.json
```

---

## Huấn luyện

```bash
python train.py \
  --train_csv train_split.csv \
  --val_csv val_split.csv \
  --image_dir /path/to/images \
  --vision_model google/siglip-base-patch16-224 \
  --bartpho_model vinai/bartpho-syllable \
  --fusion_type text2vision --num_fusion_layers 2 \
  --use_text_lora --text_lora_r 16 --text_lora_alpha 32 \
  --use_vision_gate --vision_gate_init 1.0 --vision_gate_min_alpha 0.0 \
  --use_type_loss --type_loss_weight 0.2 \
  --use_contrastive --contrastive_lambda 0.05 \
  --use_siglip_pooler \
  --use_cdw_ce --cdw_lambda 0.1 \
  --answer_weights answer_weights.json \
  --pk_sampling --pk_p 4 --pk_k 3 \
  --epochs 40 --lr 7e-5 --weight_decay 0.01 --label_smoothing 0.1 \
  --scheduler cosine --warmup_epochs 3 \
  --early_stopping --early_stopping_patience 15 --early_stopping_metric em \
  --batch_size 12 --vision_dropout_rate 0.10 --num_workers 4 --seed 42 \
  --output_dir checkpoints_run/
```

Resume từ checkpoint: `python train.py ... --resume checkpoints_run/last_model.pt`.
Xem toàn bộ tham số: `python train.py --help`.

---

## Đánh giá

```bash
python eval.py \
  --checkpoint checkpoints_run/best_model.pt \
  --csv_path test.csv \
  --image_folder /path/to/images \
  --output_csv result.csv \
  --num_beams 3 --repetition_penalty 1.3 --max_length 10 \
  --use_synonyms --use_constrained \
  --train_csv_for_trie train_split.csv
```

In ra EM / F1 / ROUGE và ghi dự đoán từng mẫu vào `result.csv`.

---

## Cấu trúc mã nguồn

```
model.py                     Các lớp mô hình
dataset.py                   Dataset, phân loại câu hỏi (rule-based), P-K sampler
unsupervised_type.py         [WIP] Khám phá loại câu hỏi không giám sát
train.py                     Huấn luyện + toàn bộ CLI args
eval.py / eval_v4.py         Đánh giá
evaluator.py                 Wrapper gọi eval
main.py                      Vòng lặp fine-tune tự động (xem PIPELINE.md)
generation.py                Sinh ảnh FLUX.1-dev + kiểm định bằng Qwen2-VL/YOLO
pipeline_utils.py            Tiện ích cho pipeline
config.py                    Cấu hình pipeline
type_conditioned_adapter.py  Adapter theo loại (thử nghiệm)
model_soup.py                Trộn trọng số checkpoint
probe.py                     Linear probe chẩn đoán
diagnose_*.py                Script chẩn đoán TCVG / vision shortcut
compute_answer_weights.py    Sinh answer_weights.json
compute_vivqax_weights.py    Answer weights cho ViVQA-X
create_val_split.py          Tách train/val
preprocess_vivqax.py         Tiền xử lý ViVQA-X
```

---

## Đang phát triển

- **`unsupervised_type.py`** — thay bộ luật gán loại câu hỏi bằng gom cụm embedding
  (BARTpho + KMeans), sinh cột `question_type` cho train loop. Chưa nối mặc định vào `train.py`.
- **Chưng cất tri thức qua sinh ảnh** — mở rộng vòng lặp fine-tune: sinh ảnh mới (FLUX.1-dev),
  dùng teacher đa phương thức tạo/kiểm định nhãn, rồi bơm mẫu tổng hợp trở lại tập train.
  Khung nền trong `PIPELINE.md`, phần chưng cất đang được bổ sung.

---

## Trích dẫn

```bibtex
@inproceedings{duong2026vigctvqa,
  title     = {{ViGCT-VQA}: Gated Cross-Modal Attention with Type-Conditioned
               Visual Gating for Vietnamese {VQA}},
  author    = {Duong, Nghia Trung and Luong, Thanh Xuan and Le, Tung},
  booktitle = {Proceedings of the 30th International Conference on
               Knowledge-Based and Intelligent Information \& Engineering
               Systems ({KES} 2026)},
  year      = {2026},
  publisher = {Elsevier}
}
```
