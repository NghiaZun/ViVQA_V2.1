# ViVQA V2.1 — Vietnamese Visual Question Answering

SigLIP + BARTpho seq2seq VQA model for Vietnamese. Generates free-form answers (not classification).

**Current best:** EM ≈ 68%, F1 ≈ 73% on ViVQA val set (~10K samples).

---

## Architecture

```
Image ──► SigLIP-base-patch16-224 ──► [B, 196, 1024]
                                              │
                                     VisionGating (TCVG)
                                              │ α per patch
Question ──► BARTpho-syllable encoder ──► [B, L, 1024]
                │                                │
                └──── FlamingoGatedCrossAttn ────┘
                       (2 layers, text2vision)
                              │
                     BARTpho decoder
                              │
                     Answer tokens (seq2seq)
```

### Key components

| Component | File | Description |
|---|---|---|
| `DeterministicVQA` | `model.py` | Main model — no VAE/KL |
| `FlamingoGatedCrossAttention` | `model.py` | Gated cross-attn, 3 modes |
| `VisionGating` (TCVG) | `model.py` | Type-conditioned per-patch α gate |
| `TypePredictionHead` | `model.py` | Auxiliary type classifier (OBJECT/COUNT/COLOR/LOCATION) |
| `TypeAwareLogitsBias` | `model.py` | Soft vocab biasing per type (optional, risky) |
| `VQAGenDataset` | `dataset.py` | Dataset with P-K sampler, curriculum |
| `detect_question_type` | `dataset.py` | Rule-based Vietnamese type classifier |

### Fusion types (`--fusion_type`)

- `text2vision` — vision patches attend to text (original Flamingo direction). **Default, recommended.**
- `vision2text` — text tokens attend to vision patches.
- `bidirectional` — both directions in parallel (2× grad noise, use lower LR and vision dropout ≤ 0.05).

### VisionGating (TCVG)

Per-patch gating conditioned on question type:
```
query = proj(concat(text_cls, type_emb))      # [B, D]
α = sigmoid(gate_net(concat(v_proj, query)) + vision_bias)  # [B, P]
α = clamp(α, min=min_alpha)                   # floor to prevent over-suppression
gated_vision = α * v_proj + (1-α) * text_pooled
```

**`vision_gate_min_alpha=0.4` is recommended.** Without it, COLOR learns α≈0.18 and COUNT α≈0.33 (over-suppressed). With floor=0.4, both stay in visual range.

---

## Files

```
ViVQA_V2.1/
├── model.py                   # All model classes
├── dataset.py                 # VQAGenDataset, detect_question_type, PKSampler
├── train.py                   # Training loop, all CLI args
├── eval.py                    # Evaluation script (EM/F1/ROUGE, per-type breakdown, CSV export)
├── diagnose_tcvg.py           # Diagnose whether TCVG is learning (weight + runtime analysis)
├── diagnose_vision_shortcut.py# Test blank/noise/real vision → detect text shortcut
├── compute_answer_weights.py  # Generate answer_weights.json for balanced loss
├── create_val_split.py        # Split train.csv → train_split.csv + val_split.csv
├── answer_weights.json        # Pre-computed inverse-frequency weights
├── requirements.txt
└── archive/                   # Old CSV splits
```

---

## Setup

```bash
pip install -r requirements.txt
```

```
torch>=2.6.0
transformers>=4.40.0
peft>=0.10.0
Pillow>=10.0.0
rouge-score>=0.1.2
sentencepiece>=0.1.99
```

---

## Data Format

CSV with columns: `question`, `answer`, `filename` (image filename only, not full path).

```
question,answer,filename
Đây là con gì?,con mèo,img_001.jpg
Có bao nhiêu người?,3,img_002.jpg
```

Images live in a flat folder: `--image_dir /path/to/images/`.

### Prepare val split

```bash
python create_val_split.py \
    --input_csv train.csv \
    --train_out train_split.csv \
    --val_out val_split.csv \
    --val_ratio 0.1 \
    --seed 42
```

### Compute answer weights (for balanced loss)

```bash
python compute_answer_weights.py \
    --train_csv train_split.csv \
    --output answer_weights.json
```

---

## Training

### Recommended command (100 epochs, no early stopping, full features)

```bash
python train.py \
  --train_csv train_split.csv \
  --val_csv val_split.csv \
  --image_dir /path/to/images \
  --vision_model google/siglip-base-patch16-224 \
  --bartpho_model vinai/bartpho-syllable \
  --fusion_type text2vision \
  --num_fusion_layers 2 \
  --use_text_lora --text_lora_r 16 --text_lora_alpha 32 \
  --use_vision_gate --vision_gate_init 1.5 \
  --vision_gate_min_alpha 0.4 \
  --use_type_loss \
  --use_contrastive --contrastive_lambda 0.1 --contrastive_temp 0.07 \
  --answer_weights answer_weights.json \
  --pk_sampling --pk_p 4 --pk_k 8 \
  --epochs 100 \
  --lr 7e-5 \
  --weight_decay 0.01 \
  --dropout 0.1 \
  --label_smoothing 0.1 \
  --scheduler cosine \
  --warmup_epochs 5 \
  --early_stopping_metric f1 \
  --use_scst --scst_start_epoch 15 --scst_lambda 0.05 \
  --sample_every 5 \
  --output_dir checkpoints/ \
  --seed 42
```

### With early stopping (faster iteration)

Add: `--early_stopping --early_stopping_patience 12 --early_stopping_metric f1`

**Do not use `--early_stopping_metric loss` with `--label_smoothing` — smoothed loss is not a reliable proxy for EM/F1.**

### Resume from checkpoint

```bash
python train.py ... --resume checkpoints/last_model.pt
```

Add `--reset_lr` to restart LR schedule from scratch (useful for warm restart after plateau).

### Key training args

| Arg | Default | Notes |
|---|---|---|
| `--lr` | 2e-5 | 7e-5 works well for this setup (flat LR, no differential) |
| `--scheduler` | `plateau` | `cosine` recommended for long runs; `plateau` for short exploratory runs |
| `--warmup_epochs` | 0 | 5 recommended with cosine scheduler |
| `--label_smoothing` | 0.1 | Standard; do NOT monitor loss for early stopping when > 0 |
| `--vision_gate_min_alpha` | 0.0 | **Set to 0.4** to prevent COLOR/COUNT vision over-suppression |
| `--use_scst` | off | SCST directly optimizes F1 reward; start after CE warmup (epoch 15+) |
| `--scst_lambda` | 0.1 | 0.05 for stability; 0.1 for stronger EM push |
| `--use_contrastive` | off | InfoNCE on fused_vision ↔ text_cls; bridges SigLIP English / BARTpho Vietnamese gap |
| `--pk_sampling` | off | P-K sampler groups by answer type for contrastive; use with `--use_contrastive` |
| `--gradient_accumulation_steps` | 1 | Increase for effective larger batch when GPU memory is tight |

---

## Evaluation

```bash
python eval.py \
  --checkpoint checkpoints/best_model.pt \
  --csv_path val_split.csv \
  --image_folder /path/to/images \
  --output_csv result.csv \
  --num_beams 3 \
  --repetition_penalty 1.3
```

Outputs:
- EM, F1, ROUGE-1, ROUGE-L overall
- Per-type breakdown (OBJECT / COUNT / COLOR / LOCATION)
- Type prediction accuracy (if model has `--use_type_loss`)
- `result.csv` with per-sample predictions and error flags

### Metrics

- **EM (Exact Match)**: after NFC normalization + lowercase + strip. Binary per sample.
- **F1**: token-overlap F1 after same normalization. Continuous — better for early stopping.
- **ROUGE-1/L**: token-level unigram/LCS overlap.

**Early stopping:** monitor **F1** (smoother than EM which is a step function; EM can plateau for 5-10 epochs while F1 still improves).

---

## Diagnostics

### TCVG diagnostic (is VisionGating learning?)

```bash
# Static weight analysis only (fast)
python diagnose_tcvg.py \
    --checkpoint checkpoints/best_model.pt \
    --csv_path val_split.csv \
    --image_folder /path/to/images \
    --weights_only

# Full runtime analysis (per-type alpha distribution)
python diagnose_tcvg.py \
    --checkpoint checkpoints/best_model.pt \
    --csv_path val_split.csv \
    --image_folder /path/to/images
```

Checks:
1. `vision_bias` — σ(bias) > 0.95 = gate always open (not selective)
2. Type embedding cosine similarity — > 0.7 = types not differentiated
3. Per-type α distribution — spread < 0.05 = TCVG is a no-op

**Healthy values:** avg cosine sim ≈ -0.02, α spread ≈ 0.8, COLOR α ≥ 0.4, COUNT α ≥ 0.4.

### Vision shortcut diagnostic (is model actually using images?)

```bash
python diagnose_vision_shortcut.py \
    --checkpoint checkpoints/best_model.pt \
    --val_csv val_split.csv \
    --image_dir /path/to/images \
    --num_samples 200
```

Compares EM with blank vision / noise vision / real vision. If EM(blank) ≈ EM(real) → text shortcut.

---

## Known bottlenecks

| Type | EM | Root cause | Fix |
|---|---|---|---|
| OBJECT | ~80% | Synonym variability | Post-processing synonym map |
| COLOR | ~65% | SigLIP-base 14×14 patch resolution | Larger vision encoder (384px) |
| COUNT | ~50% | Off-by-1 bias at 196 patches | SigLIP-large or counting augmentation |
| LOCATION | ~25% | 2-layer fusion insufficient for spatial reasoning | More fusion layers or explicit spatial features |

**Architecture ceiling with SigLIP-base-224 + 2 Flamingo layers: EM ~72–75%.** Pushing beyond requires:
- SigLIP-large or SigLIP-SO400M at 384px (better COUNT/COLOR)
- More fusion layers (4+) or bidirectional fusion (LOCATION)

---

## Checkpoints

After training, `output_dir/` contains:
- `best_model.pt` — best by `--early_stopping_metric` (default: F1)
- `last_model.pt` — latest epoch (used for `--resume`)
- `training_curves.png` — loss/LR/EM/F1 plots
- `metrics.csv` — per-epoch metrics table

Checkpoint keys: `epoch`, `model_state_dict`, `optimizer_state_dict`, `best_monitor`, `args`, `training_history`.

---

## BARTpho decoder notes

- `decoder_start_token_id == eos_token_id` (BART architecture property) — BOS is not in `all_special_ids`, so `tokenizer.decode(skip_special_tokens=True)` does NOT remove it. Both `eval.py` and `train.py` use `_decode_gt()` which manually filters `bos_token_id`.
- `repetition_penalty` excludes BOS/EOS/PAD from penalty to prevent EOS suppression (which would cause infinite generation).
- `max_length=10` for generation (Vietnamese answers are short: 1–4 tokens typical).
