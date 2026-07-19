"""
TRAINING SCRIPT FOR DETERMINISTIC VQA (No Latent Reasoning)
============================================================

Simplified training without VAE/KL complexity:
- Stage 1: SKIP (no latent to train)
- Stage 2: SKIP (no KL warmup needed)
- Stage 3: Direct end-to-end training

Focus: Maximize accuracy and training stability for Vietnamese VQA.

Version: 2.0 with improvements:
- LR scheduler (ReduceLROnPlateau)
- Early stopping
- Better metrics (EM + F1 score)
- Label smoothing
- Proper beam search generation
"""

import os
# Must be set before CUDA initializes — force override any existing value
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
# PYTHONHASHSEED inside Python only affects child processes, not the current
# interpreter (it's read at startup). Set it as a shell var before launching:
#   PYTHONHASHSEED=42 python train.py ...
# We still set it here so subprocesses (e.g. DataLoader workers) inherit it.
os.environ["PYTHONHASHSEED"] = "42"

import json
import argparse
import random
import unicodedata
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from collections import Counter
import csv
import matplotlib.pyplot as plt
from dataset import VQAGenDataset

from model import DeterministicVQA

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    print("⚠️  Warning: rouge_score not installed. ROUGE metrics will be skipped.")
    print("   Install with: pip install rouge-score")

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# ============================================================================
# VISION DROPOUT AUGMENTATION (Inline approach - simpler!)
# ============================================================================

# NOTE: We apply vision dropout by directly zeroing pixel_values before forward pass
# This is simpler than monkey-patching and avoids nn.Module assignment issues.
#
# Type-specific dropout rates (based on diagnostic results):
#   - COUNT (42.6% acc) → 0.4 (aggressive)
#   - COLOR (49.9% acc) → 0.35 (high)
#   - OBJECT (61.5% acc) → 0.2 (low)
#   - LOCATION (65% acc) → 0.2 (low)
#
# See implementation in run_one_epoch_deterministic() around line 410


# ============================================================================
# UTILITIES
# ============================================================================

def _normalize_vn(text: str) -> str:
    """
    Chuẩn hóa text tiếng Việt trước khi so sánh.
    NFC: đảm bảo cùng byte representation cho ký tự tổ hợp (ệ, ổ, ẫ, ...)
    Tránh false negative khi tokenizer decode ra NFC nhưng CSV lưu NFD.
    """
    return unicodedata.normalize('NFC', text).strip().lower()


def compute_exact_match(prediction: str, ground_truth: str) -> float:
    return 1.0 if _normalize_vn(prediction) == _normalize_vn(ground_truth) else 0.0


def _decode_gt(tokenizer, label_token_ids: list) -> str:
    """Decode ground-truth label ids with the same BOS filtering as model._decode_seq.
    Prevents false EM mismatches when BOS token is not in tokenizer.all_special_ids."""
    ids = [t for t in label_token_ids if t != tokenizer.bos_token_id]
    return tokenizer.decode(ids, skip_special_tokens=True).strip()


class EarlyStopping:
    """Early stopping to prevent overfitting — supports both min (loss) and max (EM/F1) modes"""
    def __init__(self, patience=5, min_delta=0.001, verbose=True, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.mode = mode  # 'min' for loss, 'max' for EM/F1
        self.counter = 0
        self.best_score = float('-inf') if mode == 'max' else float('inf')
        self.best_loss = float('inf')   # kept for checkpoint compat
        self.early_stop = False

    def __call__(self, score):
        improved = (
            score > self.best_score + self.min_delta if self.mode == 'max'
            else score < self.best_score - self.min_delta
        )
        if improved:
            if self.verbose:
                print(f"  📉 {'↑' if self.mode == 'max' else '↓'} Metric improved: {self.best_score:.4f} → {score:.4f}")
            self.best_score = score
            self.best_loss = score   # mirror for compat
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.verbose:
                print(f"  ⚠️  No improvement for {self.counter}/{self.patience} epochs (best={self.best_score:.4f})")
            if self.counter >= self.patience:
                if self.verbose:
                    print(f"  🛑 Early stopping triggered!")
                self.early_stop = True
                return True
        return False


def compute_f1_score(prediction: str, ground_truth: str) -> float:
    """
    Compute F1 score between prediction and ground truth
    
    F1 is better than exact match for VQA because it gives partial credit!
    """
    pred_tokens = _normalize_vn(prediction).split()
    gt_tokens   = _normalize_vn(ground_truth).split()
    
    # Edge case: both empty (should be 1.0, not 0.0)
    # If both model and ground truth produce nothing, it's technically correct
    if len(pred_tokens) == 0 and len(gt_tokens) == 0:
        return 1.0
    
    # Edge case: one empty, one not
    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return 0.0
    
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        return 0.0
    
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    
    return f1


def compute_rouge_scores(prediction: str, ground_truth: str) -> dict:
    """
    Compute ROUGE-1 and ROUGE-L scores
    
    ROUGE-1: Unigram overlap (measures word-level similarity)
    ROUGE-L: Longest common subsequence (measures fluency/order)
    
    Returns:
        dict with 'rouge1' and 'rougeL' F1 scores (0-1 range)
    """
    if not ROUGE_AVAILABLE:
        return {'rouge1': 0.0, 'rougeL': 0.0}
    
    # use_stemmer=False because Vietnamese is an isolating language
    # (tiếng Việt là ngôn ngữ đơn lập - words don't change form like English)
    # Stemming is designed for inflectional languages (English: running→run)
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=False)
    scores = scorer.score(ground_truth, prediction)
    
    return {
        'rouge1': scores['rouge1'].fmeasure,
        'rougeL': scores['rougeL'].fmeasure
    }


class CurriculumSampler(Sampler):
    """
    Curriculum learning sampler: sorts training samples from easy (frequent answers)
    to hard (rare answers), then uses a pacing function to include increasingly
    harder samples as training progresses.

    Call set_epoch(epoch, total_epochs) before each training epoch.
    """

    def __init__(self, dataset, seed=42, start_ratio=0.4):
        answer_freq = Counter(dataset.data['answer'].tolist())
        # Sort descending by frequency (high freq = easy = first)
        self.indices_by_difficulty = sorted(
            range(len(dataset)),
            key=lambda i: -answer_freq.get(dataset.data.iloc[i]['answer'], 0)
        )
        self.seed = seed
        self.start_ratio = start_ratio
        self._epoch = 1
        self._total = 1

    def set_epoch(self, epoch: int, total_epochs: int):
        self._epoch = epoch
        self._total = total_epochs

    def _active_size(self) -> int:
        ratio = self.start_ratio + (1.0 - self.start_ratio) * (self._epoch - 1) / max(1, self._total - 1)
        return max(1, int(len(self.indices_by_difficulty) * min(ratio, 1.0)))

    def __iter__(self):
        n = self._active_size()
        subset = list(self.indices_by_difficulty[:n])
        rng = random.Random(self.seed + self._epoch)
        rng.shuffle(subset)
        return iter(subset)

    def __len__(self):
        return self._active_size()


def analyze_dataset(dataset, tokenizer, num_samples=1000):
    """Analyze dataset to detect imbalance"""
    print("\n[Dataset Analysis]")
    
    # Handle Subset (from random_split)
    from torch.utils.data import Subset
    actual_dataset = dataset.dataset if isinstance(dataset, Subset) else dataset
    indices = dataset.indices if isinstance(dataset, Subset) else range(len(dataset))
    
    answers = []
    question_lengths = []
    answer_lengths = []
    
    sample_indices = list(indices)[:min(num_samples, len(indices))]
    
    for idx in sample_indices:
        item = actual_dataset[idx]
        
        # Handle both dict and tuple returns
        if isinstance(item, dict):
            labels = item['labels']
            input_ids = item['input_ids']
        else:
            # tuple: (pixel_values, input_ids, attention_mask, labels)
            _, input_ids, _, labels = item
        
        label_tokens = labels[labels != -100]
        answer = tokenizer.decode(label_tokens, skip_special_tokens=True)
        answers.append(answer)
        
        question = tokenizer.decode(input_ids, skip_special_tokens=True)
        question_lengths.append(len(question.split()))
        answer_lengths.append(len(answer.split()))
    
    answer_counts = Counter(answers)
    
    print(f"  Unique answers: {len(answer_counts)}")
    print(f"  Top 10 most common answers:")
    for ans, count in answer_counts.most_common(10):
        pct = count / len(answers) * 100
        print(f"    '{ans}': {count} ({pct:.1f}%)")
    
    # Check imbalance
    if answer_counts.most_common(1)[0][1] / len(answers) > 0.3:
        print(f"  ⚠️  Dataset appears imbalanced! Top answer accounts for {answer_counts.most_common(1)[0][1] / len(answers) * 100:.1f}%")
    
    print(f"  Avg question length: {sum(question_lengths)/len(question_lengths):.1f} tokens")
    print(f"  Avg answer length: {sum(answer_lengths)/len(answer_lengths):.1f} tokens")


def plot_training_curves(history, output_dir):
    """
    Plot and save training curves
    
    Args:
        history: List of dicts with metrics per epoch
        output_dir: Directory to save plots
    """
    if not history:
        return
    
    epochs = [h['epoch'] for h in history]
    train_losses = [h['train_loss'] for h in history]
    val_losses = [h['val_loss'] for h in history]
    learning_rates = [h['learning_rate'] for h in history]
    
    # Extract metrics if available
    exact_matches = [h.get('exact_match', None) for h in history]
    f1_scores = [h.get('f1_score', None) for h in history]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Metrics', fontsize=16, fontweight='bold')
    
    # 1. Loss curves
    axes[0, 0].plot(epochs, train_losses, 'b-o', label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs, val_losses, 'r-o', label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training & Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Learning rate
    axes[0, 1].plot(epochs, learning_rates, 'g-o', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Learning Rate')
    axes[0, 1].set_title('Learning Rate Schedule')
    axes[0, 1].set_yscale('log')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Exact Match (if available)
    if any(em is not None for em in exact_matches):
        valid_epochs = [e for e, em in zip(epochs, exact_matches) if em is not None]
        valid_ems = [em for em in exact_matches if em is not None]
        axes[1, 0].plot(valid_epochs, valid_ems, 'm-o', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Exact Match (%)')
        axes[1, 0].set_title('Exact Match Score')
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, 'No EM data', ha='center', va='center')
        axes[1, 0].set_title('Exact Match Score')
    
    # 4. F1 Score (if available)
    if any(f1 is not None for f1 in f1_scores):
        valid_epochs = [e for e, f1 in zip(epochs, f1_scores) if f1 is not None]
        valid_f1s = [f1 for f1 in f1_scores if f1 is not None]
        axes[1, 1].plot(valid_epochs, valid_f1s, 'c-o', linewidth=2)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('F1 Score (%)')
        axes[1, 1].set_title('F1 Score')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'No F1 data', ha='center', va='center')
        axes[1, 1].set_title('F1 Score')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"  📊 Saved training curves to: {plot_path}")
    plt.close()


def save_metrics_csv(history, output_dir):
    """
    Save training metrics to CSV
    
    Args:
        history: List of dicts with metrics per epoch
        output_dir: Directory to save CSV
    """
    if not history:
        return
    
    csv_path = os.path.join(output_dir, 'training_metrics.csv')
    
    # Get all possible keys
    all_keys = set()
    for h in history:
        all_keys.update(h.keys())
    all_keys = sorted(all_keys)
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(history)
    
    print(f"  📊 Saved metrics CSV to: {csv_path}")



# ============================================================================
# TRAINING UTILITIES
# ============================================================================

def run_one_epoch_deterministic(
    model, dataloader, optimizer, scaler, device,
    is_training=True, max_norm=1.0, stage=3, gradient_accumulation_steps=1,
    answer_weights=None, use_type_loss=False, vision_dropout_rate=0.10,
    text_dropout_rate=0.0,
    use_scst=False, scst_start_epoch=5, scst_lambda=0.1,
    scst_sample_temp=1.0, current_epoch=0, amp_dtype=torch.float16,
    log_file=None,
    type_sample_weight_cfg=None,  # dict with type_defaults + count_overrides for sample-level weighting
    use_rdrop=False, rdrop_alpha=0.1,  # 🔥 R-Drop: KL consistency between two dropout sub-models
    use_cdw_ce=False, cdw_lambda=0.1, cdw_ordinal_weights=None,  # 🔥 CDW-CE: ordinal penalty for COUNT
    ema_model=None, ema_decay=0.999,  # 🔥 EMA: exponential moving average of weights
):
    """
    Run one epoch for deterministic model (no KL diagnostics needed!)
    
    Args:
        gradient_accumulation_steps: Accumulate gradients over multiple batches
                                     for effective larger batch size
        answer_weights: Tensor of token-level weights for balanced loss
        use_type_loss: Whether to apply type-conditional loss weighting
        vision_dropout_rate: Probability of zeroing pixel_values (modality dropout).
            - text2vision:    0.10 recommended (only vision path affected)
            - bidirectional:  0.05 or 0.0 (2x gradient noise from blank vision)
            - vision2text:    0.10 OK
    
    Returns:
        dict with metrics: loss, answer_loss, type_loss
    """
    model.train() if is_training else model.eval()
    _amp_on = is_training and amp_dtype in (torch.float16, torch.bfloat16)

    total_loss = 0.0
    total_answer_loss = 0.0
    total_type_loss = 0.0  # 🔥 NEW: Track type loss
    nan_loss_steps = 0
    nan_grad_steps = 0
    num_batches = 0
    steps_since_update = 0

    if is_training and optimizer is not None:
        optimizer.zero_grad()
    
    with torch.set_grad_enabled(is_training):
        pbar = tqdm(dataloader, desc=f"{'Train' if is_training else 'Val'} Stage {stage}")
        
        for batch_idx, batch in enumerate(pbar):
            pixel_values = batch['pixel_values'].to(device)
            # 🚀 SPEED OPTIMIZATION: Convert to channels_last for faster conv ops
            pixel_values = pixel_values.to(memory_format=torch.channels_last)
            
            # 🔥 VISION DROPOUT (modality dropout — chỉ lúc training)
            # Rate được truyền vào từ caller theo fusion_type:
            #   text2vision:   0.10 — chỉ vision path bị ảnh hưởng, safe
            #   bidirectional: 0.05 — 2x gradient paths, giảm noise
            #   vision2text:   0.10 — text path là primary, vision dropout OK
            # Save original pixel_values before dropout — SCST must use real images
            pixel_values_orig = pixel_values
            if is_training and vision_dropout_rate > 0:
                drop_mask = torch.rand(pixel_values.size(0), device=pixel_values.device) < vision_dropout_rate
                if drop_mask.any():
                    pixel_values = pixel_values * (~drop_mask).float().view(-1, 1, 1, 1)
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # TEXT DROPOUT (Phase 1: image-flow training)
            # Replace question with padding → model buộc phải dùng image để trả lời.
            # Dùng rate cao (0.85) ở pha 1, 0.0 ở pha 2 (bình thường).
            if is_training and text_dropout_rate > 0 and random.random() < text_dropout_rate:
                pad_id = model.config.pad_token_id
                bos_id = model.config.decoder_start_token_id
                input_ids = torch.full_like(input_ids, pad_id)
                input_ids[:, 0] = bos_id  # giữ BOS để encoder không thấy chuỗi hoàn toàn trống
                attention_mask = torch.zeros_like(attention_mask)
                attention_mask[:, 0] = 1

            # 🔥 Extract question types if using type-conditional loss
            question_types = None
            if use_type_loss and 'question_type' in batch:
                question_types = batch['question_type'].to(device)

            # 🔥🔥🔥 Extract teacher inputs for distillation
            images_384 = None
            raw_questions = None
            if 'images_384' in batch:
                images_384 = batch['images_384'].to(device)
                # 🚀 Convert teacher images to channels_last too
                images_384 = images_384.to(memory_format=torch.channels_last)
            if 'raw_question' in batch:
                raw_questions = batch['raw_question']  # List[str], keep on CPU

            # Sample-level type-conditional weighting
            sample_weights = None
            if type_sample_weight_cfg is not None and is_training and 'raw_answer' in batch and 'question_type' in batch:
                _TYPE_MAP = {0: 'OBJECT', 1: 'COUNT', 2: 'COLOR', 3: 'LOCATION'}
                _defaults = type_sample_weight_cfg.get('type_defaults', {})
                _count_ov = type_sample_weight_cfg.get('count_overrides', {})
                _sw = []
                for _i, _ans in enumerate(batch['raw_answer']):
                    _tname = _TYPE_MAP.get(batch['question_type'][_i].item(), 'OBJECT')
                    if _tname == 'COUNT':
                        _w = _count_ov.get(_ans, _defaults.get('COUNT', 1.0))
                    else:
                        _w = _defaults.get(_tname, 1.0)
                    _sw.append(_w)
                sample_weights = torch.tensor(_sw, dtype=torch.float32, device=device)

            # Forward pass with mixed precision
            with autocast('cuda', enabled=_amp_on, dtype=amp_dtype):
                outputs = model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    stage=stage,
                    answer_weights=answer_weights,  # Token-level weights
                    question_types=question_types,  # For type-loss aux head
                    images_384=images_384,
                    raw_questions=raw_questions,
                    sample_weights=sample_weights,  # Sample-level type-conditional weights
                )
                
                loss = outputs.total_loss

                # 🔥 R-Drop: second forward pass with different dropout mask → KL consistency
                if use_rdrop and is_training:
                    outputs2 = model(
                        pixel_values=pixel_values,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        stage=stage,
                        answer_weights=answer_weights,
                        question_types=question_types,
                        sample_weights=sample_weights,
                    )
                    l1 = outputs.answer_logits[:, 0, :].float()
                    l2 = outputs2.answer_logits[:, 0, :].float()
                    p1 = F.softmax(l1, dim=-1).clamp(min=1e-8)
                    p2 = F.softmax(l2, dim=-1).clamp(min=1e-8)
                    rdrop_kl = 0.5 * (
                        F.kl_div(p1.log(), p2.detach(), reduction='batchmean') +
                        F.kl_div(p2.log(), p1.detach(), reduction='batchmean')
                    )
                    loss = 0.5 * (loss + outputs2.total_loss) + rdrop_alpha * rdrop_kl

                # 🔥 CDW-CE: ordinal distance penalty for COUNT-type samples only
                if use_cdw_ce and is_training and cdw_ordinal_weights is not None and 'question_type' in batch:
                    _TYPE_MAP = {0: 'OBJECT', 1: 'COUNT', 2: 'COLOR', 3: 'LOCATION'}
                    _qtype_cpu = batch['question_type']
                    _count_mask = torch.tensor(
                        [_TYPE_MAP.get(t.item(), 'X') == 'COUNT' for t in _qtype_cpu],
                        dtype=torch.bool, device=device
                    )
                    if _count_mask.any():
                        _count_logits = outputs.answer_logits[_count_mask, 0, :].float()  # [N, V]
                        _count_labels = labels[_count_mask]  # [N, seq]
                        _ord = cdw_ordinal_weights.to(device)  # [V]
                        _gt_ords = []
                        for _lbl in _count_labels:
                            _valid = _lbl[_lbl != -100]
                            _tok = _valid[0].item() if len(_valid) > 0 else 0
                            _gt_ords.append(_ord[_tok].item())
                        _gt_ords = torch.tensor(_gt_ords, dtype=torch.float32, device=device)  # [N]
                        _probs = F.softmax(_count_logits, dim=-1)  # [N, V]
                        _dist = (_ord.unsqueeze(0) - _gt_ords.unsqueeze(1)).abs()  # [N, V]
                        _cdw_loss = (_probs * _dist).sum(dim=-1).mean()
                        loss = loss + cdw_lambda * _cdw_loss

                # Scale loss for gradient accumulation
                if is_training and gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps
            
            if is_training and loss is not None:
                # Guard: skip batch if loss is NaN/Inf BEFORE backward.
                # Must NOT call scaler.update() here — scaler.scale() hasn't
                # been called yet so there are no inf checks recorded.
                # Calling scaler.update() without a prior scale+backward raises
                # "No inf checks were recorded prior to update."
                if torch.isnan(loss) or torch.isinf(loss):
                    print("⚠️  NaN/Inf loss detected, skipping backward")
                    nan_loss_steps += 1
                    optimizer.zero_grad()
                    steps_since_update = 0
                    continue

                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                # ── SCST backward (accumulated alongside CE gradients) ────────
                if use_scst and current_epoch >= scst_start_epoch:
                    _pv = pixel_values_orig
                    # Sample in eval mode + no_grad:
                    #   1. Disables dropout so sampled sequences aren't noise-corrupted
                    #   2. Avoids building a useless 10-step autoregressive computation graph
                    model.eval()
                    with torch.no_grad():
                        _greedy = model.generate(_pv, input_ids, attention_mask, max_length=10, num_beams=1)
                        _sample = model.generate_sample(_pv, input_ids, attention_mask, max_length=10, temperature=scst_sample_temp)
                    model.train()
                    _adv = []
                    for _i in range(len(_greedy)):
                        _lbl = labels[_i][labels[_i] != -100].cpu().tolist()
                        _gt  = _decode_gt(model.tokenizer, _lbl)
                        _adv.append(compute_exact_match(_sample[_i], _gt) - compute_exact_match(_greedy[_i], _gt))
                    _advantage = torch.tensor(_adv, dtype=torch.float32, device=device)
                    # Positive-only advantage: only reward samples strictly better than greedy.
                    # Negative advantage penalizes the whole sampled sequence including any
                    # correct prefix tokens (e.g. "màu đen" in "màu đen ám ván ván") which
                    # collapses EOS probability and creates repetition loops.
                    _pos_mask = (_advantage > 1e-3)
                    if _pos_mask.any():
                        _sample_enc = model.tokenizer(
                            _sample, truncation=True, padding='max_length',
                            max_length=10, return_tensors='pt'
                        )
                        _sample_ids = _sample_enc['input_ids'].to(device)
                        _sample_ids[_sample_ids == model.tokenizer.pad_token_id] = -100
                        with autocast('cuda', enabled=_amp_on, dtype=amp_dtype):
                            _log_prob = model.compute_seq_logprob(_pv, input_ids, attention_mask, _sample_ids)
                            _masked_adv = (_advantage * _pos_mask.float()).detach()
                            _scst_loss = -(_masked_adv * _log_prob).mean()
                            _scst_scaled = scst_lambda * _scst_loss / gradient_accumulation_steps
                        if scaler is not None:
                            scaler.scale(_scst_scaled).backward()
                        else:
                            _scst_scaled.backward()

                steps_since_update += 1

                # Update weights after accumulating gradients
                if steps_since_update == gradient_accumulation_steps:
                    if scaler is not None:
                        # scaler.unscale_() must come AFTER scale+backward,
                        # BEFORE clip_grad_norm and step.
                        scaler.unscale_(optimizer)
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                        if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                            print("⚠️  NaN/Inf gradient detected, skipping step")
                            nan_grad_steps += 1
                            optimizer.zero_grad()
                            scaler.update()  # safe: scale+backward was called above
                            steps_since_update = 0
                            continue
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                        if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                            print("⚠️  NaN/Inf gradient detected, skipping step")
                            nan_grad_steps += 1
                            optimizer.zero_grad()
                            steps_since_update = 0
                            continue
                        optimizer.step()

                    # 🔥 EMA update after each optimizer step
                    if ema_model is not None:
                        with torch.no_grad():
                            for p_ema, p_m in zip(ema_model.parameters(), model.parameters()):
                                p_ema.data.mul_(ema_decay).add_(p_m.data, alpha=1.0 - ema_decay)

                    optimizer.zero_grad()
                    steps_since_update = 0
            
            # Accumulate metrics (use original loss, not scaled)
            if loss is not None:
                actual_loss = loss.item() * gradient_accumulation_steps if gradient_accumulation_steps > 1 else loss.item()
                total_loss += actual_loss
                total_answer_loss += outputs.answer_loss.item()
                
                # 🔥 NEW: Track type loss if available
                if outputs.type_loss is not None:
                    total_type_loss += outputs.type_loss.item()
                
                # 🔥🔥🔥 NEW: Track distillation losses
                total_vision_kd_loss = 0
                total_text_kd_loss = 0
                if outputs.vision_kd_loss is not None:
                    total_vision_kd_loss += outputs.vision_kd_loss.item()
                if outputs.text_kd_loss is not None:
                    total_text_kd_loss += outputs.text_kd_loss.item()
                
                num_batches += 1
                
                # 🔥 Extract gate statistics + type loss for progress bar
                postfix = {
                    'loss': f"{actual_loss:.3f}",
                    'ans': f"{outputs.answer_loss.item():.3f}"
                }
                
                # Add type loss to display if available
                if outputs.type_loss is not None:
                    postfix['type'] = f"{outputs.type_loss.item():.3f}"
                
                # 🔥🔥🔥 Add distillation losses
                if outputs.vision_kd_loss is not None:
                    postfix['vkd'] = f"{outputs.vision_kd_loss.item():.3f}"
                if outputs.text_kd_loss is not None:
                    postfix['tkd'] = f"{outputs.text_kd_loss.item():.3f}"
                # 🔥 Contrastive loss display
                if outputs.contrastive_loss is not None:
                    postfix['ctr'] = f"{outputs.contrastive_loss.item():.3f}"
                if outputs.divergence_loss is not None:
                    postfix['div'] = f"{outputs.divergence_loss.item():.4f}"
                if outputs.gate_stats is not None:
                    stats = outputs.gate_stats
                    postfix.update({
                        'α_mean': f"{stats['mean']:.2f}",
                        'α_std': f"{stats['std']:.2f}"
                    })
                
                pbar.set_postfix(postfix)

                if log_file is not None:
                    mode = 'TR' if is_training else 'VA'
                    step_str = " ".join(f"{k}={v}" for k, v in postfix.items())
                    log_file.write(f"[E{current_epoch:02d}|{mode}|{batch_idx+1:04d}/{len(dataloader):04d}] {step_str}\n")
                    log_file.flush()

    # Flush remaining accumulated gradients at end of epoch
    if is_training and optimizer is not None and steps_since_update > 0:
        if scaler is not None:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                print("⚠️  NaN/Inf gradient at end-of-epoch flush, skipping step")
                nan_grad_steps += 1
                scaler.update()
            else:
                scaler.step(optimizer)
                scaler.update()
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            if not (torch.isnan(grad_norm) or torch.isinf(grad_norm)):
                optimizer.step()
        # 🔥 EMA update at end-of-epoch flush
        if ema_model is not None:
            with torch.no_grad():
                for p_ema, p_m in zip(ema_model.parameters(), model.parameters()):
                    p_ema.data.mul_(ema_decay).add_(p_m.data, alpha=1.0 - ema_decay)
        optimizer.zero_grad()

    if num_batches == 0:
        return {
            'loss': 0.0,
            'answer_loss': 0.0,
            'type_loss': 0.0
        }
    
    return {
        'loss': total_loss / num_batches,
        'answer_loss': total_answer_loss / num_batches,
        'type_loss': total_type_loss / num_batches if use_type_loss else 0.0,  # 🔥 NEW
        'nan_loss_steps': nan_loss_steps,
        'nan_grad_steps': nan_grad_steps
    }


def evaluate_full_val(model, dataloader, tokenizer, device):
    """
    Tính EM và F1 trên TOÀN BỘ val set — gọi mỗi epoch để track best model.

    Returns:
        dict với 'exact_match', 'f1_score', 'rouge1', 'rougeL' (%)
        và per-type EM: 'em_object', 'em_counting', 'em_color', 'em_location' (nếu có)
    """
    _TYPE_NAMES = {0: 'object', 1: 'counting', 2: 'color', 3: 'location'}
    model.eval()
    exact_matches = []
    f1_scores = []
    rouge1_scores = []
    rougeL_scores = []
    per_type_em: dict = {k: [] for k in _TYPE_NAMES}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Eval EM/F1', leave=False):
            pixel_values   = batch['pixel_values'].to(device)
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels         = batch['labels'].to(device)
            qtypes         = batch.get('question_type', None)

            predictions = model.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=10,
                num_beams=1,
            )

            for idx, (pred, label) in enumerate(zip(predictions, labels)):
                label_tokens = label[label != -100].cpu().tolist()
                gt = _decode_gt(tokenizer, label_tokens)

                em = 1.0 if _normalize_vn(pred) == _normalize_vn(gt) else 0.0
                exact_matches.append(em)
                f1_scores.append(compute_f1_score(pred, gt))

                rouge = compute_rouge_scores(pred, gt)
                rouge1_scores.append(rouge['rouge1'])
                rougeL_scores.append(rouge['rougeL'])

                if qtypes is not None:
                    per_type_em[qtypes[idx].item()].append(em)

    n = len(exact_matches)
    if n == 0:
        return {'exact_match': 0.0, 'f1_score': 0.0, 'rouge1': 0.0, 'rougeL': 0.0}

    result = {
        'exact_match': sum(exact_matches) / n * 100,
        'f1_score':    sum(f1_scores)     / n * 100,
        'rouge1':      sum(rouge1_scores) / n * 100,
        'rougeL':      sum(rougeL_scores) / n * 100,
    }
    for t, ems in per_type_em.items():
        if ems:
            result[f'em_{_TYPE_NAMES[t]}'] = sum(ems) / len(ems) * 100
    # Macro EM: unweighted mean across all 4 types — not dominated by Object (41.6%)
    type_em_vals = [result[f'em_{_TYPE_NAMES[t]}'] for t in _TYPE_NAMES if f'em_{_TYPE_NAMES[t]}' in result]
    if len(type_em_vals) == 4:
        result['macro_em'] = sum(type_em_vals) / 4
    return result


def sample_predictions(model, dataloader, tokenizer, device, num_samples=10, compute_metrics=True):
    """
    Sample predictions for qualitative evaluation with metrics
    
    Returns:
        samples: List of dicts with predictions
        metrics: Dict with EM, F1, ROUGE-1, ROUGE-L scores (if compute_metrics=True)
    """
    model.eval()
    samples = []
    
    exact_matches = []
    f1_scores = []
    rouge1_scores = []
    rougeL_scores = []
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_samples:
                break
            
            pixel_values = batch['pixel_values'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Generate predictions (now with REAL beam search!)
            predictions = model.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=10,
                num_beams=3,
                repetition_penalty=1.3,
            )

            # Decode labels — use _decode_gt for consistency with evaluate_full_val
            label_texts = []
            for label in labels:
                label_tokens = label[label != -100].cpu().tolist()
                label_text = _decode_gt(tokenizer, label_tokens)
                label_texts.append(label_text)
            
            # Decode questions
            question_texts = []
            for inp in input_ids:
                question_text = tokenizer.decode(inp, skip_special_tokens=True)
                question_texts.append(question_text)
            
            # Compute metrics
            for q, pred, gt in zip(question_texts, predictions, label_texts):
                # Exact match với NFC normalization cho tiếng Việt
                em = 1.0 if _normalize_vn(pred) == _normalize_vn(gt) else 0.0
                exact_matches.append(em)
                
                # F1 score
                f1 = compute_f1_score(pred, gt)
                f1_scores.append(f1)
                
                # ROUGE scores
                rouge_scores = compute_rouge_scores(pred, gt)
                rouge1_scores.append(rouge_scores['rouge1'])
                rougeL_scores.append(rouge_scores['rougeL'])
                
                samples.append({
                    'question': q,
                    'prediction': pred,
                    'ground_truth': gt,
                    'exact_match': em,
                    'f1_score': f1,
                    'rouge1': rouge_scores['rouge1'],
                    'rougeL': rouge_scores['rougeL']
                })
    
    metrics = None
    if compute_metrics and exact_matches:
        metrics = {
            'exact_match': sum(exact_matches) / len(exact_matches) * 100,
            'f1_score': sum(f1_scores) / len(f1_scores) * 100,
            'rouge1': sum(rouge1_scores) / len(rouge1_scores) * 100,
            'rougeL': sum(rougeL_scores) / len(rougeL_scores) * 100
        }
    
    return samples, metrics


# ============================================================================
# MAIN TRAINING
# ============================================================================

def main():
    # ========================================================================
    # ARGUMENT PARSER
    # ========================================================================
    
    parser = argparse.ArgumentParser(description='Train Deterministic VQA (No Latent)')
    
    # Data
    parser.add_argument('--data_dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--train_csv', type=str, default=None, help='Path to train CSV file (if not using data_dir/split structure)')
    parser.add_argument('--val_csv', type=str, default=None, help='Path to val CSV file (if not using data_dir/split structure)')
    parser.add_argument('--image_dir', type=str, default=None, help='Path to image directory (if not using data_dir/split structure)')
    parser.add_argument('--val_split', type=float, default=0.1, help='Validation split ratio if val_csv not provided (default: 0.1 = 10%%)')
    parser.add_argument('--batch_size', type=int, default=12, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of dataloader workers')
    
    # PK Sampling
    parser.add_argument('--pk_sampling', action='store_true',
                       help='Enable PK Sampling: P question types × K samples/type per batch. '
                            'Batch size becomes P × K (overrides --batch_size). '
                            'Recommended: --pk_p 4 --pk_k 8 (batch=32) or --pk_p 4 --pk_k 16 (batch=64).')
    parser.add_argument('--pk_p', type=int, default=4,
                       help='P: number of question types per batch for PK Sampling (max 4, default: 4)')
    parser.add_argument('--pk_k', type=int, default=8,
                       help='K: number of samples per type per batch for PK Sampling (default: 8, batch=32)')
    
    # Model
    parser.add_argument('--vision_model', type=str, default='google/siglip-base-patch16-224', 
                       help='Vision encoder model (default: SigLIP-base)')
    parser.add_argument('--bartpho_model', type=str, default='vinai/bartpho-syllable', help='BARTpho model')
    parser.add_argument('--bartpho_revision', type=str, default=None,
                       help='Specific commit hash for BARTpho model (e.g. adf951dd to pin old weights)')
    parser.add_argument('--num_fusion_layers', type=int, default=2, help='Number of Flamingo fusion layers')
    parser.add_argument('--fusion_type', type=str, default='text2vision', 
                       choices=['text2vision', 'vision2text', 'bidirectional'],
                       help='Fusion direction: text2vision (vision attends to text), vision2text (text attends to vision), or bidirectional (both)')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--label_smoothing', type=float, default=0.05,
                       help='Label smoothing factor for answer generation (0.0-0.2, default=0.05)')
    
    # Training
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=2e-5,
                       help='Learning rate for all parameters (flat, default: 2e-5). '
                            'Differential LR was removed — empirically caused regressions.')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    # AdamW optimizer knobs — defaults reproduce prior hard-coded behaviour (eps=1e-8, PyTorch default betas).
    # Paper (KES_SUBMIT) reports eps=1e-10, betas=(0.9, 0.98): pass those to reproduce paper spec.
    parser.add_argument('--adam_eps', type=float, default=1e-8, help='AdamW epsilon (paper: 1e-10)')
    parser.add_argument('--adam_beta1', type=float, default=0.9, help='AdamW beta1 (paper: 0.9)')
    parser.add_argument('--adam_beta2', type=float, default=0.999, help='AdamW beta2 (paper: 0.98)')
    parser.add_argument('--max_norm', type=float, default=1.0, help='Gradient clipping max norm')
    parser.add_argument('--gate_lr_multiplier', type=float, default=1.0,
                       help='LR multiplier for vision_gating params. '
                            'Gate gradient is weak after GCA alignment — multiply to compensate.')
    parser.add_argument('--gate_no_weight_decay', action='store_true',
                       help='Set weight_decay=0 for gate params. '
                            'WD pushes gate weights → 0 → uniform gating; removing it lets gate diversify.')
    parser.add_argument('--gate_detach_input', action='store_true',
                       help='Detach v_proj before gate_net input. Cuts gradient feedback loop '
                            'gate_net → vision_proj → Flamingo that causes training oscillation. '
                            'Flamingo still receives decoder gradient via alpha*v_proj output.')
    parser.add_argument('--use_delta_gate', action='store_true',
                       help='Delta gate: gate_input = cat([orig_proj(v_orig), delta_proj(v_delta), q]). '
                            'v_orig=pre-Flamingo SigLIP features, v_delta=v_fused−v_orig (Flamingo attention '
                            'fingerprint per patch). Gives gate strong per-question per-patch signal '
                            'for true instance-level gating. Requires --gate_detach_input.')
    parser.add_argument('--text_only_warmup_epochs', type=int, default=0,
                       help='H3: number of initial epochs to run in text-only mode (zero vision). '
                            'Decoder learns Vietnamese answer patterns (numbers, colors, locations) '
                            'before vision is introduced. Reduces cold-start COUNT=0%% at E01-E02.')
    parser.add_argument('--no_amp', action='store_true', help='Disable automatic mixed precision')
    parser.add_argument('--amp_dtype', type=str, default='bf16', choices=['fp16', 'bf16', 'fp32'],
                       help='AMP dtype (fp16/bf16/fp32). Use bf16 on H100 for stability.')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, 
                       help='Number of gradient accumulation steps (for effective larger batch size)')
    # GradScaler tuning (helps H100 BF16/FP16 stability)
    parser.add_argument('--scaler_init_scale', type=float, default=256.0,
                       help='GradScaler initial scale (default: 256)')
    parser.add_argument('--scaler_growth_factor', type=float, default=1.5,
                       help='GradScaler growth factor (default: 1.5)')
    parser.add_argument('--scaler_backoff_factor', type=float, default=0.5,
                       help='GradScaler backoff factor (default: 0.5)')
    parser.add_argument('--scaler_growth_interval', type=int, default=2000,
                       help='GradScaler growth interval (default: 2000 steps)')
    
    # LR scheduler & early stopping
    parser.add_argument('--scheduler', type=str, default='plateau', choices=['none', 'plateau', 'cosine', 'cosine_restart'],
                       help='LR scheduler type (cosine_restart = SGDR with warm restarts)')
    parser.add_argument('--scheduler_patience', type=int, default=3, help='Patience for ReduceLROnPlateau')
    parser.add_argument('--scheduler_factor', type=float, default=0.5, help='Factor for ReduceLROnPlateau')
    parser.add_argument('--warmup_epochs', type=int, default=0,
                       help='Linear warmup epochs before main scheduler (default: 0, recommend 3 for bidirectional)')
    parser.add_argument('--early_stopping', action='store_true', help='Enable early stopping')
    parser.add_argument('--early_stopping_patience', type=int, default=8, help='Early stopping patience')
    parser.add_argument('--early_stopping_metric', type=str, default='em',
                       choices=['loss', 'em', 'f1', 'rouge1', 'rougeL', 'macro_em'],
                       help='Metric to monitor for early stopping and best model (default: em). Use macro_em for unweighted average across all 4 types.')
    
    # Freezing
    parser.add_argument('--unfreeze_encoder_layers', type=int, default=3, help='Number of text encoder layers to unfreeze')
    parser.add_argument('--freeze_decoder', action='store_true', help='Freeze decoder (default: unfrozen)')
    
    # 🔥 Vision Adaptation (LoRA recommended for low-resource)
    parser.add_argument('--use_vision_lora', action='store_true',
                       help='Use LoRA for vision encoder adaptation (RECOMMENDED for ~10K samples)')
    parser.add_argument('--vision_lora_r', type=int, default=8,
                       help='LoRA rank for vision encoder (default: 8, safe for low-resource)')
    parser.add_argument('--vision_lora_alpha', type=int, default=16,
                       help='LoRA alpha scaling (default: 16)')
    parser.add_argument('--vision_lora_dropout', type=float, default=0.1,
                       help='LoRA dropout rate (default: 0.1)')
    
    # 🔥 Text Encoder Adaptation (LoRA vs unfreeze layers)
    parser.add_argument('--use_text_lora', action='store_true',
                       help='Use LoRA for text encoder (BETTER than unfreezing layers for ~10K samples)')
    parser.add_argument('--text_lora_r', type=int, default=16,
                       help='LoRA rank for text encoder (default: 16, higher than vision)')
    parser.add_argument('--text_lora_alpha', type=int, default=32,
                       help='LoRA alpha scaling for text (default: 32)')
    parser.add_argument('--text_lora_dropout', type=float, default=0.1,
                       help='LoRA dropout for text encoder (default: 0.1)')

    # 🔥 Vision Dependency (combat text shortcut)
    parser.add_argument('--use_vision_gate', action='store_true',
                       help='Enable learnable vision gating (boost vision importance)')
    parser.add_argument('--vision_gate_init', type=float, default=1.5,
                       help='Initial vision gate value (>1.0 = prefer vision, default=1.5)')
    parser.add_argument('--vision_gate_min_alpha', type=float, default=0.35,
                       help='Minimum alpha floor for VisionGating (0.35 recommended to prevent '
                            'COLOR/COUNT from over-suppressing vision: α≈0.18/0.33 without floor)')
    parser.add_argument('--use_mean_pool_cls', action='store_true',
                       help='Use mean-pool of valid tokens as text_cls instead of BOS (position 0). '
                            'BOS token in BARTpho has low norm (4.35 vs 19.11 for content) and near-zero '
                            'cosine similarity with sentence meaning — mean pooling gives a richer signal '
                            'for TypePredictionHead and VisionGating.')
    parser.add_argument('--use_attn_pool_cls', action='store_true',
                       help='Use learned attention pooling over encoder tokens as text_cls. '
                            'A trainable scoring vector (Linear(D,1)) softmax-weights each token, '
                            'letting the model focus on content words (object nouns, color adjectives) '
                            'rather than BOS (no context) or mean-pool (diluted by function words). '
                            'Mutually exclusive with --use_mean_pool_cls; attn_pool takes priority.')
    parser.add_argument('--use_siglip_pooler', action='store_true',
                       help='Prepend SigLIP pooler_output (global image feature) as an extra vision token '
                            'before the 196 patch tokens. Gives decoder access to SigLIP\'s trained global '
                            'image representation (CLS-aggregated), not just local patches.')
    parser.add_argument('--use_type_text_adapter', action='store_true',
                       help='Add type-specific bottleneck adapters on BARTpho encoder output. '
                            '4 separate up-projections (one per question type) eliminate cross-type '
                            'LoRA gradient interference. Zero-init → pure residual at start. '
                            'Training uses gold type labels; inference uses TypePredictionHead output.')
    parser.add_argument('--type_text_adapter_bottleneck', type=int, default=64,
                       help='Bottleneck dimension for TypeSpecificTextAdapter (default: 64 ≈ 330K params).')
    parser.add_argument('--vision_gate_max_alpha', type=float, default=1.0,
                       help='Maximum alpha ceiling for VisionGating. '
                            'Uses scaled sigmoid α=min+(max-min)·σ(·) so gradient never dies at boundary. '
                            'Default=1.0 (original formula). '
                            '0.85 recommended with --use_delta_gate to prevent α→1.0 saturation '
                            'that kills gate discriminativity (α_std→0). '
                            'Forces gate to keep ≥(1-max) text contribution, '
                            'maintaining non-zero gradient signal through (v_proj - text_pooled).')
    
    # 🔥 Type-Conditioned Vision Adapter (NEW!)
    parser.add_argument('--use_type_adapter', action='store_true',
                       help='Enable type-conditioned vision adapter (4 expert networks)')
    parser.add_argument('--type_adapter_rank', type=int, default=64,
                       help='Low-rank bottleneck for adapter experts (default: 64)')
    parser.add_argument('--type_adapter_bias', type=float, default=2.0,
                       help='Type supervision bias strength (default: 2.0)')
    
    # 🔥 Answer-aware & Type-conditional Loss
    parser.add_argument('--answer_weights', type=str, default=None,
                       help='Path to answer_weights.json for balanced loss (use compute_answer_weights.py)')
    parser.add_argument('--type_sample_weights', type=str, default=None,
                       help='Path to type_sample_weights.json for sample-level type-conditional loss weighting. '
                            'Applies per-example multipliers based on (question_type, answer) — no token-level ambiguity.')
    parser.add_argument('--use_type_loss', action='store_true',
                       help='Enable type prediction head auxiliary loss (TypePredictionHead, safe)')
    parser.add_argument('--type_loss_weight', type=float, default=0.2,
                       help='Weight for auxiliary type loss (default: 0.2, try 0.5 for stronger type learning)')
    parser.add_argument('--focal_gamma', type=float, default=0.0,
                       help='Focal loss gamma (0=standard CE, 2=standard focal loss). Down-weights easy examples dynamically.')
    parser.add_argument('--type_label_smoothing', type=str, default=None,
                       help='Per-type label smoothing as JSON: e.g. \'{"0":0.1,"1":0.0,"2":0.05,"3":0.1}\'. '
                            'Keys: 0=OBJECT 1=COUNT 2=COLOR 3=LOCATION. Falls back to --label_smoothing for missing keys.')
    parser.add_argument('--use_logits_bias', action='store_true',
                       help='Enable type-aware logits biasing (TypeAwareLogitsBias, risky - use separately)')
    
    # 🔥🔥🔥 ONLINE KNOWLEDGE DISTILLATION 🔥🔥🔥
    parser.add_argument('--use_distillation', action='store_true',
                       help='Enable online knowledge distillation from large teachers')
    parser.add_argument('--distill_vision', action='store_true', default=True,
                       help='Use vision KD component (SigLIP-SO400M teacher). Default: ON when --use_distillation')
    parser.add_argument('--no_distill_vision', dest='distill_vision', action='store_false',
                       help='Disable vision KD component (keep only text KD)')
    parser.add_argument('--distill_text', action='store_true', default=True,
                       help='Use text KD component (PhoBERT-large teacher). Default: ON when --use_distillation')
    parser.add_argument('--no_distill_text', dest='distill_text', action='store_false',
                       help='Disable text KD component (keep only vision KD)')
    parser.add_argument('--vision_teacher', type=str, default='google/siglip-so400m-patch14-384',
                       help='Vision teacher model (default: SigLIP-SO400M at 384px)')
    parser.add_argument('--text_teacher', type=str, default='vinai/phobert-large',
                       help='Text teacher model (default: PhoBERT-large)')
    parser.add_argument('--distill_alpha', type=float, default=0.5,
                       help='Distillation weight: CE + α*KD_normalized (default: 0.5, recommended: 0.1-0.2)')
    
    # 🔥 Cross-Modal Contrastive Alignment Loss
    parser.add_argument('--use_contrastive', action='store_true',
                       help='Enable cross-modal contrastive alignment loss (InfoNCE fused_vision ↔ text_cls). '
                            'Addresses English-vision / Vietnamese-text alignment gap. '
                            'Recommended: --contrastive_lambda 0.1 --contrastive_temp 0.07')
    parser.add_argument('--contrastive_lambda', type=float, default=0.1,
                       help='Weight λ_c for contrastive loss (default: 0.1, range: 0.05–0.15)')
    parser.add_argument('--contrastive_temp', type=float, default=0.07,
                       help='Temperature τ for InfoNCE (default: 0.07, SimCLR standard)')
    parser.add_argument('--use_gate_divergence', action='store_true',
                       help='Enable inter-type gate divergence loss. Forces VisionGating to produce '
                            'different alpha patterns per question type (data-driven, no manual targets). '
                            'Requires --use_vision_gate and --pk_sampling. '
                            'Recommended: --gate_divergence_lambda 0.05')
    parser.add_argument('--gate_divergence_lambda', type=float, default=0.05,
                       help='Weight λ_div for inter-type gate divergence loss (default: 0.05, range: 0.03–0.1)')
    
    # Image augmentation
    parser.add_argument('--use_img_aug', action='store_true',
                       help='Safe photometric augmentation: brightness+contrast+blur (p=0.5 per sample). '
                            'No geometric transforms — safe for all 4 question types. Training only.')

    # EMA (Exponential Moving Average weights)
    parser.add_argument('--use_ema', action='store_true',
                       help='Maintain EMA copy of weights (τ=ema_decay). Use EMA model for val eval '
                            'and checkpoint saving. Smooths oscillations → more stable LOCATION/COUNT.')
    parser.add_argument('--ema_decay', type=float, default=0.999,
                       help='EMA decay rate τ (default: 0.999). Higher = slower update, more stable.')

    # R-Drop regularization
    parser.add_argument('--use_rdrop', action='store_true',
                       help='Enable R-Drop: run two dropout sub-models per batch, add bidirectional KL '
                            'consistency penalty. Forces same input → same output regardless of dropout mask. '
                            '~1.8x forward cost. Recommended: --rdrop_alpha 0.1')
    parser.add_argument('--rdrop_alpha', type=float, default=0.1,
                       help='Weight α for R-Drop KL penalty (default: 0.1, range: 0.05–0.3)')

    # CDW-CE ordinal loss for COUNT
    parser.add_argument('--use_cdw_ce', action='store_true',
                       help='Enable Class Distance Weighted CE for COUNT-type questions. '
                            'Penalizes off-by-one errors less than large misses (e.g. "hai" vs "ba" < "hai" vs "mười"). '
                            'Zero risk to LOCATION — gated strictly to COUNT samples.')
    parser.add_argument('--cdw_lambda', type=float, default=0.1,
                       help='Weight λ for CDW-CE ordinal penalty (default: 0.1, range: 0.05–0.3)')

    # Curriculum Learning
    parser.add_argument('--curriculum', action='store_true',
                       help='Enable curriculum learning: train easy (frequent) answers first, gradually add harder ones')
    parser.add_argument('--curriculum_start_ratio', type=float, default=0.4,
                       help='Fraction of easiest samples used in epoch 1 (default: 0.4, linearly ramps to 1.0)')

    # SCST (Self-Critical Sequence Training)
    parser.add_argument('--use_scst', action='store_true',
                       help='Enable SCST: optimise F1 reward directly via REINFORCE after CE warmup')
    parser.add_argument('--scst_start_epoch', type=int, default=5,
                       help='Epoch from which SCST loss is added (default: 5, after CE warmup)')
    parser.add_argument('--scst_lambda', type=float, default=0.1,
                       help='Weight of SCST loss relative to CE loss (default: 0.1)')
    parser.add_argument('--scst_sample_temp', type=float, default=1.0,
                       help='Sampling temperature for SCST rollouts (default: 1.0)')

    # Vision dropout override (two-phase training)
    parser.add_argument('--vision_dropout_rate', type=float, default=None,
                       help='Override vision dropout rate (0.0-1.0). '
                            'If None, auto-set by fusion_type (text2vision=0.10, bidirectional=0.05).')

    # Text/question dropout (Phase 1 image-flow training)
    parser.add_argument('--text_dropout_rate', type=float, default=0.0,
                       help='Question dropout rate (0.0-1.0): mask question with padding tokens. '
                            'Use high value (0.85) in Phase 1 so model learns from image only. '
                            'Set 0.0 (default) for normal training with both image + question.')

    # Checkpointing
    parser.add_argument('--output_dir', type=str, default='./checkpoints_no_latent', help='Output directory for checkpoints')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--resume_reset_epoch', action='store_true',
                       help='When resuming, reset epoch counter and early-stopping state (fresh schedule with loaded weights)')
    parser.add_argument('--reset_lr', action='store_true',
                       help='When resuming, reset LR to --lr value and reinitialize scheduler (warm restart)')
    parser.add_argument('--save_every', type=int, default=1, help='Save checkpoint every N epochs')
    parser.add_argument('--sample_every', type=int, default=3, help='Sample predictions every N epochs')

    # SWA (Stochastic Weight Averaging)
    parser.add_argument('--use_swa', action='store_true',
                       help='Enable SWA: uniform average of model snapshots after swa_start_epoch')
    parser.add_argument('--swa_start_epoch', type=int, default=20,
                       help='Epoch from which SWA starts collecting snapshots (default: 20)')
    parser.add_argument('--swa_lr', type=float, default=1e-5,
                       help='Constant LR used during SWA phase (default: 1e-5)')

    # SGDR (Cosine Annealing with Warm Restarts)
    parser.add_argument('--sgdr_t0', type=int, default=10,
                       help='T_0: first restart period in epochs for cosine_restart scheduler (default: 10)')

    # Misc
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--no_gradient_checkpointing', action='store_true', help='Disable gradient checkpointing')
    parser.add_argument('--analyze_dataset', action='store_true', help='Analyze dataset before training')
    
    # Weights & Biases (optional)
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases for experiment tracking')
    parser.add_argument('--wandb_project', type=str, default='vietnamese-vqa-deterministic', 
                       help='W&B project name')
    parser.add_argument('--wandb_name', type=str, default=None, help='W&B run name (auto-generated if None)')
    
    args = parser.parse_args()
    
    # Validate data arguments
    if args.train_csv or args.image_dir:
        if not (args.train_csv and args.image_dir):
            raise ValueError("If using CSV structure, must provide both: --train_csv and --image_dir")
        # val_csv is optional - will auto-split if not provided
    
    # ========================================================================
    # Random seed (basic setup like 7/2)
    # ========================================================================
    # Set all random seeds for reproducibility
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    # NOTE: cudnn.deterministic=True breaks MBart SDPA on newer transformers
    # (CUDNN_STATUS_NOT_INITIALIZED on H100 MIG). benchmark=False is enough
    # to reduce cross-run variance for conv-heavy vision encoders.
    torch.backends.cudnn.benchmark = False
    # TF32 left at H100 default (True) — matches run87 training conditions.
    
    # ========================================================================
    # CONFIG (from args)
    # ========================================================================
    
    # Data
    data_dir = args.data_dir
    batch_size = args.batch_size
    num_workers = args.num_workers
    
    # Model
    vision_model = args.vision_model
    bartpho_model = args.bartpho_model
    num_fusion_layers = args.num_fusion_layers
    
    # Training (Stage 3 ONLY)
    stage3_epochs = args.epochs
    learning_rate = args.lr
    weight_decay = args.weight_decay
    max_norm = args.max_norm
    use_amp = not args.no_amp and args.amp_dtype != 'fp32'
    amp_dtype_map = {
        'fp16': torch.float16,
        'bf16': torch.bfloat16,
        'fp32': torch.float32,
    }
    amp_dtype = amp_dtype_map[args.amp_dtype]
    
    # Freezing strategy
    unfreeze_encoder_layers = args.unfreeze_encoder_layers
    unfreeze_decoder = not args.freeze_decoder
    
    # Checkpointing
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    train_log_path = os.path.join(output_dir, 'train_log.txt')
    train_log_file = open(train_log_path, 'a', encoding='utf-8')
    print(f"[Log] Step-level log → {train_log_path}")

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Device] Using: {device}")
    
    print("\n" + "="*80)
    print("TRAINING CONFIGURATION")
    print("="*80)
    print(f"  Data dir: {data_dir}")
    print(f"  Batch size: {batch_size}")
    if args.pk_sampling:
        print(f"  🔥 PK Sampling: P={args.pk_p} types × K={args.pk_k} samples → batch={args.pk_p * args.pk_k}")
    print(f"  Epochs: {stage3_epochs}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Weight decay: {weight_decay}")
    print(f"  Gradient clipping: {max_norm}")
    print(f"  Mixed precision: {use_amp} ({args.amp_dtype})")
    print(f"  Fusion layers: {num_fusion_layers}")
    print(f"  Unfreeze encoder layers: {unfreeze_encoder_layers}")
    print(f"  Unfreeze decoder: {unfreeze_decoder}")
    if args.use_vision_lora:
        print(f"  🔥 Vision LoRA: r={args.vision_lora_r}, alpha={args.vision_lora_alpha}, dropout={args.vision_lora_dropout}")
    if args.use_text_lora:
        print(f"  🔥 Text LoRA: r={args.text_lora_r}, alpha={args.text_lora_alpha}, dropout={args.text_lora_dropout}")
    if args.answer_weights:
        print(f"  🔥 Answer-aware loss: {args.answer_weights}")
    if args.use_type_loss:
        print(f"  🔥 Type-conditional loss: 1.5x counting, 1.4x location, 1.3x color")
        print(f"  🔥 Type loss weight: {args.type_loss_weight}")
    if args.use_contrastive:
        print(f"  🔥 Contrastive alignment: λ={args.contrastive_lambda}, τ={args.contrastive_temp}")
    if args.use_gate_divergence:
        print(f"  🔥 Gate divergence: λ={args.gate_divergence_lambda}")
    if args.use_img_aug:
        print(f"  🔥 Image aug: brightness+contrast(0.25) + GaussianBlur (p=0.5, train only)")
    if args.use_ema:
        print(f"  🔥 EMA weights: τ={args.ema_decay} (val/save uses EMA model)")
    if args.use_rdrop:
        print(f"  🔥 R-Drop: α={args.rdrop_alpha} (2x forward pass, KL consistency)")
    if args.use_cdw_ce:
        print(f"  🔥 CDW-CE ordinal loss: λ={args.cdw_lambda} (COUNT only)")
    print(f"  Output dir: {output_dir}")
    print(f"  Random seed: {args.seed}")
    print("="*80 + "\n")
    
    # 🔥 Load answer weights if provided
    answer_weights_tensor = None
    if args.answer_weights:
        print(f"[Answer Weights] Loading from {args.answer_weights}...")
        import json
        with open(args.answer_weights, 'r', encoding='utf-8') as f:
            weights_data = json.load(f)

        token_weights = weights_data['token_weights']
        answer_weights_tensor = torch.tensor(token_weights, dtype=torch.float32, device=device)

        print(f"  Loaded {len(token_weights)} token weights")
        print(f"  Weight range: [{min(token_weights):.2f}, {max(token_weights):.2f}]")
        print(f"  Weighted tokens: {(answer_weights_tensor > 1.0).sum().item()}/{len(token_weights)}")

    # Load type_sample_weights config if provided
    type_sample_weight_cfg = None
    if args.type_sample_weights:
        print(f"[Type Sample Weights] Loading from {args.type_sample_weights}...")
        import json
        with open(args.type_sample_weights, 'r', encoding='utf-8') as f:
            type_sample_weight_cfg = json.load(f)
        print(f"  Type defaults: {type_sample_weight_cfg.get('type_defaults', {})}")
        print(f"  COUNT overrides: {type_sample_weight_cfg.get('count_overrides', {})}")
    
    # 🔥 Initialize Weights & Biases (optional)
    if args.use_wandb:
        if not WANDB_AVAILABLE:
            print("⚠️  Warning: wandb not installed. Logging disabled.")
            print("   Install with: pip install wandb")
            args.use_wandb = False
        else:
            run_name = args.wandb_name or f"exp_{args.scheduler}_lr{args.lr}_bs{batch_size}"
            wandb.init(
                project=args.wandb_project,
                name=run_name,
                config=vars(args),
                tags=['deterministic', 'no-latent', f'scheduler-{args.scheduler}']
            )
            print(f"[W&B] Initialized: {args.wandb_project}/{run_name}")
            print(f"[W&B] View at: {wandb.run.url}\n")
    
    # ========================================================================
    # DATASET
    # ========================================================================
    
    print("\n[Data] Loading datasets...")
    
    # Check if using CSV/image_dir structure or data_dir/split structure
    if args.train_csv and args.image_dir:
        print("[Data] Using CSV + image directory structure")
        # Need to check if VQAGenDataset exists in dataset.py
        from dataset import VQAGenDataset
        from transformers import AutoProcessor
        from torch.utils.data import random_split
        
        vision_processor = AutoProcessor.from_pretrained(vision_model)
        
        # 🔥🔥🔥 Load teacher vision processor if vision KD enabled
        teacher_vision_processor = None
        if args.use_distillation and args.distill_vision:
            print(f"[Distillation] Loading teacher vision processor: {args.vision_teacher}")
            teacher_vision_processor = AutoProcessor.from_pretrained(args.vision_teacher)
        
        # Load full training dataset
        full_train_dataset = VQAGenDataset(
            csv_path=args.train_csv,
            image_folder=args.image_dir,
            vision_processor=vision_processor,
            tokenizer_name=bartpho_model,
            include_question_type=args.use_type_loss or (args.type_sample_weights is not None) or args.use_cdw_ce,
            auto_detect_type=False,  # Dùng cột 'type' từ CSV (ground truth), không dùng regex
            use_distillation=args.use_distillation,  # 🔥🔥🔥
            teacher_vision_processor=teacher_vision_processor,  # 🔥🔥🔥
            use_img_aug=args.use_img_aug,
            is_training=True,
        )

        # Check if val_csv provided
        if args.val_csv:
            print(f"[Data] Using provided validation CSV: {args.val_csv}")
            val_dataset = VQAGenDataset(
                csv_path=args.val_csv,
                image_folder=args.image_dir,
                vision_processor=vision_processor,
                tokenizer_name=bartpho_model,
                include_question_type=args.use_type_loss or (args.type_sample_weights is not None) or args.use_cdw_ce,
                auto_detect_type=False,  # Dùng cột 'type' từ CSV (ground truth)
                use_distillation=args.use_distillation,
                teacher_vision_processor=teacher_vision_processor
            )
            train_dataset = full_train_dataset  # 10,800 mẫu (đã tách val ra rồi)
            print(f"[Data] Train: {len(train_dataset):,} | Val: {len(val_dataset):,}")
        else:
            # Auto-split train into train/val
            val_ratio = args.val_split
            val_size = int(len(full_train_dataset) * val_ratio)
            train_size = len(full_train_dataset) - val_size
            
            print(f"[Data] No val_csv provided. Auto-splitting with {val_ratio*100:.0f}% validation")
            print(f"[Data] Split: {train_size} train + {val_size} val = {len(full_train_dataset)} total")
            
            # Set seed for reproducible split
            generator = torch.Generator().manual_seed(args.seed)
            train_dataset, val_dataset = random_split(
                full_train_dataset, 
                [train_size, val_size],
                generator=generator
            )
    else:
        print("[Data] Using data_dir + split structure")
        train_dataset = VQAGenDataset(
            data_dir=data_dir,
            split='train',
            bartpho_model_name=bartpho_model
        )
        
        val_dataset = VQAGenDataset(
            data_dir=data_dir,
            split='val',
            bartpho_model_name=bartpho_model
        )
    
    # Create generator for reproducible shuffling
    train_generator = torch.Generator().manual_seed(args.seed)

    # ── Curriculum sampler (overrides PK sampler if both set) ─────────────
    curriculum_sampler = None
    if args.curriculum:
        _actual_train = train_dataset.dataset if hasattr(train_dataset, 'dataset') else train_dataset
        curriculum_sampler = CurriculumSampler(
            dataset=_actual_train,
            seed=args.seed,
            start_ratio=args.curriculum_start_ratio,
        )
        print(f"[Curriculum] Enabled: start_ratio={args.curriculum_start_ratio:.1%} → 100% over {stage3_epochs} epochs")

    # ── PK Sampling setup ─────────────────────────────────────────────────
    pk_sampler = None
    if args.pk_sampling and curriculum_sampler is None:
        from dataset import PKSampler
        random.seed(args.seed)  # PKSampler uses Python random internally
        pk_sampler = PKSampler(
            dataset=train_dataset,
            p=args.pk_p,
            k=args.pk_k,
            shuffle=True,
            drop_last=True
        )
        # When using PKSampler, effective batch size = P × K
        pk_batch_size = args.pk_p * args.pk_k
        print(f"[PK Sampling] Enabled: P={args.pk_p} × K={args.pk_k} = batch_size={pk_batch_size}")
        print(f"[PK Sampling] Batches/epoch: {len(pk_sampler) // pk_batch_size}")
    
    def _worker_init_fn(worker_id):
        seed = args.seed + worker_id
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    _train_sampler = curriculum_sampler or pk_sampler
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.pk_p * args.pk_k if args.pk_sampling else batch_size,
        shuffle=False if _train_sampler is not None else True,
        sampler=_train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None,
        generator=train_generator if _train_sampler is None else None,
        worker_init_fn=_worker_init_fn if num_workers > 0 else None
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None,
        worker_init_fn=_worker_init_fn if num_workers > 0 else None
    )

    print(f"[Data] Train: {len(train_dataset)} samples")
    print(f"[Data] Val: {len(val_dataset)} samples")
    
    # Dataset analysis (if requested)
    if args.analyze_dataset:
        # Get tokenizer from dataset
        from torch.utils.data import Subset
        actual_dataset = train_dataset.dataset if isinstance(train_dataset, Subset) else train_dataset
        tokenizer = actual_dataset.tokenizer
        
        analyze_dataset(train_dataset, tokenizer, num_samples=1000)
    
    # ========================================================================
    # MODEL
    # ========================================================================
    
    print("\n[Model] Building Deterministic VQA...")
    model = DeterministicVQA(
        vision_model_name=vision_model,
        bartpho_model_name=bartpho_model,
        bartpho_revision=args.bartpho_revision,
        num_fusion_layers=num_fusion_layers,
        fusion_type=args.fusion_type,  # 🔥 NEW: Fusion direction
        num_heads=args.num_heads,
        dropout=args.dropout,
        label_smoothing=args.label_smoothing,
        focal_gamma=args.focal_gamma,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        use_vision_lora=args.use_vision_lora,  # 🔥 LoRA for vision encoder
        vision_lora_r=args.vision_lora_r,
        vision_lora_alpha=args.vision_lora_alpha,
        vision_lora_dropout=args.vision_lora_dropout,
        use_text_lora=args.use_text_lora,  # 🔥 NEW: LoRA for text encoder
        text_lora_r=args.text_lora_r,  # 🔥 NEW
        text_lora_alpha=args.text_lora_alpha,  # 🔥 NEW
        text_lora_dropout=args.text_lora_dropout,  # 🔥 NEW
        use_type_task=args.use_type_loss,       # 🔥 type head auxiliary loss only
        use_logits_bias=args.use_logits_bias,   # 🔥 type-aware logits bias (risky, separate flag)
    type_loss_weight=args.type_loss_weight,
        use_vision_gate=args.use_vision_gate,
        vision_gate_init=args.vision_gate_init,
        vision_gate_min_alpha=args.vision_gate_min_alpha,
        vision_gate_max_alpha=args.vision_gate_max_alpha,
        use_type_adapter=args.use_type_adapter,  # 🔥 NEW: Type-conditioned adapter
        type_adapter_rank=args.type_adapter_rank,  # 🔥 NEW
        type_adapter_bias=args.type_adapter_bias,  # 🔥 NEW
        use_distillation=args.use_distillation,  # 🔥🔥🔥 ONLINE DISTILLATION
        distill_vision=args.distill_vision,      # 🔥🔥🔥 Vision KD on/off
        distill_text=args.distill_text,          # 🔥🔥🔥 Text KD on/off
        vision_teacher_name=args.vision_teacher,  # 🔥🔥🔥
        text_teacher_name=args.text_teacher,  # 🔥🔥🔥
        distill_alpha=args.distill_alpha,  # 🔥🔥🔥
        use_contrastive=args.use_contrastive,          # 🔥 Cross-modal contrastive
        contrastive_lambda=args.contrastive_lambda,    # 🔥
        contrastive_temp=args.contrastive_temp,        # 🔥
        use_gate_divergence=args.use_gate_divergence,          # Inter-type gate divergence
        gate_divergence_lambda=args.gate_divergence_lambda,    # λ_div
        use_delta_gate=args.use_delta_gate,
        use_mean_pool_cls=args.use_mean_pool_cls,
        use_attn_pool_cls=args.use_attn_pool_cls,
        use_siglip_pooler=args.use_siglip_pooler,
        use_type_text_adapter=args.use_type_text_adapter,
        type_text_adapter_bottleneck=args.type_text_adapter_bottleneck,
    ).to(device)
    
    model.gate_detach_input = args.gate_detach_input
    model.text_only_mode = False  # controlled per-epoch by --text_only_warmup_epochs
    if args.gate_detach_input:
        print("[Gate] detach_for_gate=True: gate_net gradient cut from Flamingo (stops feedback oscillation)")
    if args.use_delta_gate:
        print("[Gate] delta_gate=True: gate_input=cat([v_orig, v_delta, q]) — instance-level Flamingo fingerprint")
        if not args.gate_detach_input:
            print("[Gate] WARNING: --use_delta_gate without --gate_detach_input: "
                  "gate_net gradient flows into v_orig (SigLIP frozen) and v_delta (Flamingo). "
                  "This can cause Flamingo oscillation. Consider adding --gate_detach_input.")
    if args.vision_gate_max_alpha < 1.0:
        print(f"[Gate] max_alpha={args.vision_gate_max_alpha:.2f}: "
              f"scaled sigmoid α∈[{args.vision_gate_min_alpha:.2f}, {args.vision_gate_max_alpha:.2f}] — "
              f"gradient never dies at boundary, prevents α→1 saturation")

    model.freeze_pretrained(
        unfreeze_encoder_layers=unfreeze_encoder_layers,
        unfreeze_decoder=unfreeze_decoder
    )

    # Type-weighted CE for TypePredictionHead — inverse-frequency weights per type
    if args.use_type_loss and hasattr(full_train_dataset, 'data') and 'type' in full_train_dataset.data.columns:
        type_counts = full_train_dataset.data['type'].value_counts()
        n_total = len(full_train_dataset)
        type_weights = torch.zeros(4, device=device)
        for t in range(4):
            cnt = int(type_counts.get(t, 1))
            type_weights[t] = n_total / (4 * cnt)
        model.type_class_weights = type_weights
        print(f"[TypeLoss] Class weights: Object={type_weights[0]:.3f}, Count={type_weights[1]:.3f}, "
              f"Color={type_weights[2]:.3f}, Location={type_weights[3]:.3f}")

    # Per-type label smoothing (e.g. COUNT=0.0, COLOR=0.05)
    if args.type_label_smoothing:
        import json
        raw = json.loads(args.type_label_smoothing)
        model.type_label_smoothing = {int(k): float(v) for k, v in raw.items()}
        print(f"[Loss] Per-type label smoothing: {model.type_label_smoothing} "
              f"(fallback={args.label_smoothing})")

    # 🔥 EMA: create shadow copy of model weights (after model moved to device)
    import copy as _copy
    ema_model = None
    if args.use_ema:
        ema_model = _copy.deepcopy(model)
        ema_model.requires_grad_(False)
        print(f"[EMA] Initialized shadow model (τ={args.ema_decay})")

    # 🔥 SWA: uniform average of periodic snapshots after swa_start_epoch
    swa_model = None
    swa_n = 0  # number of snapshots averaged so far
    if args.use_swa:
        from torch.optim.swa_utils import AveragedModel
        swa_model = AveragedModel(model)
        swa_model.requires_grad_(False)
        print(f"[SWA] Initialized averaged model (start_epoch={args.swa_start_epoch}, swa_lr={args.swa_lr})")

    # 🔥 CDW-CE: precompute ordinal values for Vietnamese number tokens
    cdw_ordinal_weights = None
    if args.use_cdw_ce:
        _VIET_NUM = {
            'không': 0, 'một': 1, 'hai': 2, 'ba': 3, 'bốn': 4,
            'năm': 5, 'sáu': 6, 'bảy': 7, 'tám': 8, 'chín': 9, 'mười': 10,
        }
        _tok = model.tokenizer
        _vocab_size = _tok.vocab_size
        cdw_ordinal_weights = torch.zeros(_vocab_size, dtype=torch.float32)
        _mapped = 0
        for _word, _val in _VIET_NUM.items():
            _ids = _tok.encode(_word, add_special_tokens=False)
            if len(_ids) == 1 and _ids[0] < _vocab_size:
                cdw_ordinal_weights[_ids[0]] = float(_val)
                _mapped += 1
        print(f"[CDW-CE] Mapped {_mapped}/{len(_VIET_NUM)} Vietnamese number words to ordinal values")

    # 🚀 SPEED OPTIMIZATION: channels_last memory format for conv layers (+10-20% speed)
    try:
        model = model.to(memory_format=torch.channels_last)
        print("🚀 [Optimization] Enabled channels_last memory format (+10-20% speed)")
    except Exception as e:
        print(f"   ⚠️  Could not enable channels_last: {e}")
    
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"[Model] Total params: {total_params:.1f}M")
    print(f"[Model] Trainable params: {trainable_params:.1f}M ({trainable_params/total_params*100:.1f}%)")
    
    # ========================================================================
    # OPTIMIZER
    # ========================================================================
    
    # ── Optimizer: Flat LR (proved best empirically) ────────────────────────
    # LESSON LEARNED: Differential LR bị loại bỏ vì thực nghiệm cho thấy nó
    # gây hại thay vì giúp ích:
    #
    #   text2vision  flat LR cũ → 66.7%  |  differential LR mới → 60.58%  (-6.1%)
    #   bidirectional flat LR cũ → 66.9% |  differential LR mới → 66.14%  (-0.76%)
    #   bidirectional ratio 3x           → 60.38%  (-5.76% thêm)
    #
    # Nguyên nhân:
    #   1. LoRA LR 5x (1e-4) QUÁ CAO cho decoder LoRA → kéo BARTpho khỏi
    #      pretrained distribution, giảm generation quality
    #   2. Encoder LR 0.5x (1e-5) QUÁ THẤP → text encoder gần như không học,
    #      mất đi khả năng hiểu tiếng Việt
    #   3. Gate init=0 + new-init 5x tạo mâu thuẫn: gate không đóng góp gì
    #      trong khi decoder LoRA lại học nhanh hơn → text shortcut thắng
    #
    # Kết luận: Flat LR đơn giản + gate init fix (0.5) là đủ.
    # ------------------------------------------------------------------
    gate_named = [(n, p) for n, p in model.named_parameters()
                  if p.requires_grad and 'vision_gating' in n]
    other_named = [(n, p) for n, p in model.named_parameters()
                   if p.requires_grad and 'vision_gating' not in n]
    gate_lr = learning_rate * args.gate_lr_multiplier
    gate_wd = 0.0 if args.gate_no_weight_decay else weight_decay
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Optimizer] Base LR = {learning_rate:.2e} | trainable params: {total_trainable/1e6:.2f}M")
    if args.gate_lr_multiplier != 1.0 or args.gate_no_weight_decay:
        print(f"[Optimizer] Gate group ({len(gate_named)} tensors): "
              f"LR={gate_lr:.2e} (×{args.gate_lr_multiplier}), WD={gate_wd}")

    optimizer = torch.optim.AdamW(
        [
            {'params': [p for _, p in other_named], 'lr': learning_rate, 'weight_decay': weight_decay},
            {'params': [p for _, p in gate_named],  'lr': gate_lr,       'weight_decay': gate_wd},
        ],
        eps=args.adam_eps,
        betas=(args.adam_beta1, args.adam_beta2),
    )
    print(f"[Optimizer] AdamW eps={args.adam_eps:.0e}, betas=({args.adam_beta1}, {args.adam_beta2})")
    
    # Mixed precision: float16 + GradScaler (proven reproducible)
    amp_dtype = amp_dtype if use_amp else torch.float32

    if use_amp:
        try:
            from torch.amp import GradScaler as NewGradScaler
            scaler = NewGradScaler(
                'cuda',
                init_scale=args.scaler_init_scale,
                growth_factor=args.scaler_growth_factor,
                backoff_factor=args.scaler_backoff_factor,
                growth_interval=args.scaler_growth_interval,
            )
        except (ImportError, AttributeError):
            scaler = GradScaler(
                init_scale=args.scaler_init_scale,
                growth_factor=args.scaler_growth_factor,
                backoff_factor=args.scaler_backoff_factor,
                growth_interval=args.scaler_growth_interval,
            )
        print(f"[AMP] {args.amp_dtype} + GradScaler (init={args.scaler_init_scale}, growth={args.scaler_growth_factor})")
    else:
        scaler = None
    
    # 🔥 LR Scheduler
    scheduler = None
    if args.scheduler == 'plateau':
        plateau_mode = 'min' if args.early_stopping_metric == 'loss' else 'max'
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode=plateau_mode,
            factor=args.scheduler_factor,
            patience=args.scheduler_patience,
            min_lr=1e-7
        )
        print(f"[Scheduler] ReduceLROnPlateau (mode={plateau_mode}, monitor={args.early_stopping_metric}, patience={args.scheduler_patience}, factor={args.scheduler_factor})")
    elif args.scheduler == 'cosine':
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
        if args.warmup_epochs > 0:
            warmup_sched = LinearLR(
                optimizer, start_factor=0.1, end_factor=1.0,
                total_iters=args.warmup_epochs
            )
            cosine_sched = CosineAnnealingLR(
                optimizer, T_max=max(1, args.epochs - args.warmup_epochs), eta_min=1e-6
            )
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_sched, cosine_sched],
                milestones=[args.warmup_epochs]
            )
            print(f"[Scheduler] Warmup({args.warmup_epochs}ep) + CosineAnnealingLR (T_max={args.epochs - args.warmup_epochs})")
        else:
            scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
            print(f"[Scheduler] CosineAnnealingLR (T_max={args.epochs})")
    elif args.scheduler == 'cosine_restart':
        from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR
        sgdr = CosineAnnealingWarmRestarts(optimizer, T_0=args.sgdr_t0, T_mult=1, eta_min=1e-6)
        if args.warmup_epochs > 0:
            warmup_sched = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=args.warmup_epochs)
            scheduler = SequentialLR(optimizer, schedulers=[warmup_sched, sgdr], milestones=[args.warmup_epochs])
            print(f"[Scheduler] Warmup({args.warmup_epochs}ep) + SGDR (T_0={args.sgdr_t0}, restarts every {args.sgdr_t0} ep)")
        else:
            scheduler = sgdr
            print(f"[Scheduler] SGDR CosineAnnealingWarmRestarts (T_0={args.sgdr_t0})")
    else:
        print(f"[Scheduler] None (fixed LR)")
    
    # 🔥 Early Stopping
    early_stopping = None
    if args.early_stopping:
        es_mode = 'min' if args.early_stopping_metric == 'loss' else 'max'
        early_stopping = EarlyStopping(
            patience=args.early_stopping_patience,
            min_delta=0.01,   # 0.01% EM — tránh floating point false-negative
            verbose=True,
            mode=es_mode
        )
        print(f"[Early Stopping] Enabled (patience={args.early_stopping_patience}, metric={args.early_stopping_metric}, mode={es_mode})")
    
    # Resume from checkpoint if specified
    start_epoch = 1
    best_val_loss = float('inf')
    # best_monitor: giá trị tốt nhất của metric đang track (EM, F1, loss, ...)
    # Dùng -inf cho max-metrics (em/f1/rouge), +inf cho loss
    _es_metric = args.early_stopping_metric  # shorthand
    best_monitor = 0.0 if _es_metric != 'loss' else float('inf')
    
    # Initialize accumulated NaN counters (will be restored from checkpoint if resuming)
    total_nan_loss_steps = 0
    total_nan_grad_steps = 0

    if args.resume:
        print(f"\n[Resume] Loading checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))

        # ✅ Restore best_monitor (EM/F1/loss) từ checkpoint để tránh ghi đè best_model sai
        best_monitor = checkpoint.get('best_monitor', best_monitor)
        args._best_monitor = best_monitor   # sync với vòng lặp training
        print(f"[Resume] best_monitor ({_es_metric}) restored: {best_monitor:.4f}")

        if scaler and 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        if scheduler and 'scheduler_state_dict' in checkpoint and not args.reset_lr:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        # ✅ reset_lr: flat LR (single group), chỉ cần set 1 giá trị
        if args.reset_lr:
            for pg in optimizer.param_groups:
                pg['lr'] = learning_rate
            print(f"[Resume] LR reset → {learning_rate:.2e} (flat, all params)")
        
        # ✅ Restore early stopping state — phục hồi cả counter LẪN best_score
        if early_stopping is not None and 'early_stopping_state' in checkpoint:
            es_state = checkpoint['early_stopping_state']
            early_stopping.counter = es_state['counter']
            early_stopping.best_score = es_state.get('best_score', best_monitor)  # ← dùng best_score
            early_stopping.best_loss = early_stopping.best_score   # mirror field cũ
            early_stopping.early_stop = es_state['early_stop']
            print(f"[Resume] Early stopping restored: counter={early_stopping.counter}/{early_stopping.patience}, best_score={early_stopping.best_score:.4f}")

        # Optional: reset epoch/ES state while keeping weights
        if args.resume_reset_epoch:
            start_epoch = 1
            best_val_loss = float('inf')
            best_monitor = 0.0 if _es_metric != 'loss' else float('inf')
            args._best_monitor = best_monitor
            if early_stopping is not None:
                early_stopping.counter = 0
                early_stopping.best_score = best_monitor
                early_stopping.best_loss = best_monitor
                early_stopping.early_stop = False
            print("[Resume] Epoch/early-stopping state reset (resume_reset_epoch enabled)")
        print(f"[Resume] Resuming from epoch {start_epoch}, best val loss: {best_val_loss:.4f}")
        # Restore accumulated NaN counters if present in checkpoint
        total_nan_loss_steps = checkpoint.get('total_nan_loss_steps', 0)
        total_nan_grad_steps = checkpoint.get('total_nan_grad_steps', 0)
    
    # ========================================================================
    # STAGE 3: END-TO-END TRAINING (NO STAGES 1/2!)
    # ========================================================================
    
    print("\n" + "="*80)
    print("STAGE 3: END-TO-END TRAINING (No Latent/KL Warmup)")
    print("="*80)
    print(f"  • Epochs: {stage3_epochs} (starting from {start_epoch})")
    print(f"  • Learning rate: {learning_rate}")
    print(f"  • Focus: Direct optimization for accuracy")
    print("="*80)
    
    print("\n" + "🚀"*40)
    print("SPEED OPTIMIZATIONS ENABLED:")
    print("  ✅ DataLoader: 4 workers + persistent_workers + prefetch_factor=2")
    print("  ✅ channels_last: Memory format optimization for conv layers")
    print("  📈 Expected speedup: ~40% faster training!")
    print("🚀"*40 + "\n")
    
    stage = 3
    
    # Vision dropout rate — explicit override or auto from fusion_type
    if args.vision_dropout_rate is not None:
        _vd_rate = args.vision_dropout_rate
        print(f"[Vision Dropout] rate={_vd_rate:.2f} (manual override via --vision_dropout_rate)")
    else:
        _vision_dropout = {'text2vision': 0.10, 'vision2text': 0.10, 'bidirectional': 0.05}
        _vd_rate = _vision_dropout.get(args.fusion_type, 0.10)
        print(f"[Vision Dropout] rate={_vd_rate:.2f} for fusion_type={args.fusion_type} (auto)")

    # Training history for plots and CSV
    training_history = []
    
    for epoch in range(start_epoch, stage3_epochs + 1):
        # H3: text-only warmup — decoder pre-warms on answer patterns before vision
        prev_text_only = getattr(model, 'text_only_mode', False)
        model.text_only_mode = (epoch <= args.text_only_warmup_epochs)
        if prev_text_only and not model.text_only_mode:
            print(f"[TextWarmup] E{epoch}: switching to full multimodal training")
        elif model.text_only_mode:
            print(f"[TextWarmup] E{epoch}/{args.text_only_warmup_epochs}: text-only mode (zero vision)")

        print(f"\n[Stage 3 | Epoch {epoch}/{stage3_epochs}]")

        # Update curriculum pacing before creating DataLoader iterator
        if curriculum_sampler is not None:
            curriculum_sampler.set_epoch(epoch, stage3_epochs)
            _active = len(curriculum_sampler)
            print(f"  [Curriculum] Active samples: {_active}/{len(train_dataset)} "
                  f"({_active/len(train_dataset):.1%})")

        # Training
        train_metrics = run_one_epoch_deterministic(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            is_training=True,
            max_norm=max_norm,
            stage=stage,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            answer_weights=answer_weights_tensor,
            use_type_loss=args.use_type_loss,
            vision_dropout_rate=_vd_rate,
            text_dropout_rate=args.text_dropout_rate,
            use_scst=args.use_scst,
            scst_start_epoch=args.scst_start_epoch,
            scst_lambda=args.scst_lambda,
            scst_sample_temp=args.scst_sample_temp,
            current_epoch=epoch,
            amp_dtype=amp_dtype,
            log_file=train_log_file,
            type_sample_weight_cfg=type_sample_weight_cfg,
            use_rdrop=args.use_rdrop,
            rdrop_alpha=args.rdrop_alpha,
            use_cdw_ce=args.use_cdw_ce,
            cdw_lambda=args.cdw_lambda,
            cdw_ordinal_weights=cdw_ordinal_weights,
            ema_model=ema_model,
            ema_decay=args.ema_decay,
        )

        print(f"  TRAIN -> Loss: {train_metrics['loss']:.4f} | Answer: {train_metrics['answer_loss']:.4f}")
        print(f"  TRAIN -> NaN loss steps: {train_metrics['nan_loss_steps']} | NaN grad steps: {train_metrics['nan_grad_steps']}")
        total_nan_loss_steps += train_metrics['nan_loss_steps']
        total_nan_grad_steps += train_metrics['nan_grad_steps']
        
        # Validation
        val_metrics = run_one_epoch_deterministic(
            model=model,
            dataloader=val_loader,
            optimizer=None,
            scaler=None,
            device=device,
            is_training=False,
            stage=stage,
            answer_weights=answer_weights_tensor,
            use_type_loss=args.use_type_loss,
            vision_dropout_rate=0.0,
            amp_dtype=amp_dtype,
            current_epoch=epoch,
            log_file=train_log_file,
        )

        print(f"  VAL   -> Loss: {val_metrics['loss']:.4f} | Answer: {val_metrics['answer_loss']:.4f}")

        # 🔥 Tính EM/F1 trên TOÀN BỘ val set mỗi epoch (dùng EMA model nếu có)
        _eval_model = ema_model if ema_model is not None else model
        full_val = evaluate_full_val(_eval_model, val_loader, model.tokenizer, device)
        print(f"  VAL   -> EM={full_val['exact_match']:.2f}% | F1={full_val['f1_score']:.2f}% | ROUGE-1={full_val['rouge1']:.2f}% | ROUGE-L={full_val['rougeL']:.2f}%")
        _type_keys = [k for k in full_val if k.startswith('em_')]
        if _type_keys:
            _type_str = ' | '.join(f"{k[3:]}={full_val[k]:.1f}%" for k in sorted(_type_keys))
            print(f"  VAL   -> per-type: {_type_str}")

        # Epoch summary to log file
        if train_log_file is not None:
            _per_type = ' | '.join(f"{k[3:]}={full_val[k]:.1f}%" for k in sorted(_type_keys)) if _type_keys else ''
            train_log_file.write(
                f"\n[EPOCH {epoch:02d} SUMMARY] "
                f"train_loss={train_metrics['loss']:.4f} val_loss={val_metrics['loss']:.4f} "
                f"EM={full_val['exact_match']:.2f}% F1={full_val['f1_score']:.2f}% "
                f"macro_em={full_val.get('macro_em', 0.0):.2f}% "
                f"lr={optimizer.param_groups[0]['lr']:.2e}"
                + (f" | {_per_type}" if _per_type else "") + "\n\n"
            )
            train_log_file.flush()

        # Track metrics in history
        epoch_metrics = {
            'epoch': epoch,
            'train_loss': train_metrics['loss'],
            'train_answer_loss': train_metrics['answer_loss'],
            'val_loss': val_metrics['loss'],
            'val_answer_loss': val_metrics['answer_loss'],
            'exact_match': full_val['exact_match'],
            'f1_score': full_val['f1_score'],
            'rouge1': full_val['rouge1'],
            'rougeL': full_val['rougeL'],
            'em_object':   full_val.get('em_object',   0.0),
            'em_counting': full_val.get('em_counting', 0.0),
            'em_color':    full_val.get('em_color',    0.0),
            'em_location': full_val.get('em_location', 0.0),
            'macro_em':    full_val.get('macro_em',    0.0),
            'learning_rate': optimizer.param_groups[0]['lr']
        }
        
        # 🔥 Add gate penalty to metrics if available
        if 'gate_penalty' in train_metrics:
            epoch_metrics['train_gate_penalty'] = train_metrics['gate_penalty']
        
        # 🔥 Log to W&B
        if args.use_wandb:
            wandb_log = {
                'epoch': epoch,
                'train/loss': train_metrics['loss'],
                'train/answer_loss': train_metrics['answer_loss'],
                'val/loss': val_metrics['loss'],
                'val/answer_loss': val_metrics['answer_loss'],
                'learning_rate': optimizer.param_groups[0]['lr']
            }
            
            # 🔥 Add gate penalty to W&B if available
            if 'gate_penalty' in train_metrics:
                wandb_log['train/gate_penalty'] = train_metrics['gate_penalty']

            # 🔥 Log full val metrics to W&B
            wandb_log.update({
                'val/exact_match': full_val['exact_match'],
                'val/f1_score':    full_val['f1_score'],
                'val/rouge1':      full_val['rouge1'],
                'val/rougeL':      full_val['rougeL'],
            })

        # 🔥 LR Scheduler step
        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                if args.early_stopping_metric == 'loss':
                    scheduler_metric = val_metrics['loss']
                else:
                    scheduler_key_map = {
                        'em': 'exact_match',
                        'f1': 'f1_score',
                        'rouge1': 'rouge1',
                        'rougeL': 'rougeL',
                        'macro_em': 'macro_em',
                    }
                    scheduler_metric = full_val[scheduler_key_map[args.early_stopping_metric]]
                scheduler.step(scheduler_metric)
            else:
                scheduler.step()
            
            # Print current LR
            print(f"  📊 Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")

        # 🔥 SWA: collect snapshot and switch to flat SWA LR
        if swa_model is not None and epoch >= args.swa_start_epoch:
            swa_model.update_parameters(model)
            swa_n += 1
            for pg in optimizer.param_groups:
                pg['lr'] = args.swa_lr
            print(f"  [SWA] Snapshot #{swa_n} collected (epoch={epoch}, swa_lr={args.swa_lr:.1e})")

        # Sample predictions every N epochs (chỉ để xem ví dụ, không dùng cho best/ES)
        if epoch % args.sample_every == 0:
            print("\n  [Sample Predictions (qualitative)]")
            samples, _ = sample_predictions(
                model, val_loader, model.tokenizer, device, num_samples=5, compute_metrics=False
            )
            for i, s in enumerate(samples, 1):
                em_symbol = "✓" if _normalize_vn(s['prediction']) == _normalize_vn(s['ground_truth']) else "✗"
                print(f"    {i}. {em_symbol} Q: {s['question']}")
                print(f"       Pred: {s['prediction']} | GT: {s['ground_truth']}")
        
        # 🔥 Send W&B log
        if args.use_wandb:
            wandb.log(wandb_log)
        
        # Add to training history
        training_history.append(epoch_metrics)

        # ── Chọn giá trị để so sánh best / early stopping ────────────────
        metric_key = args.early_stopping_metric   # 'loss', 'em', 'f1', 'rouge1', 'rougeL'
        if metric_key == 'loss':
            monitor_val  = val_metrics['loss']
            is_better    = lambda v, best: v < best
            best_monitor = best_val_loss
        else:
            key_map = {'em': 'exact_match', 'f1': 'f1_score',
                       'rouge1': 'rouge1', 'rougeL': 'rougeL',
                       'macro_em': 'macro_em'}
            monitor_val  = full_val[key_map[metric_key]]
            is_better    = lambda v, best: v > best
            best_monitor = getattr(args, '_best_monitor', 0.0)        # Save best model checkpoint (BEFORE early stopping check)
        is_best = is_better(monitor_val, best_monitor)
        if is_best:
            if metric_key == 'loss':
                best_val_loss = monitor_val
                best_monitor  = monitor_val
            else:
                args._best_monitor = monitor_val   # store on args for persistence
                best_monitor  = monitor_val
            print(f"  ✅ NEW BEST ({metric_key}={monitor_val:.4f})! Saving checkpoint...")

            # 🔥 Save EMA weights if available, else instantaneous weights
            _save_state = ema_model.state_dict() if ema_model is not None else model.state_dict()
            checkpoint = {
                'epoch': epoch,
                'stage': stage,
                'model_state_dict': _save_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_metrics['loss'],
                'val_loss': val_metrics['loss'],
                'best_val_loss': best_val_loss,
                'best_monitor': best_monitor,   # ✅ lưu để resume đúng
                'total_nan_loss_steps': total_nan_loss_steps,
                'total_nan_grad_steps': total_nan_grad_steps,
                'args': vars(args)
            }
            
            # ✅ Lưu early stopping state vào best_model.pt để resume từ best cũng đúng
            if early_stopping is not None:
                checkpoint['early_stopping_state'] = {
                    'counter': early_stopping.counter,
                    'best_score': early_stopping.best_score,
                    'best_loss': early_stopping.best_score,
                    'early_stop': early_stopping.early_stop,
                }

            if scaler is not None:
                checkpoint['scaler_state_dict'] = scaler.state_dict()
            
            if scheduler is not None:
                checkpoint['scheduler_state_dict'] = scheduler.state_dict()
            
            best_path = os.path.join(output_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            print(f"  💾 Saved to: {best_path}")
            if train_log_file is not None:
                train_log_file.write(f"[BEST] E{epoch:02d} saved → {best_path} ({metric_key}={monitor_val:.4f})\n")
                train_log_file.flush()
        
        # 🔥 ALWAYS save last model (for resume)
        last_checkpoint = {
            'epoch': epoch,
            'stage': stage,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_metrics['loss'],
            'val_loss': val_metrics['loss'],
            'best_val_loss': best_val_loss,
            'best_monitor': best_monitor,       # ✅ lưu để resume đúng
            'total_nan_loss_steps': total_nan_loss_steps,
            'total_nan_grad_steps': total_nan_grad_steps,
            'training_history': training_history,  # Include history for resume
            'args': vars(args)
        }
        
        # ✅ Save early stopping state — lưu best_score (không phải best_loss cũ)
        if early_stopping is not None:
            last_checkpoint['early_stopping_state'] = {
                'counter': early_stopping.counter,
                'best_score': early_stopping.best_score,   # ← field đúng
                'best_loss': early_stopping.best_score,    # ← backward compat
                'early_stop': early_stopping.early_stop,
            }
        
        if scaler is not None:
            last_checkpoint['scaler_state_dict'] = scaler.state_dict()
        
        if scheduler is not None:
            last_checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        last_path = os.path.join(output_dir, 'last_model.pt')
        try:
            torch.save(last_checkpoint, last_path)
            print(f"  💾 Saved last model to: {last_path} (for resume)")
        except OSError as e:
            print(f"  ⚠️  Failed to save last model: {e}")
        
        # 🔥 Save training curves and CSV after each epoch
        try:
            plot_training_curves(training_history, output_dir)
            save_metrics_csv(training_history, output_dir)
        except Exception as e:
            print(f"  ⚠️  Failed to save plots/CSV: {e}")
        
        # 🔥 Early stopping check (AFTER saving best/last model)
        if early_stopping is not None:
            if early_stopping(monitor_val):
                print(f"\n🛑 Early stopping at epoch {epoch}! ({metric_key}={monitor_val:.4f})")
                break
    
    # 🔥 SWA finalization: update BN stats, eval, save if better than best
    if swa_model is not None and swa_n > 0:
        print(f"\n[SWA] Finalizing averaged model ({swa_n} snapshots)...")
        from torch.optim.swa_utils import update_bn
        try:
            update_bn(train_loader, swa_model, device=device)
            print("[SWA] BatchNorm stats updated.")
        except Exception as _e:
            print(f"[SWA] BN update skipped (no BN layers or error): {_e}")
        swa_val = evaluate_full_val(swa_model.module, val_loader, model.tokenizer, device)
        swa_em = swa_val['exact_match']
        print(f"[SWA] Val EM={swa_em:.2f}%  (best so far={best_monitor:.4f}%)")
        _type_keys = [k for k in swa_val if k.startswith('em_')]
        if _type_keys:
            print("  " + " | ".join(f"{k[3:]}={swa_val[k]:.1f}%" for k in sorted(_type_keys)))
        if swa_em > best_monitor:
            print(f"[SWA] NEW BEST! {best_monitor:.2f}% → {swa_em:.2f}%. Saving SWA checkpoint...")
            swa_ckpt = {
                'epoch': 'swa',
                'stage': 3,
                'model_state_dict': swa_model.module.state_dict(),
                'best_monitor': swa_em,
                'args': vars(args),
            }
            torch.save(swa_ckpt, os.path.join(output_dir, 'best_model.pt'))
            print(f"[SWA] Saved to {output_dir}/best_model.pt")
        else:
            print(f"[SWA] Did not improve over best ({best_monitor:.2f}%). Keeping original best_model.pt.")

    if train_log_file is not None:
        train_log_file.write(f"\n[DONE] Training complete. best_val_loss={best_val_loss:.4f} output={output_dir}\n")
        train_log_file.close()

    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"  Best validation loss: {best_val_loss:.4f}")
    print(f"  Total NaN loss steps: {total_nan_loss_steps}")
    print(f"  Total NaN grad steps: {total_nan_grad_steps}")
    print(f"  Checkpoints saved to: {output_dir}")
    print("="*80)
    
    # 🔥 Finish W&B run
    if args.use_wandb:
        wandb.finish()
        print("\n[W&B] Run finished and synced!")


if __name__ == '__main__':
    main()
