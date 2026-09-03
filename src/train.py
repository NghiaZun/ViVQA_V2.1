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
import pickle
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

# ─────────────────────────────────────────────────────────────────────────────
# 🔬 OGM-GE (Peng et al., CVPR 2022 Oral) — CHUYEN TU CAP MODALITY SANG CAP MODULE.
#
# VAN DE (da do, B.9 trong THESIS_DRAFT_vi.md): GCA va TCVG GIANH VIEC nhau. GCA giam loss
# nhanh hon nen chiem gradient; TCVG lui ve anh xa dong nhat (alpha 0.9995 o LOCATION,
# 0.9989 o OBJECT). Tat GCA luc suy luan thi TCVG dang gia +1.64 EM (duong 3/3 seed)
# => nang luc CO THAT nhung bi che. Trong literature day la "modality competition" /
# "greedy learner" / "modality laziness", chi khac o cho no dien ra giua hai MODULE thay vi
# giua hai modality (BalanceBenchmark, arXiv 2502.10816, ghi nhan truong hop module-level
# con it duoc nghien cuu). ACM MM 2025 (10.1145/3746027.3754820) con CHUNG MINH rang fusion
# DONG — dung cai ma TCVG la — LAM GREEDY NANG THEM.
#
# KHAC BIET SO VOI --gca_dropout DA CHAY (ket qua -0.35, COLOR -4.48):
#   gca_dropout doi FORWARD PASS (tat GCA ngau nhien) -> doi ham ma model tinh -> gate tu
#     tai phan bo de bu, COLOR sap.
#   OGM-GE GIU NGUYEN FORWARD TUYET DOI. Moi buoc model van tinh dung mot ham. Chi
#     TOC DO HOC cua nhanh dang thang bi ham lai.
# Day la truc DUY NHAT chua dung: ca ~14 can thiep truoc deu sua forward.
#
# TAI SAO TRAN ORACLE-ALPHA KHONG CHAN HUONG NAY (diem phai tu kiem nghiem nhat):
#   Tran do (per-type dang -0.07, per-sample khong nhan dang duoc, AUC 0.53) duoc do tren
#   BIEU DIEN CO DINH — no tra loi "voi feature NAY, alpha tot nhat dang bao nhieu".
#   OGM-GE khong co cai thien alpha. No doi BIEU DIEN ma nhanh TCVG hoc duoc. Cac tran
#   da do KHONG rang buoc duoc dieu do.
#
# RUI RO da biet: B.9 van co the can — gate co the tu tai phan bo qua qua trinh train du
#   forward giu nguyen, va COLOR sap lai. Do la CAU HOI, khong phai loi cai dat.
# ─────────────────────────────────────────────────────────────────────────────

def _ogm_gold_score(logits, labels):
    """Xac suat trung binh model gan cho token GOLD — ban seq2seq cua `score` trong OGM-GE.

    Ban goc (phan loai 1 nhan): score = sum_b softmax(out_b)[label_b].
    O day dap an la MOT CHUOI token nen lay trung binh tren cac vi tri hop le (labels != -100).
    Trung binh chu khong phai tong: batch nay 12 mau, do dai dap an khac nhau giua cac batch,
    tong se lam ty so phu thuoc do dai chuoi thay vi phu thuoc nang luc nhanh.
    """
    v = (labels != -100)
    if v.sum() == 0:
        return None
    lp = F.log_softmax(logits.float(), dim=-1)
    g = lp.gather(-1, labels.clamp(min=0).unsqueeze(-1)).squeeze(-1).exp()   # [B, T]
    return ((g * v).sum() / v.sum()).item()


def _build_token_prior(train_csv, tokenizer, max_len=10):
    """p_max(token | loai, vi tri) uoc luong tu chinh tap train.

    Do duoc tren ViVQA: MOI dap an COLOR bat dau bang token 'mau' (100%), va COLOR chi co
    2.36 token/dap an -> 42% ngan sach loss cua COLOR tieu cho mot token KHONG BAO GIO SAI.
    Tren toan train, 9.93% tong so token dap an la token 'mau' mien phi do.
    Gradient o do bang khong ve mat thong tin nhung van chiem cho trong chuan hoa CE, nen no
    PHA LOANG tin hieu hoc — ke ca tin hieu chay ve gate, va COLOR la loai DUY NHAT gate co
    gain co y nghia (+0.94, p=0.0053, 9/10 seed).

    Bang duoc uoc luong TU DU LIEU, khong hardcode token nao, nen tu dong dung tren bo khac.
    """
    import pandas as _pd, unicodedata as _ud, collections as _c
    _n = lambda x: _ud.normalize('NFC', str(x)).strip().lower()
    df = _pd.read_csv(train_csv)
    cnt = _c.defaultdict(_c.Counter)
    for t, a in zip(df['type'].values, df['answer'].map(_n).values):
        ids = tokenizer(a, add_special_tokens=False).input_ids[:max_len]
        for i, tid in enumerate(ids):
            cnt[(int(t), i)][tid] += 1
    pmax = {}
    for k, c in cnt.items():
        tot = sum(c.values())
        pmax[k] = c.most_common(1)[0][1] / tot if tot else 0.0
    return pmax


def _gge_token_weights(model, fwd_kwargs, labels, floor=0.0):
    """Trong so GGE tung vi tri nhan: w = clamp(1 - p_GCA(gold), 0, 1), nang len san `floor`.

    p_GCA = xac suat nhanh GCA DON DOC (ep alpha = 1, TCVG thanh anh xa dong nhat) gan cho token
    gold. Mau nao GCA da giai duoc -> w ~ 0 -> khong con dong gop gradient; mau nao GCA thua ->
    w ~ 1. Tuc CE bi ep chi hoi ve PHAN DU ma GCA khong giai thich duoc.

    Day la tin hieu PER-SAMPLE dau tien trong toan bo chuoi can thiep: gca_dropout va OGM-GE deu
    chi noi "gate hoat dong di" ma khong noi O DAU. Do dac noi 32% phuong sai alpha oracle nam o
    chieu per-sample va dang +3.50 den +11.27.

    Khong can tinh nhan dang duoc luc suy luan (AUC 0.53): day la nan trong so luc TRAIN bang gold,
    khong phai quy tac suy luan.
    """
    vg = getattr(model, 'vision_gating', None)
    if vg is None:
        return None
    _prev = getattr(vg, 'alpha_override', None)
    try:
        vg.alpha_override = torch.ones(labels.size(0), 1, device=labels.device)
        with torch.no_grad():
            lg = model(**fwd_kwargs).answer_logits
    finally:
        vg.alpha_override = _prev
    p = F.log_softmax(lg.float(), dim=-1).gather(
        -1, labels.clamp(min=0).unsqueeze(-1)).squeeze(-1).exp()
    w = (1.0 - p).clamp(min=float(floor), max=1.0)
    # Vi tri bi bo qua duoc model nhan 0 qua mat na hop le; dat 1.0 o day cho sach ve so hoc.
    return torch.where(labels != -100, w, torch.ones_like(w)).detach()


def _gge_residual_stats(model, fwd_kwargs, labels):
    """CHAN DOAN cho huong GGE (Han et al., ICCV 2021 Oral) TRUOC KHI cai dat.

    GGE huan luyen mo hinh chinh tren PHAN DU cua mo hinh da giai duoc phan de:
        y_gradient = clamp(one_hot(gold) - p_bias(gold), 0, 1)
        loss       = -(log_softmax(logits) * y_gradient).sum()
    O day "mo hinh bias" = nhanh GCA don doc (ep alpha = 1). Trong so hieu dung moi mau
    la (1 - p_GCA(gold)).

    RUI RO CHET DA BIET: gate_distill da chet vi nhan rong — mo hinh dat 99.77% tren TRAIN
    nen alpha oracle suy nguoc ra gan nhu khong co tin hieu. Neu p_GCA(gold) tren TRAIN cung
    bao hoa ve 1 thi y_gradient ~ 0 deu, va GGE chi con la ha learning rate = null tu gay ra.

    Cai QUYET DINH khong phai do lon tuyet doi ma la DO TAP TRUNG tuong doi: neu phan lon
    khoi luong loss don ve mot thieu so mau thi day la phep tai trong so that; neu trai deu
    thi khong phai.
    """
    vg = getattr(model, 'vision_gating', None)
    if vg is None:
        return None
    _prev_ov = getattr(vg, 'alpha_override', None)
    try:
        vg.alpha_override = torch.ones(labels.size(0), 1, device=labels.device)
        with torch.no_grad():
            lg = model(**fwd_kwargs).answer_logits
    finally:
        vg.alpha_override = _prev_ov
    v = (labels != -100)
    if v.sum() == 0:
        return None
    p = F.log_softmax(lg.float(), dim=-1).gather(
        -1, labels.clamp(min=0).unsqueeze(-1)).squeeze(-1).exp()      # p_GCA(gold) [B,T]
    w = (1.0 - p)[v]                                                  # trong so GGE
    ws, _ = torch.sort(w, descending=True)
    n = ws.numel()
    tot = ws.sum().clamp(min=1e-12)
    return {
        'w_mean': w.mean().item(),
        'w_med': ws[n // 2].item(),
        'top10_mass': (ws[:max(1, n // 10)].sum() / tot).item(),   # 0.10 = trai deu, ->1 = tap trung
        'frac_gt_05': (w > 0.5).float().mean().item(),
    }


def _ogm_branch_scores(model, fwd_kwargs, labels):
    """Do nang luc DON DOC cua hai nhanh, khong dung gradient.

    GCA don doc  : ep alpha = 1 mọi patch -> TCVG thanh anh xa dong nhat -> dung nhanh T0.
    TCVG don doc : gca_strength = 0       -> Flamingo khong con cong residual vao vision.

    BAY DA NE: DeterministicVQA.forward GHI DE `_fl.gca_strength` tu `self.gca_strength` o
    MOI lan goi. Set truc tiep tren tung fusion layer se bi xoa ngay trong forward do ->
    phai set o CAP MODEL. Da tung mat mot luot GPU vi dung cai bay hinh dang nay
    (blend_gamma sigmoid(-6), va gca_dropout set trong nhanh train).

    Giu model o che do train() va chay duoi no_grad: R-Drop trong file nay cung goi
    forward lan hai o che do train, nen day la duong da co tien le. Doi sang eval() se
    dung dropout va co the lat che do cua cac submodule dang co y de o eval.
    """
    vg = getattr(model, 'vision_gating', None)
    if vg is None or not hasattr(model, 'flamingo_fusion'):
        return None, None

    # ── nhanh GCA don doc: alpha := 1
    _prev_ov = getattr(vg, 'alpha_override', None)
    try:
        vg.alpha_override = torch.ones(labels.size(0), 1, device=labels.device)
        with torch.no_grad():
            s_gca = _ogm_gold_score(model(**fwd_kwargs).answer_logits, labels)
    finally:
        vg.alpha_override = _prev_ov

    # ── nhanh TCVG don doc: gca_strength := 0 (dat o CAP MODEL, xem ghi chu tren)
    _prev_gs = getattr(model, 'gca_strength', 1.0)
    try:
        model.gca_strength = 0.0
        with torch.no_grad():
            s_tcvg = _ogm_gold_score(model(**fwd_kwargs).answer_logits, labels)
    finally:
        model.gca_strength = _prev_gs

    return s_gca, s_tcvg


def _ogm_apply_grads(model, coeff, noise):
    """Nhan `coeff` vao gradient cua RIENG flamingo_fusion (GCA), roi cong nhieu Gauss (GE).

    Tra ve so tensor da dong vao (0 = khong lam gi) de kiem chung duoc tu ben ngoai.
    """
    if not hasattr(model, 'flamingo_fusion'):
        return 0
    if coeff >= 1.0 and noise <= 0:
        return 0
    n = 0
    for p in model.flamingo_fusion.parameters():
        if p.grad is None:
            continue
        if noise > 0 and p.grad.numel() > 1:
            _sd = p.grad.std()          # std cua grad GOC, dung thu tu code OGM-GE goc
            p.grad.mul_(coeff).add_(torch.randn_like(p.grad) * (_sd * noise))
        else:
            p.grad.mul_(coeff)
        n += 1
    return n


def _ogm_coeff(ratio, alpha):
    """coeff = 1 - tanh(alpha * relu(ratio)) — dung cong thuc code goc OGM-GE.

    Chi ap dung cho nhanh DANG THANG (ratio > 1); nhanh yeu giu he so 1.0.
    """
    import math
    return 1.0 - math.tanh(alpha * max(ratio, 0.0))


def run_one_epoch_deterministic(
    model, dataloader, optimizer, scaler, device,
    is_training=True, max_norm=1.0, stage=3, gradient_accumulation_steps=1,
    answer_weights=None, use_type_loss=False, gate_types_no_typeloss=False, vision_dropout_rate=0.10, gate_sparsity_lambda=0.0,
    kl_pretrained_lambda=0.0,
    token_prior_gamma=0.0, token_prior_table=None,
    gate_distill_alpha=None, gate_distill_lambda=0.0, gate_distill_mode='mse',
    slot_bind_lambda=0.0,
    text_dropout_rate=0.0,
    use_scst=False, scst_start_epoch=5, scst_lambda=0.1,
    scst_sample_temp=1.0, current_epoch=0, amp_dtype=torch.float16,
    log_file=None,
    type_sample_weight_cfg=None,  # dict with type_defaults + count_overrides for sample-level weighting
    use_rdrop=False, rdrop_alpha=0.1, rdrop_all_pos=False,  # 🔥 R-Drop: KL consistency between two dropout sub-models
    use_cdw_ce=False, cdw_lambda=0.1, cdw_ordinal_weights=None,  # 🔥 CDW-CE: ordinal penalty for COUNT
    hard_margin=0.0, hard_margin_m=1.0,  # 🔬 PHA B2: margin chong lang gieng (xem RESEARCH_PLAN_next.md)
    answer_cls_lambda=0.0, answer_cls_map=None,   # 🔬 dau phan loai tren tap dap an
    box_ground_lambda=0.0, box_count_lambda=0.0, box_count_typed=False,
    box_class_lambda=0.0,  # 🔬 CE 81 lop COCO tung patch (region_map mang GIA TRI LOP)
    ema_model=None, ema_decay=0.999,  # 🔥 EMA: exponential moving average of weights
    region_map_lookup=None,  # 🔬 dict {img_id: int16[num_patches]} tu build_patch_region_map.py, None = tat (hanh vi cu)
    ogm_ge=0.0, ogm_ge_noise=0.0, ogm_ge_every=1, ogm_ge_ema=0.9,   # 🔬 OGM-GE cap module (xem ghi chu tren)
    ogm_ge_start_epoch=0, ogm_ge_end_epoch=10**9, ogm_state=None,
    gge_diag=0,   # 🔬 CHI CHAN DOAN: moi k buoc, do phan bo trong so GGE (1 - p_GCA(gold)). 0 = tat.
    gge=0.0, gge_floor=0.0, gge_start_epoch=0,   # 🔬 GGE that su (doi loss)
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

    def _apply_ogm_to_grads():
        """Nhan he so vao gradient cua RIENG flamingo_fusion (GCA), roi cong nhieu Gauss (GE).

        VI TRI BAT BUOC: sau scaler.unscale_() va TRUOC clip_grad_norm_.
          - truoc unscale_: grad con dang bi nhan he so AMP -> std tinh ra sai bac do
          - sau clip:       norm da bi chuan hoa -> vua ham gradient xong lai bi clip keo len,
                            can thiep bi triet tieu ma khong bao loi. Day dung la kieu null
                            tu gay ra ma minh da dinh mot lan.
        """
        if ogm_ge <= 0 or ogm_state is None:
            return
        c = ogm_state.get('coeff', 1.0)
        if _ogm_apply_grads(model, c, ogm_ge_noise) > 0:
            ogm_state['coeff_sum'] = ogm_state.get('coeff_sum', 0.0) + c
            ogm_state['coeff_n'] = ogm_state.get('coeff_n', 0) + 1

    total_loss = 0.0
    total_answer_loss = 0.0
    total_type_loss = 0.0  # 🔥 NEW: Track type loss
    nan_loss_steps = 0
    nan_grad_steps = 0
    num_batches = 0
    grad_norm_sum = 0.0
    grad_norm_max = 0.0
    grad_norm_count = 0
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
            # 🔬 gate_types_no_typeloss (2026-08-30): nhan loai vao GATE ma KHONG co type_loss.
            #   O nay cua ma tran 2x2 truoc gio khong chay duoc: bo --use_type_loss lam
            #   question_types = None -> gate mat luon nhan loai, tuc bo MOT LUC hai thu.
            #   Phan ra do duoc: bo type_loss giup LOCATION (+0.72 p=0.023 9/10, Tblind),
            #   nhung lam gate mu thi HAI LOCATION (-0.80, 0/4, notype). Hai tac dong nguoc
            #   chieu, nen o "gate biet loai + khong type_loss" moi la o duoc du doan tot nhat.
            question_types = None
            if (use_type_loss or gate_types_no_typeloss) and 'question_type' in batch:
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

            # 🔬 region_map (COCO instance region tung patch, tcvg_spatial_blend): tra cuu theo
            # img_id trong batch. Anh khong co annotation (thieu trong lookup) -> tat ca patch
            # chung 1 region (0) = pool toan cuc, fallback an toan (khong crash, khong loi hanh vi).
            region_map = None
            if region_map_lookup is not None and 'img_id' in batch:
                _ids = batch['img_id'].tolist() if torch.is_tensor(batch['img_id']) else list(batch['img_id'])
                _rows = [region_map_lookup.get(int(_iid)) for _iid in _ids]
                _np_default = next((r.shape[0] for r in _rows if r is not None), None)
                if _np_default is not None:
                    _mat = np.stack([r if r is not None else np.zeros(_np_default, dtype=np.int16)
                                      for r in _rows], axis=0)
                    region_map = torch.from_numpy(_mat).long().to(device)

            # 🔬 GGE: trong so tung vi tri nhan tu PHAN DU cua nhanh GCA. Phai tinh TRUOC luot
            # truyen chinh vi no la dau vao cua loss. Tinh chat da kiem (verify_gge_identity.py):
            # trong so deu => TRUNG KHOP TUYET DOI baseline, va bieu thuc tu chuan hoa theo ty le
            # => che do that bai la "khong doi gi", khong phai "ha learning rate".
            _tw = None
            # 🔬 token_prior_gamma: ha trong so token doan duoc tu prior (bang uoc tu train csv)
            if is_training and token_prior_gamma > 0 and token_prior_table is not None:
                _tw = torch.ones_like(labels, dtype=torch.float32)
                _qt = question_types.detach().cpu().tolist() if question_types is not None else None
                for _b in range(labels.size(0)):
                    _t = int(_qt[_b]) if _qt is not None else -1
                    for _i in range(labels.size(1)):
                        if labels[_b, _i].item() == -100: continue
                        _p = token_prior_table.get((_t, _i))
                        if _p is not None:
                            _tw[_b, _i] = max(1.0 - token_prior_gamma * _p, 1e-3)
                _tw = _tw.to(labels.device)
            if is_training and gge > 0 and current_epoch >= gge_start_epoch:
                with autocast('cuda', enabled=_amp_on, dtype=amp_dtype):
                    _tw = _gge_token_weights(
                        model,
                        dict(pixel_values=pixel_values, input_ids=input_ids,
                             attention_mask=attention_mask, labels=labels, stage=stage,
                             answer_weights=answer_weights, question_types=question_types,
                             images_384=images_384, raw_questions=raw_questions,
                             sample_weights=sample_weights, region_map=region_map),
                        labels, gge_floor)
                if _tw is not None and ogm_state is not None:
                    _m = (labels != -100)
                    ogm_state['ggew_sum'] = ogm_state.get('ggew_sum', 0.0) + _tw[_m].mean().item()
                    ogm_state['ggew_n'] = ogm_state.get('ggew_n', 0) + 1

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
                    region_map=region_map,
                    token_weights=_tw,              # 🔬 GGE (None = hanh vi cu y nguyen)
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
                    if rdrop_all_pos:
                        # PHA B3. Ban goc chi lay answer_logits[:, 0, :] — DUY NHAT vi tri token dau.
                        #   Moi dap an COLOR deu bat dau bang token "mau", nen KL o vi tri 0 gan nhu
                        #   khong rang buoc gi cho COLOR (cung cai bay da lam hong luot do gan-dung
                        #   dau tien hom nay, va la ly do --hard_margin phai tinh theo tung vi tri).
                        #   Giu NGUYEN quy uoc detach cua ban goc de chi doi DUNG MOT bien.
                        _v = (labels != -100).float()                       # [B, T]
                        _lp1 = F.log_softmax(outputs.answer_logits.float(), dim=-1)
                        _lp2 = F.log_softmax(outputs2.answer_logits.float(), dim=-1)
                        _kl = 0.5 * (
                            (_lp1.exp() * (_lp1 - _lp2.detach())).sum(-1) +
                            (_lp2.exp() * (_lp2 - _lp1.detach())).sum(-1)
                        )                                                   # [B, T]
                        rdrop_kl = (_kl * _v).sum() / _v.sum().clamp(min=1)
                    else:
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

                # 🔬 PHA B2 — MARGIN CHONG LANG GIENG.
                #   Co so do luong (RESEARCH_PLAN_next.md muc 2.2 + 5): 98.8% loi la mot dap an
                #   HOP LE khac, va no cach gold DUNG MOT BUOC (COUNT lech 1 = 74.5%, COLOR mau ke,
                #   OBJECT/LOCATION sai cap phan cap). CE thuong day gold len so voi TOAN BO tu vung,
                #   khong he biet doi thu that su la ai. O day phat TRUC TIEP vao doi thu manh nhat
                #   ma chinh model chon, tai tung vi tri.
                #   VI SAO TINH THEO TUNG VI TRI chu khong chi vi tri 0: moi dap an COLOR deu bat dau
                #   bang token "mau", nen margin o vi tri 0 la vo nghia cho COLOR (cung la cai bay da
                #   lam hong lan do do gan-dung dau tien).
                if hard_margin > 0 and is_training:
                    _lg = outputs.answer_logits.float()                      # [B, T, V]
                    _valid = labels != -100                                  # [B, T]
                    if _valid.any():
                        _gi = labels.clamp(min=0).unsqueeze(-1)              # [B, T, 1]
                        _gold = _lg.gather(-1, _gi).squeeze(-1)              # [B, T]
                        _neg = _lg.scatter(-1, _gi, float('-inf')).max(-1).values   # doi thu manh nhat
                        _hinge = F.relu(hard_margin_m - (_gold - _neg)) * _valid
                        loss = loss + hard_margin * _hinge.sum() / _valid.sum().clamp(min=1)

                # 🔬 BOX-GROUNDED MULTI-TASK: box la NHAN, khong bao gio la input -> luc suy luan
                #   khong can annotation nao. Hai dau phu doc feature ma decoder that su doc.
                #   CHO DE SAI CHET NGUOI: mot hang region_map toan 0 co the la THIEU annotation
                #   (0.6% anh) chu khong phai "anh khong co vat the nao". Neu tinh BCE tren cac hang
                #   do thi dang day model rang nhung anh ay khong co vat the -> nhieu nhan. Vi vay
                #   CHI tinh loss tren cac anh CO it nhat mot ca the.
                # 🔬 CE cho DAU PHAN LOAI TREN TAP DAP AN.
                #   94.8% loi (759/801) la chon nham MOT DAP AN HOP LE KHAC; beam 1/3/5/10 va trie
                #   mo rong deu cho 73.31 -> tim kiem khong phai nut that, THU HANG moi sai.
                #   Sinh chuoi toi uu CE tung token; day toi uu THANG mot softmax tren 328 lop.
                if answer_cls_lambda > 0 and answer_cls_map is not None \
                        and getattr(outputs, 'answer_cls_logits', None) is not None and 'row_idx' in batch:
                    _acy = answer_cls_map[batch['row_idx']].to(outputs.answer_cls_logits.device)
                    _ok = _acy >= 0                     # -1 = dap an khong nam trong tap lop
                    if _ok.any():
                        loss = loss + answer_cls_lambda * F.cross_entropy(
                            outputs.answer_cls_logits[_ok], _acy[_ok])

                # 🔬 CE du doan LOP COCO tung patch — KHOI DOC LAP.
                #   Truoc day khoi nay bi nhet BEN TRONG khoi box_ground, ma dieu kien cua khoi do la
                #   (box_ground_lambda>0 or box_count_lambda>0) and last_ground_logit is not None.
                #   Voi arm chi bat --box_class_lambda thi ca hai lambda = 0 -> box_ground=False ->
                #   last_ground_logit khong ton tai -> CE lop KHONG BAO GIO chay (do duoc 2026-08-11:
                #   loss tong = ans + 0.2*type + 0.05*ctr dung khop, phan du = 0.000).
                #   Dung loai bug im lang da lam 5 co gate thanh no-op va lam nhanh box_g chet.
                #   Dung --tcvg_spatial_blend_region_map patch_class_map_flat.pkl (gia tri = ID LOP 0..80).
                if box_class_lambda > 0 and region_map is not None \
                        and getattr(model, 'last_class_logit', None) is not None:
                    _cl = model.last_class_logit                      # [B,P_model,n_class]
                    _rmc = region_map                                 # [B,196]
                    _offc = _cl.size(1) - _rmc.size(1)
                    if _offc > 0:
                        _cl = _cl[:, _offc:, :]                       # bo pooler token o dau
                    elif _offc < 0:
                        _rmc = _rmc[:, -_cl.size(1):]
                    _hasc = (_rmc > 0).any(dim=1)                     # anh CO annotation
                    if _hasc.any():
                        loss = loss + box_class_lambda * F.cross_entropy(
                            _cl[_hasc].reshape(-1, _cl.size(-1)),
                            _rmc[_hasc].reshape(-1).long())

                if (box_ground_lambda > 0 or box_count_lambda > 0) and region_map is not None \
                        and getattr(model, 'last_ground_logit', None) is not None:
                    _gl = model.last_ground_logit                     # [B,P_model]
                    _rmg = region_map                                 # [B,196]
                    _offg = _gl.size(1) - _rmg.size(1)
                    if _offg > 0:
                        _gl = _gl[:, _offg:]                          # bo pooler token o dau
                    elif _offg < 0:
                        _rmg = _rmg[:, -_gl.size(1):]
                    _hasann = (_rmg > 0).any(dim=1)                   # [B] anh CO annotation
                    # 🔬 SUA 2026-08-11 (do duoc tu run bgc): ap L1 count len MOI anh lam COLOR
                    #   sut -3.52 trong khi COUNT len +1.35. Nguyen nhan: count_head doc vector POOL
                    #   DUNG CHUNG, nen voi cau hoi mau model bi buoc nhet "anh co may ca the" vao
                    #   dung cai vector no can de doc mau -> nhieu thuan. Chi ap tren mau COUNT thi
                    #   can thiep thanh phau thuat (tu so la dap an DOC QUYEN cua COUNT: do duoc
                    #   1599/1599 dong COUNT, 0 o OBJECT/COLOR/LOCATION).
                    if box_count_typed and 'question_type' in batch:
                        _isc = torch.tensor([t.item() == 1 for t in batch['question_type']],
                                            dtype=torch.bool, device=_gl.device)
                        _hascnt = _hasann & _isc
                    else:
                        _hascnt = _hasann
                    if _hasann.any():
                        if box_ground_lambda > 0:
                            _tgtg = (_rmg > 0).to(_gl.dtype)
                            loss = loss + box_ground_lambda * F.binary_cross_entropy_with_logits(
                                _gl[_hasann], _tgtg[_hasann])
                        if box_count_lambda > 0 and _hascnt.any() \
                                and getattr(model, 'last_count_pred', None) is not None:
                            _ntgt = torch.tensor(
                                [torch.unique(_r[_r > 0]).numel() for _r in _rmg],
                                device=_gl.device, dtype=model.last_count_pred.dtype)
                            loss = loss + box_count_lambda * F.l1_loss(
                                model.last_count_pred[_hascnt], _ntgt[_hascnt])


                # 🔬 SLOT BINDING SUPERVISION (nhanh B, CHAN DOAN): day slot attention gan cac patch
                # THUOC CUNG MOT VAT THE COCO vao CUNG MOT slot. Muc dich duy nhat la tach bach hai
                # cach doc mot ket qua slot am:
                #   (a) co che cong (tao token moi) cung vo dung  <-- ket luan manh
                #   (b) co che dung nhung 10.8k mau + chi tin hieu CE khong du de hoc binding tu 0
                # Neu B thang ma A khong -> (b), va ban giu duoc lap luan "khong can bbox" bang cach
                # thay nhan COCO bang pseudo-region tu cum feature (nhanh D).
                # LUU Y HOC THUAT: nhan nay tu bbox COCO, CUNG NGUON voi nhanh Box cua cong trinh so
                # sanh. Dung B lam CHAN DOAN thi hop le; bao cao B nhu ket qua chinh thi KHONG duoc
                # noi "chung toi khong dung bbox".
                #
                # Cong thuc (BAT BIEN VOI HOAN VI slot -- khong can Hungarian matching):
                #   A[b,:,p] = phan bo gan cua patch p tren K slot (softmax theo K, tong=1)
                #   S[b,p,q] = <A[:,p], A[:,q]> / (|A[:,p]||A[:,q]|)  = do "cung slot" cua p va q
                #   M[b,p,q] = 1 neu p,q cung region COCO, 0 neu khac
                #   loss = MSE(S, M) tren cac cap KHONG phai ca-hai-la-nen
                # Khong can biet slot nao ung vat the nao -- chi can quan he cung/khac nhom, nen
                # loss nay bat bien voi moi hoan vi cua K slot. Do la diem quan trong cua thiet ke.
                if (slot_bind_lambda > 0 and loss is not None and region_map is not None
                        and getattr(model, 'slot_attn', None) is not None
                        and getattr(model.slot_attn, 'last_assign', None) is not None):
                    _A = model.slot_attn.last_assign                     # [B,K,P_model]
                    _rm = region_map                                     # [B,196]
                    # P_model co the = 197 (use_siglip_pooler chen 1 token toan cuc o dau).
                    # Token toan cuc KHONG nam tren luoi -> bo khoi phep do, khong gan nhan cho no.
                    _off = _A.size(2) - _rm.size(1)
                    if _off > 0:
                        _A = _A[:, :, _off:]
                    if _A.size(2) == _rm.size(1):
                        _An = F.normalize(_A, dim=1)                     # chuan hoa theo K
                        _S = torch.einsum('bkp,bkq->bpq', _An, _An)      # [B,P,P] in [0,1]
                        _M = (_rm.unsqueeze(2) == _rm.unsqueeze(1)).to(_S.dtype)
                        _fg = (_rm > 0)                                  # patch thuoc vat the
                        _w = (_fg.unsqueeze(2) | _fg.unsqueeze(1)).to(_S.dtype)   # bo cap nen-nen
                        _den = _w.sum().clamp(min=1)
                        loss = loss + slot_bind_lambda * ((_S - _M) ** 2 * _w).sum() / _den

                # 🔥 Phat thua thot len gate: buoc mo hinh phai chon patch nao dang giu
                if (gate_sparsity_lambda > 0 and loss is not None
                        and getattr(model, 'vision_gating', None) is not None
                        and getattr(model.vision_gating, 'last_alpha', None) is not None):
                    loss = loss + gate_sparsity_lambda * model.vision_gating.last_alpha.mean()

                # 🔬 KL-to-pretrained: chong SUP DO phan bo dau ra do fine-tune.
                #   Chuan hoa theo CE la BAT BUOC, khong phai lam dep. Do duoc
                #   (probe_kl_magnitude.py, 10 batch x 12):
                #       chua train : CE 15.4624 | KL 11.7676   -> lambda 0.5 = 27.6% tong loss
                #       da hoi tu  : CE  0.0878 | KL  8.4054   -> lambda 0.5 = 98.0% tong loss
                #   CE roi 176x trong khi KL gan nhu dung yen, nen MOT lambda CO DINH se truot tu
                #   ~28% len ~98% ty trong va nuot luon nhiem vu. Chuan hoa theo CE.detach()/KL.detach()
                #   giu ty trong KHONG DOI suot train, dung muc tieu 10-30% da dang ky truoc.
                _klp = getattr(outputs, 'kl_pretrained_loss', None)
                if (kl_pretrained_lambda > 0 and _klp is not None and loss is not None):
                    _ce = loss.detach()
                    _scale = (_ce / _klp.detach().clamp(min=1e-6)).clamp(max=1e3)
                    loss = loss + kl_pretrained_lambda * _scale * _klp

                # 🔬 GATE DISTILL (A3): ep alpha hoc duoc bam theo alpha ORACLE (alpha toi uu
                # per-mau tinh nguoc tu nhan gold bang eval.py --oracle_alpha, xem run_oracle_alpha.sh).
                # LY DO: E0 do duoc rang ho gate CO chua loi giai (tran tren >> T2) nhung LM loss
                # khong tim ra no -> van de la TIN HIEU HOC. Day la tin hieu do: mot muc tieu hoi
                # quy TRUC TIEP tren alpha, dac hon nhieu so voi gradient loang qua ca decoder.
                # Neu van khong len: alpha_oracle KHONG du doan duoc tu (v, q) -> tran tren do la
                # thong tin PHU THUOC DAP AN, khong gate hoc duoc nao voi toi -> dong TCVG that su.
                if (gate_distill_lambda > 0 and loss is not None and gate_distill_alpha is not None
                        and getattr(model, 'vision_gating', None) is not None
                        and getattr(model.vision_gating, 'last_alpha', None) is not None
                        and 'row_idx' in batch):
                    _ap = model.vision_gating.last_alpha
                    if _ap.dim() == 3:
                        _ap = _ap.mean(-1)
                    _tgt = gate_distill_alpha[batch['row_idx'].to(gate_distill_alpha.device)]
                    _tgt = _tgt.to(device=_ap.device, dtype=_ap.dtype)
                    if _tgt.size(1) == _ap.size(1):
                        if gate_distill_mode == 'tail':
                            # 🔬 Do dac alpha_oracle (analyze_oracle_alpha.py): 82% patch co
                            # target > 0.9 va chi 14% < 0.5 (trung vi 11/196 patch bi nen).
                            # MSE deu -> 82% khoi luong loss nam o phan "giu nguyen" TAM THUONG,
                            # tin hieu that (patch nao dang bi VETO) bi pha loang. Trong so
                            # 1+9*(1-target) dua patch bi nen len gap ~10 lan.
                            _w = 1.0 + 9.0 * (1.0 - _tgt)
                            loss = loss + gate_distill_lambda * \
                                (_w * (_ap - _tgt) ** 2).sum() / _w.sum()
                        else:
                            loss = loss + gate_distill_lambda * F.mse_loss(_ap, _tgt)
                    elif not getattr(model, '_warned_distill_shape', False):
                        print(f"⚠️  gate_distill: lech so patch (alpha {_ap.shape} vs nhan {_tgt.shape}) "
                              f"-> BO QUA distill. Kiem lai nhan sinh tu dung checkpoint/cau hinh chua.")
                        model._warned_distill_shape = True

                # Scale loss for gradient accumulation
                if is_training and gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps

            # 🔬 CHAN DOAN GGE: chi DO, khong doi loss/forward/gradient gi ca.
            if is_training and gge_diag > 0 and ogm_state is not None:
                ogm_state['dsteps'] = ogm_state.get('dsteps', 0) + 1
            if (is_training and gge_diag > 0 and ogm_state is not None
                    and (ogm_state['dsteps'] - 1) % gge_diag == 0):
                with autocast('cuda', enabled=_amp_on, dtype=amp_dtype):
                    _st = _gge_residual_stats(
                        model,
                        dict(pixel_values=pixel_values, input_ids=input_ids,
                             attention_mask=attention_mask, labels=labels, stage=stage,
                             answer_weights=answer_weights, question_types=question_types,
                             images_384=images_384, raw_questions=raw_questions,
                             sample_weights=sample_weights, region_map=region_map),
                        labels)
                if _st is not None:
                    for _k, _v in _st.items():
                        ogm_state['gge_' + _k] = ogm_state.get('gge_' + _k, 0.0) + _v
                    ogm_state['gge_n'] = ogm_state.get('gge_n', 0) + 1

            # 🔬 OGM-GE: do do lech nang luc giua hai nhanh, ra he so ham gradient cho nhanh
            # dang thang. KHONG dong vao forward/loss — chi tinh mot con so.
            if (is_training and ogm_ge > 0 and ogm_state is not None
                    and ogm_ge_start_epoch <= current_epoch < ogm_ge_end_epoch):
                ogm_state['steps'] = ogm_state.get('steps', 0) + 1
                if (ogm_state['steps'] - 1) % max(1, ogm_ge_every) == 0:
                    with autocast('cuda', enabled=_amp_on, dtype=amp_dtype):
                        _s_gca, _s_tcvg = _ogm_branch_scores(
                            model,
                            dict(pixel_values=pixel_values, input_ids=input_ids,
                                 attention_mask=attention_mask, labels=labels, stage=stage,
                                 answer_weights=answer_weights, question_types=question_types,
                                 images_384=images_384, raw_questions=raw_questions,
                                 sample_weights=sample_weights, region_map=region_map),
                            labels)
                    if _s_gca is not None and _s_tcvg is not None and _s_tcvg > 1e-8:
                        _r = _s_gca / _s_tcvg
                        # Lam tron EMA: batch o day chi 12 mau (OGM-GE goc dung 64) nen ty so
                        # tung batch rat on. Lam tron thi he so bam vao xu the that, khong bam
                        # vao mot batch may rui.
                        _prev = ogm_state.get('ratio_ema')
                        ogm_state['ratio_ema'] = _r if _prev is None else \
                            ogm_ge_ema * _prev + (1 - ogm_ge_ema) * _r
                        ogm_state['ratio_raw_sum'] = ogm_state.get('ratio_raw_sum', 0.0) + _r
                        ogm_state['ratio_raw_n'] = ogm_state.get('ratio_raw_n', 0) + 1
                # He so chi < 1 khi GCA DANG THANG (ratio > 1). Neu TCVG dang thang thi
                # KHONG ham TCVG — muc tieu la cuu nhanh yeu, khong phai can bang doi xung.
                _re = ogm_state.get('ratio_ema')
                ogm_state['coeff'] = _ogm_coeff(_re, ogm_ge) if (_re is not None and _re > 1.0) else 1.0

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
                        _apply_ogm_to_grads()
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                        if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                            print("⚠️  NaN/Inf gradient detected, skipping step")
                            nan_grad_steps += 1
                            optimizer.zero_grad()
                            scaler.update()  # safe: scale+backward was called above
                            steps_since_update = 0
                            continue
                        grad_norm_sum += grad_norm.item()
                        grad_norm_max = max(grad_norm_max, grad_norm.item())
                        grad_norm_count += 1
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        _apply_ogm_to_grads()
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                        if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                            print("⚠️  NaN/Inf gradient detected, skipping step")
                            nan_grad_steps += 1
                            optimizer.zero_grad()
                            steps_since_update = 0
                            continue
                        grad_norm_sum += grad_norm.item()
                        grad_norm_max = max(grad_norm_max, grad_norm.item())
                        grad_norm_count += 1
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
                _gacv = getattr(model, '_last_gac', None)
                if _gacv is not None:
                    postfix['gac'] = f"{_gacv:.3f}"
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
            _apply_ogm_to_grads()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                print("⚠️  NaN/Inf gradient at end-of-epoch flush, skipping step")
                nan_grad_steps += 1
                scaler.update()
            else:
                scaler.step(optimizer)
                scaler.update()
        else:
            _apply_ogm_to_grads()
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
            'type_loss': 0.0,
            'nan_loss_steps': 0,
            'nan_grad_steps': 0,
            'grad_norm_mean': 0.0,
            'grad_norm_max': 0.0,
        }

    return {
        'loss': total_loss / num_batches,
        'answer_loss': total_answer_loss / num_batches,
        'type_loss': total_type_loss / num_batches if use_type_loss else 0.0,  # 🔥 NEW
        'nan_loss_steps': nan_loss_steps,
        'nan_grad_steps': nan_grad_steps,
        'grad_norm_mean': grad_norm_sum / grad_norm_count if grad_norm_count > 0 else 0.0,
        'grad_norm_max': grad_norm_max,
    }


def evaluate_full_val(model, dataloader, tokenizer, device, decode_cfg=None):
    """
    Tính EM và F1 trên TOÀN BỘ val set — gọi mỗi epoch để track best model.

    decode_cfg=None (mặc định) giữ nguyên hành vi cũ: greedy, không trie, không synonym.

    decode_cfg là dict -> val dùng ĐÚNG bộ decoding của eval.py/test:
        num_beams, repetition_penalty, max_length, prefix_trie, valid_answers_set, use_synonyms
    Lý do: val EM cũ đo bằng greedy/không ràng buộc trong khi test EM đo bằng
    beam3 + trie + synonym, nên chọn checkpoint theo val là chọn theo một hàm mục
    tiêu khác. Đo được trên checkpoints_s2_T2 seed42: best_model (chọn theo val cũ)
    = 71.48 test EM còn last_model (epoch 40) = 73.31 — lệch 1.83pp.

    Returns:
        dict với 'exact_match', 'f1_score', 'rouge1', 'rougeL' (%)
        và per-type EM: 'em_object', 'em_counting', 'em_color', 'em_location' (nếu có)
    """
    _cfg = decode_cfg or {}
    _nb   = _cfg.get('num_beams', 1)
    _rp   = _cfg.get('repetition_penalty', 1.0)
    _ml   = _cfg.get('max_length', 10)
    _trie = _cfg.get('prefix_trie', None)
    _vset = _cfg.get('valid_answers_set', None)
    _syn  = _cfg.get('use_synonyms', False)
    _snap = _cfg.get('snap_fn', None)
    _norm = _cfg.get('norm_fn', None) or (lambda t: _normalize_vn(t))
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
                max_length=_ml,
                num_beams=_nb,
                repetition_penalty=_rp,
                prefix_trie=_trie,
            )

            # Snap các output bị dính chữ về đáp án hợp lệ gần nhất — y hệt eval.py
            if _vset is not None and _snap is not None:
                predictions = [_snap(p, _vset) for p in predictions]

            for idx, (pred, label) in enumerate(zip(predictions, labels)):
                label_tokens = label[label != -100].cpu().tolist()
                gt = _decode_gt(tokenizer, label_tokens)

                em = 1.0 if _norm(pred) == _norm(gt) else 0.0
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
    parser.add_argument('--max_answer_length', type=int, default=10,
                        help='Max answer tokens (target truncation + val generation). Default 10 (ViVQA short answers). '
                             'Tang len (vd 32) cho dataset open-ended (OpenViVQA).')
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

    # Val decoding khớp test — mặc định TẮT để không đổi hành vi của mọi run cũ
    parser.add_argument('--val_match_test_decoding', action='store_true',
                       help='Val EM dùng đúng bộ decoding của eval.py/test (beam3 + repetition_penalty '
                            '+ constrained trie + snap + synonym) thay vì greedy không ràng buộc. '
                            'Không bật thì val EM và test EM đo hai thứ khác nhau, nên chọn checkpoint '
                            'theo val bị lệch (đo được 1.83pp trên s2_T2 seed42).')
    parser.add_argument('--val_trie_csv', type=str, default='archive/train_split.csv',
                       help='CSV dựng trie cho val decoding. PHẢI là train split — không bao giờ dùng test.')
    parser.add_argument('--val_decode_beams', type=int, default=3)
    parser.add_argument('--val_decode_rep_penalty', type=float, default=1.3)
    parser.add_argument('--val_decode_max_length', type=int, default=10)
    parser.add_argument('--val_decode_from_epoch', type=int, default=1,
                       help='Chỉ bật bộ decoding đắt tiền từ epoch này trở đi (các epoch trước dùng '
                            'greedy cho nhanh). Đặt >1 nếu muốn tiết kiệm thời gian ở giai đoạn đầu.')

    # Freezing
    parser.add_argument('--unfreeze_encoder_layers', type=int, default=3, help='Number of text encoder layers to unfreeze')
    parser.add_argument('--freeze_decoder', action='store_true', help='Freeze decoder (default: unfrozen)')
    parser.add_argument('--gate_types_no_typeloss', action='store_true',
                        help='🔬 Dua nhan LOAI vao gate ma KHONG bat type_loss. O con thieu cua ma tran 2x2: '
                             'bo --use_type_loss von lam question_types=None nen gate mat luon nhan loai, tuc bo '
                             'HAI thu cung luc. Phan ra do duoc: bo type_loss giup LOCATION (+0.72 p=0.023 9/10), '
                             'blind gate lai HAI LOCATION (-0.80 0/4) -> hai tac dong nguoc chieu.')
    parser.add_argument('--freeze_lm_head', action='store_true',
                        help='🔬 Dong bang lm_head (tied voi embedding BARTpho pretrain). Nhanh LoRA '
                             'von van mo khoa 41M tham so nay va train nan prior TAN SUAT dap an vao do '
                             '(Spearman ||W||-||W0|| vs log tan suat = +0.507, tu -0.052 truoc train) — '
                             'chinh prior de bep lop chua tung train. Kien truc KHONG doi.')
    parser.add_argument('--decoder_lr_multiplier', type=float, default=1.0,
                        help='LR multiplier cho decoder (neo mem khi warm-start: <1 -> decoder troi cham, giu gan T0, giam break ma van train)')
    
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
    parser.add_argument('--use_decoder_lora', action='store_true',
                        help='LoRA cho BARTpho DECODER. Decoder la 243.6M (40.1%% model) va gan nhu '
                             'toan bo 279.8M trainable, tren 10.800 mau = 26.000 tham so/vi du. '
                             'Nham vao khoang cach tong quat hoa (train 99.12 vs test 73.31).')
    parser.add_argument('--decoder_lora_r', type=int, default=16)
    parser.add_argument('--decoder_lora_alpha', type=int, default=32)
    parser.add_argument('--decoder_lora_dropout', type=float, default=0.1)
    parser.add_argument('--use_text_lora', action='store_true',
                       help='Use LoRA for text encoder (BETTER than unfreezing layers for ~10K samples)')
    parser.add_argument('--text_lora_r', type=int, default=16,
                       help='LoRA rank for text encoder (default: 16, higher than vision)')
    parser.add_argument('--text_lora_alpha', type=int, default=32,
                       help='LoRA alpha scaling for text (default: 32)')
    parser.add_argument('--text_lora_dropout', type=float, default=0.1,
                       help='LoRA dropout for text encoder (default: 0.1)')

    # 🔥 Vision Dependency (combat text shortcut)
    parser.add_argument('--tcvg_fg_2pass', action='store_true',
                        help='2 luot: luot 1 chay GCA de co v^(L) roi tinh alpha tu do (DUNG Eq.4 '
                             'cua paper, giu chon loc theo tung cau hoi); luot 2 chay GCA lai tu '
                             'vision tho voi residual nhan alpha. Can --tcvg_fusion_gate.')
    parser.add_argument('--tcvg_fusion_gate', action='store_true',
                        help='TCVG dieu khien CUONG DO HOP NHAT cua GCA theo tung patch va theo loai '
                             'cau hoi (doi NOI DUNG patch) thay vi tron hau ky sau GCA (doi TRONG SO). '
                             'Decoder attention khong the tai tao thay doi noi dung -> gating khong '
                             'con du thua voi attention.')
    parser.add_argument('--tcvg_topk_random', action='store_true',
                        help='DOI CHUNG cho --tcvg_topk: chon k patch NGAU NHIEN thay vi top-alpha. '
                             'Cung kien truc, cung ngan sach token. gate > random => TCVG biet nhin dau.')
    parser.add_argument('--tcvg_topk', type=int, default=0,
                        help='>0: chi giu top-k patch theo alpha, XOA phan con lai khoi chuoi '
                             'dua vao decoder. Lam gating co nang luc ngoai khong gian ham cua '
                             'attention (attention khong the lay lai token da mat).')
    parser.add_argument('--decoder_vision_only', action='store_true',
                        help='Decoder chi cross-attend vao vision (bo text khoi encoder_hidden_states). '
                             'Dung de co lap dong gop that cua TCVG.')
    parser.add_argument('--tcvg_type_bias', action='store_true',
                        help='b (vision_bias) THEO TUNG LOAI thay vi mot scalar dung chung. '
                             'Cho phep COUNT hoc b lon (alpha->1, giu het instance) trong khi '
                             'COLOR hoc b nho (chon loc). Init = vision_gate_init cho moi loai '
                             'nen luc bat dau giong het baseline.')
    parser.add_argument('--tcvg_type_ctx', action='store_true',
                        help='Dich tron co dieu kien theo loai: blend_target = text_pooled + '
                             'tanh(g_type)*Attention(q_type, v^(L)). Nen mot patch = dung boi canh '
                             'thi giac dung loai thay cho patch cuc bo, thay vi mot vector chung. '
                             'ctx_gate init 0 -> tai khoi tao DUNG BANG cong thuc paper.')
    parser.add_argument('--tcvg_ln_mode', type=str, default='post', choices=['post','pre','none'],
                        help="Vi tri LayerNorm trong TCVG. post (goc, LN SAU phep tron) xoa sach "
                             "thanh phan bien do cua gate (do duoc: alpha=0.45 nen token con 46.4%% "
                             "do dai, LN keo ve 99.99%%). pre = LN truoc khi tron -> giu bien do. "
                             "none = dung phuong trinh §3.3 cua paper, khong LayerNorm.")
    parser.add_argument('--alpha_reg_lambda', type=float, default=0.0,
                        help='(A) Phat lambda*(1-alpha)^2: gate CHI gate khi giam answer loss du de '
                             'vuot phat. Muc dich: mo gate (alpha->1) cho loai gating khong giup (COUNT) '
                             'de HET HAI, giu gate cho COLOR. Vd 0.05.')
    parser.add_argument('--gate_layerscale_pertype', action='store_true',
                        help='NON-HARM moi loai: v_out = v + beta_type*(gated - v). beta hoc rieng '
                             'tung loai; beta=0 = identity (T0, khong hai), beta=1 = gate day du (T2).')
    parser.add_argument('--gate_layerscale_init', type=float, default=1.0,
                        help='Init cho beta_type (1.0 = khoi dau tai T2, gate_net co gradient tu dau).')
    parser.add_argument('--gate_layerscale_l2', type=float, default=0.0,
                        help='Phat L2 lambda*mean(beta^2) keo beta_type ve 0 (identity) tung loai: loai '
                             'KHONG loi (LOCATION) bi keo ve pass-through, loai co loi (COLOR) chong lai. Vd 0.02.')
    parser.add_argument('--slot_attn', action='store_true',
                        help='TIEN HOA TCVG: slot attention gom patch thanh K instance theo loai, '
                             'them K token vao decoder (patch nguyen ven). Gom instance = thu decoder '
                             '1-luot khong lam duoc. Dung KHONG kem --use_vision_gate.')
    parser.add_argument('--num_slots', type=int, default=4, help='So slot')
    parser.add_argument('--summary_token', action='store_true',
                        help='TCVG dang TOKEN-CONG-THEM: patch giu nguyen, them 1 token tom tat theo '
                             'loai vao chuoi encoder. Hai none theo cau truc (decoder co the bo qua). '
                             'Dung KHONG kem --use_vision_gate.')
    parser.add_argument('--concat_fusion', action='store_true',
                        help='Thay GCA (attention dat) bang concat cau hoi pooled + anh (re) roi TCVG. '
                             'Test: TCVG co ganh lai duoc perf khong -> TCVG load-bearing thay vi du thua.')
    parser.add_argument('--text_path_dropout', type=float, default=0.0,
                        help='H1: khi train, che text tokens khoi cross-attention decoder voi xac '
                             'suat p (theo mau). Buoc duong vision+TCVG dung vung mot minh -> TCVG '
                             'phai mang thong tin loai. Inference giu ca hai. Vd 0.3.')
    parser.add_argument('--gca_strength', type=float, default=1.0,
                        help='He so LAM YEU GCA co chu dich (1.0=day du, 0.5=nua, 0.0=GCA tat). '
                             'Delta(T2-T0) tang khi gca_strength giam = TCVG co nang luc bi che.')
    # 🔬 OGM-GE cap module (Peng et al. CVPR 2022). Xem khoi ghi chu truoc run_one_epoch_deterministic.
    # KHAC gca_dropout o cho quyet dinh: forward pass KHONG doi mot chut nao, chi ham gradient
    # cua flamingo_fusion khi GCA dang thang. Suy luan hoan toan khong doi.
    parser.add_argument('--ogm_ge', type=float, default=0.0,
                        help='He so alpha cua OGM-GE. 0.0 = tat. coeff = 1 - tanh(alpha * ratio), '
                             'ratio = score(GCA don doc) / score(TCVG don doc). Thu 0.5 truoc.')
    parser.add_argument('--ogm_ge_noise', type=float, default=0.0,
                        help='Phan GE: cong nhieu Gauss std = he_so * std(grad) vao grad da ham, '
                             'bu lai phan generalization mat di do ham gradient. Goc dung ~0.5-1.0.')
    parser.add_argument('--ogm_ge_every', type=int, default=1,
                        help='Tinh lai ty so moi k buoc (giua cac lan thi dung lai ratio_ema). '
                             'k>1 giam chi phi 2 forward phu; k=4 dua 1.7x xuong ~1.2x.')
    parser.add_argument('--ogm_ge_ema', type=float, default=0.9,
                        help='Lam tron EMA cho ty so. Batch o day 12 mau (OGM-GE goc 64) nen '
                             'ty so tung batch on; 0.0 = khong lam tron.')
    parser.add_argument('--ogm_ge_start_epoch', type=int, default=0)
    parser.add_argument('--ogm_ge_end_epoch', type=int, default=10**9,
                        help='Ngung dieu bien sau epoch nay (OGM-GE goc chi dieu bien giai doan dau).')
    # 🔬 QGND — neo cau noi thi giac->ngon ngu bang TU VUNG CAU HOI thay vi 314 dap an.
    # Do duoc: tu vung cau hoi 2336 tu vs tu vung dap an 377; "hươu cao cổ" xuat hien 164 lan
    # trong cau hoi train ma mo hinh khong noi ra duoc lan nao (0/46).
    parser.add_argument('--iqg_lambda', type=float, default=0.0,
                        help='IQG: bat DECODER sinh lai CAU HOI tu bo nho THI GIAC TRUOC HOP NHAT '
                             '(inverse VQA). Day la nguon diem-tuong-ung anh-chu MOI duy nhat lay '
                             'duoc ma khong can du lieu ngoai. 0 = tat. Thu 0.2 truoc.')
    parser.add_argument('--iqg_mask', type=float, default=0.0,
                        help='Xac suat CHE moi token dau vao decoder trong nhanh IQG. BAT BUOC > 0: '
                             'khong che thi teacher forcing tu giai duoc nhiem vu (do duoc: loss '
                             '4.66 -> 0.046, va bo nho thi giac ZERO chi lam loss tang 0.035 nats). '
                             'Thu 0.7.')
    parser.add_argument('--iqg_check_every', type=int, default=50,
                        help='Moi k buoc, tinh lai IQG loss voi bo nho thi giac ZERO va ghi lai '
                             'chenh lech. Chenh ~0 = muc tieu giai duoc KHONG CAN ANH = vo nghia.')
    parser.add_argument('--qgnd_lambda', type=float, default=0.0,
                        help='Trong so cho mat mat neo cau noi. 0 = tat hoan toan. Thu 0.1 truoc.')
    parser.add_argument('--qgnd_temp', type=float, default=0.07,
                        help='Nhiet do cho do tuong dong cosine giua thi giac va embedding tu.')
    parser.add_argument('--qgnd_drop_top', type=int, default=60,
                        help='Bo N token PHO BIEN NHAT trong cau hoi (tu chuc nang: la, gi, nao, co...). '
                             'Chung xuat hien o moi cau nen khong mang thong tin thi giac nao.')
    parser.add_argument('--qgnd_min_freq', type=int, default=5,
                        help='Chi giu token xuat hien it nhat N lan trong cau hoi train.')
    parser.add_argument('--gge', type=float, default=0.0,
                        help='GGE (Han et al., ICCV 2021) cap module: trong so moi vi tri nhan = '
                             'clamp(1 - p_GCA_don_doc(gold), 0, 1). Ep CE chi hoi ve PHAN DU ma '
                             'nhanh GCA khong giai thich duoc. 0 = tat. 1.0 = dung ban goc. '
                             'Kien truc/suy luan KHONG doi; trong so deu = trung khop baseline.')
    parser.add_argument('--gge_floor', type=float, default=0.0,
                        help='San cho trong so GGE. 0.0 = ban goc (mau GCA giai duoc bi tat han). '
                             '>0 giu lai mot phan tin hieu tren nhung mau do.')
    parser.add_argument('--gge_start_epoch', type=int, default=0,
                        help='Bat GGE tu epoch nay. Epoch dau mo hinh chua hoc gi nen trong so ~0.53 '
                             'va it thong tin; hoan lai vai epoch co the sach hon.')
    parser.add_argument('--gge_diag', type=int, default=0,
                        help='CHI CHAN DOAN, khong doi loss: moi k buoc do phan bo trong so GGE '
                             '(1 - p_GCA(gold)) tren TRAIN. Dung de biet nhan co rong khong TRUOC '
                             'khi cai dat GGE (gate_distill da chet vi nhan rong).')
    parser.add_argument('--gca_dropout', type=float, default=0.0,
                        help='Xac suat TAT GCA moi batch khi TRAIN (suy luan luon bat day du). '
                             'Ep TCVG khong the lui ve anh xa dong nhat -> buoc phai hoc chuc nang that.')
    parser.add_argument('--gca_dropout_types', type=str, default='',
                        help="'' = tat GCA cho toan batch; '0,3' = chi tat cho OBJECT va LOCATION "
                             '(cac loai co alpha ghim o 1). Map: 0=OBJECT 1=COUNT 2=COLOR 3=LOCATION.')
    parser.add_argument('--tcvg_proto_gate', action='store_true',
                        help='Gate bang PROTOTYPE THUAN cua loai: alpha_i = scale*<p_type, k(v_i)>. '
                             'Query chi phu thuoc LOAI (khong t_cls), nen hoc cai chung cua loai '
                             '(vd huong "sac mau") thay vi instance ma GCA da lo -> khong du thua GCA.')
    parser.add_argument('--tcvg_global_scalar_gate', action='store_true',
                        help='Kiem dinh gia thuyet TCVG=shrinkage estimator (khong phai spatial '
                             'selector): alpha = sigma(vision_bias), MOT so hoc duoc duy nhat, KHONG '
                             'phu thuoc patch/cau hoi/loai (gate_net bi bo qua). Neu bien the nay dat '
                             'gan T2 chuan -> toan bo gia tri TCVG nam o cuong do co trung binh, '
                             'khong phai o chon loc theo patch.')
    parser.add_argument('--gca_box_tokens', action='store_true',
                        help='Token box lam key/value PHU cho GCA: moi patch attend qua ca text VA box '
                             '-> thong tin vat the vao TRUOC gate, alpha duoc tinh tren feature da '
                             'biet-vat-the. Model tu hoc box nao quan trong voi patch nao. Output GCA van '
                             '[B,P,D] nen TCVG khong doi, KHONG vi pham "TCVG sau Flamingo". '
                             'LUU Y: box la DAU VAO -> can annotation ca luc eval (do tran, khong bao cao).\n'
                             'CAN --tcvg_spatial_blend_region_map.')
    parser.add_argument('--box_ground_lambda', type=float, default=0.0,
                        help='Box-Grounded multi-task: BCE tung patch "co thuoc mot ca the khong". '
                             'Box la NHAN dau ra, KHONG phai dau vao -> suy luan khong can annotation, '
                             'con so so duoc voi bai doi thu. CAN --tcvg_spatial_blend_region_map.')
    parser.add_argument('--answer_cls_lambda', type=float, default=0.0,
                        help='Dau phan loai tren tap dap an train (~328 lop), doc cung bieu dien ma '
                             'decoder doc. Nham vao 94.8%% loi von la CHON NHAM dap an hop le khac. '
                             'La dau PHU song song, khong thay the sinh chuoi.')
    parser.add_argument('--box_class_lambda', type=float, default=0.0,
                        help='CE du doan lop COCO tung patch (81 lop). CAN --tcvg_spatial_blend_region_map '
                             'tro vao patch_class_map_flat.pkl (gia tri = ID LOP). Giau hon objectness '
                             '6.3 bit/patch vs 1 bit; box la NHAN nen suy luan khong can annotation.')
    parser.add_argument('--box_class_n', type=int, default=81,
                        help='So lop cho box_class_head (80 lop COCO + nen)')
    parser.add_argument('--box_count_typed', action='store_true',
                        help='Chi ap loss count tren mau COUNT. Do duoc o run bgc: ap len moi anh '
                             'lam COLOR -3.52 (count_head doc vector pool DUNG CHUNG) trong khi '
                             'COUNT +1.35. Bat co nay de can thiep thanh phau thuat.')
    parser.add_argument('--box_count_lambda', type=float, default=0.0,
                        help='Box-Grounded multi-task: L1 tren so ca the phan biet trong anh.')
    parser.add_argument('--gate_spatial_pertype', type=str, default=None,
                        help='Mask "OBJECT,COUNT,COLOR,LOCATION" vd "0,0,1,0": bat phan dieu chinh alpha '
                             'THEO TUNG PATCH chi o nhung loai duoc chon; cac loai khac chi con b_type '
                             '(muc nen theo loai). Muc tieu: KHONG HAI type nao.')
    parser.add_argument('--gate_box_content', action='store_true',
                        help='Nap NOI DUNG box COCO vao 2 cho TCVG dang rong: (1) blend_target = '
                             'text_pooled + tanh(g)*box_fuse([v_proj;box_feat]), (2) logit alpha += '
                             'tanh(g2)*box_alpha_head(box_feat). Ca hai zero-init -> tai init BANG '
                             'TCVG goc. box_feat = (is_obj, region_size, n_instances) tu region_map, '
                             'bat bien voi thu tu id. CAN --tcvg_spatial_blend_region_map de nap pkl.')
    parser.add_argument('--box_class_vocab', type=int, default=0,
                        help='>0 (vd 81): nhung ID lop COCO cua tung patch vao box_feat. CAN '
                             '--tcvg_spatial_blend_region_map patch_class_map_flat.pkl.')
    parser.add_argument('--box_max_inst', type=int, default=32,
                        help='Chuan hoa so ca the: n_reg / box_max_inst')
    parser.add_argument('--tcvg_spatial_blend', action='store_true',
                        help='blend_target = tron mem(text_pooled, local-pool 3x3 cua v_proj tren '
                             'luoi patch) thay vi MOT vector text_pooled dung chung cho moi patch. '
                             'He so tron beta hoc tu query (khong hardcode danh sach loai cau hoi) -> '
                             'giu cau truc TUONG DOI giua cac vung anh cho cau hoi quan he khong gian, '
                             'trong khi van cho phep tron toan cuc cho cau hoi thuoc tinh toan cuc.')
    parser.add_argument('--tcvg_spatial_blend_region_map', type=str, default=None,
                        help='Duong dan file .pkl (tu build_patch_region_map.py) chua dict '
                             '{img_id: int16[num_patches]} chi so REGION COCO THAT (ranh gioi vat '
                             'the that) cho tung patch -- dung thay cua so 3x3 co dinh trong '
                             'tcvg_spatial_blend. Anh khong co annotation (thieu trong file) tu '
                             'dong fallback ve pool toan cuc (an toan). None = hanh vi cu (3x3).')
    parser.add_argument('--tcvg_dynamic_peek', action='store_true',
                        help='TCVG DONG: alpha khong tinh 1 lan dung chung ca cau tra loi -- '
                             'cong them tin hieu "sap can gi" vao query (peek_embedding). Train: '
                             'tu nhan that (mien phi, khong can generate). Eval: generate() tu '
                             'chay 2 pass (draft greedy -> peek -> chay that), cham hon ~2x luc eval.')
    parser.add_argument('--tcvg_alpha_from_gca', action='store_true',
                        help='HYP #1: alpha = max trong so attention (da tinh cho GCA fusion, '
                             'khong ton them chi phi) qua cac vi tri text hop le, thay vi hoc '
                             'gate_net/type_embedding/query_proj rieng. KHONG THAM SO MOI trong '
                             'duong tinh alpha -- kiem tra gia tri TCVG nam o cong thuc tron hay '
                             'o bo chon loc rieng. Chi ap dung fusion_type=text2vision.')
    parser.add_argument('--tcvg_refine_gate', action='store_true',
                        help='TCVG nhu BO TINH CHINH theo loai (FiLM residual) thay vi gate/nen. '
                             'v_hat = LN(v + tanh(g_type)*(gamma_type*v_proj + beta_type)). Init identity. '
                             'Them bien doi rieng loai (khong chi tru); song sot qua LayerNorm.')
    parser.add_argument('--tcvg_attn_gate', action='store_true',
                        help='Tinh alpha bang TICH VO HUONG <q_type, k(v_i)> thay vi concat-MLP cong. '
                             'Type nhan voi noi dung patch -> doi type BAT BUOC doi thu tu patch '
                             '(per-patch per-type la tinh chat cau truc). Nham vao chan doan: '
                             'concat-MLP lam type thanh mot do dich chung, khong chon duoc vung.')
    parser.add_argument('--use_type_codebook', action='store_true',
                        help='PHAT HIEN LOAI KHONG GIAM SAT: luong tu hoa t_cls ve prototype gan '
                             'nhat (VQ-VAE codebook), prototype dong vai tro e_type. Bo hoan toan '
                             'nhu cau nhan loai -> khong con preprocessing type cho tung bo du lieu. '
                             'Dung de xuat Future Work §5 cua paper. Dung KEM hoac THAY --use_type_loss.')
    parser.add_argument('--codebook_size', type=int, default=4, help='So prototype trong codebook')
    parser.add_argument('--codebook_beta', type=float, default=0.25, help='He so commitment VQ-VAE')
    parser.add_argument('--codebook_lambda', type=float, default=0.1,
                        help='Trong so cua vq_loss trong tong loss')
    parser.add_argument('--decoder_pool_vision', type=int, default=0,
                        help='>0: average-pool 197 token thi giac xuong K token TRUOC decoder. '
                             'Xoa kha nang chon loc theo patch cua cross-attention decoder, nen '
                             'moi su chon loc buoc phai xay ra trong TCVG. Phep do CHAN DOAN de '
                             'tach bach T0 vs T2 (K=1 la co lap triet de nhat).')
    parser.add_argument('--tcvg_norm_type_emb', action='store_true',
                        help='B: L2-normalize type_embedding roi nhan mot scalar hoc duoc.')
    parser.add_argument('--tcvg_type_null', action='store_true',
                        help='A: them offset hoc duoc theo loai vao dich tron (init 0).')
    parser.add_argument('--slot_stage', type=str, default='post', choices=['post', 'pre', 'pre_gated'],
                        help="slot doc feature nao: 'post' (cu) = SAU TCVG, nen slot phai ca the hoa "
                             "tu ban da bi nen (gate nen 35.8%% patch o COUNT, 30.0%% o COLOR theo "
                             "alpha_oracle). 'pre' = TRUOC TCVG, slot tong hop tu ban nguyen. "
                             "KHONG vi pham rang buoc 'TCVG sau Flamingo' (paper KES 2026).")
    parser.add_argument('--slot_init_std', type=float, default=0.02,
                        help='Do phan hoa slot luc init. 0.02 (cu) -> do duoc cosine giua cac slot '
                             '= 0.997, tuc K BAN SAO -> slot attention khong the pha vo doi xung. '
                             'Nen dat >= 0.5.')
    parser.add_argument('--slot_tanh_gate', action='store_true',
                        help='tanh-gate init 0 cho token slot -> KHONG GAY HAI luc init. Can vi '
                             'out_norm=LayerNorm cuong buc ||slot||=32.0 = ||patch||, nen K token '
                             'nhieu canh tranh binh dang voi patch that tu buoc 0 (dau hieu: '
                             'LOCATION tut -1.89). Quy uoc repo: type_experts/l6_fuse deu zero-init.')
    parser.add_argument('--slot_no_type', action='store_true',
                        help='ABLATION: bo e_type khoi slot init. Neu slot van tot bang thi gain '
                             'KHONG den tu dieu kien hoa theo loai -> khong duoc goi la TCVG mo rong.')
    parser.add_argument('--slot_bind_lambda', type=float, default=0.0,
                        help='Giam sat BINDING cua slot attention bang region COCO: cac patch cung '
                             'vat the phai vao cung slot. Bat bien voi hoan vi slot (khong can '
                             'Hungarian). CAN --tcvg_spatial_blend_region_map de nap region map. '
                             '0 = tat. Day la nhanh CHAN DOAN dung bbox -- xem ghi chu trong code.')
    parser.add_argument('--gate_distill_path', type=str, default=None,
                        help='.npz alpha oracle (sinh boi eval.py --dump_oracle_alpha tren CHINH '
                             'train_csv nay). Bat distill: alpha hoc duoc bi keo ve alpha oracle.')
    parser.add_argument('--gate_distill_lambda', type=float, default=0.0,
                        help='He so MSE(alpha_hoc, alpha_oracle). 0 = tat.')
    parser.add_argument('--gate_distill_mode', type=str, default='mse', choices=['mse', 'tail'],
                        help="mse: MSE deu tren 196 patch. tail: trong so 1+9*(1-target) -- dua "
                             "patch BI NEN len ~10x, vi 82% target la ~1 (giu nguyen) nen MSE deu "
                             "bi phan tam thuong lan at (do o analyze_oracle_alpha.py).")
    parser.add_argument('--gate_sparsity_lambda', type=float, default=0.0,
                        help='He so phat len mean(alpha) cua TCVG. 0 = tat (hanh vi goc). '
                             'Gia tri duong tao ap luc buoc gate phai chon loc patch.')
    parser.add_argument('--tcvg_gate_mode', type=str, default='blend',
                        choices=['blend', 'multiply'],
                        help="blend (goc): gated = a*v + (1-a)*text_pooled, tuc thay patch bang "
                             "MOT vector text dung chung. multiply: gated = a*layer_norm(v), tuc "
                             "nen patch ve 0 — dung nghia 'suppress' nhu paper mo ta.")
    parser.add_argument('--tcvg_two_layer', action='store_true',
                        help='Ap TCVG sau MOI lop GCA (gate phu sau lop GCA #1)')
    parser.add_argument('--type_emb_lr_multiplier', type=float, default=1.0,
                        help='He so nhan LR rieng cho vision_gating.type_embedding. Mac dinh 1.0 '
                             'khien vector (norm ~32) chi xoay ~0.5 do sau 40 epoch, tuc ma loai '
                             'dong bang o init ngau nhien.')
    parser.add_argument('--tcvg_type_emb_std', type=float, default=None,
                        help='std init cho type_embedding cua TCVG. Mac dinh None = nn.Embedding '
                             'N(0,1) (norm ~32, gay gan ngau nhien khuon mau gating theo seed). '
                             'Dat 0.02 de gate khoi dau type-agnostic.')
    parser.add_argument('--tcvg_type_emb_init_path', type=str, default=None,
                        help='Duong dan file .pt (tu compute_type_prototypes.py) chua tensor '
                             '[num_types, hidden_dim] prototype ngu nghia THAT (mean-pool embedding '
                             'BARTpho dong bang tren cau hoi that moi type) -- thay khoi tao ngau '
                             'nhien bang huong co y nghia, sua nguyen nhan goc cua symmetry-breaking.')
    parser.add_argument('--tcvg_type_emb_init_auto', action='store_true',
                        help='Tu dong tinh prototype tu args.train_csv (cot question/type) NGAY '
                             'TRONG train.py, dung chinh text encoder cua model (con nguyen pretrained '
                             'luc nay, truoc freeze_pretrained/LoRA) -- KHONG can chay script rieng '
                             'truoc, khong can file .pt. Tong quat cho moi dataset/taxonomy (chi can '
                             'CSV co cot question+type). Uu tien hon --tcvg_type_emb_init_path neu ca hai cung dat.')
    parser.add_argument('--use_vision_gate', action='store_true',
                       help='Enable learnable vision gating (boost vision importance)')
    parser.add_argument('--vision_gate_init', type=float, default=1.5,
                       help='Initial vision gate value (>1.0 = prefer vision, default=1.5)')
    parser.add_argument('--vision_gate_min_alpha', type=float, default=0.35,
                       help='Minimum alpha floor for VisionGating (0.35 recommended to prevent '
                            'COLOR/COUNT from over-suppressing vision: α≈0.18/0.33 without floor)')
    parser.add_argument('--vision_gate_min_alpha_pertype', type=str, default=None,
                       help='Per-type alpha floor "OBJECT,COUNT,COLOR,LOCATION" (vd "0.5,0.8,0.0,0.0"): '
                            'nang floor cho COUNT (giu distributed view) & OBJECT (bao hoa) de gate gan-inert '
                            'o loai gate hai, giu tu do cho COLOR/LOCATION. Soft-interp thuan, khong doi objective.')
    parser.add_argument('--vision_gate_max_alpha_pertype', type=str, default=None,
                       help='Per-type alpha ceiling "OBJECT,COUNT,COLOR,LOCATION" (vd "0.9,1.0,0.9,0.9"): '
                            'mien COUNT khoi tran chung (2026-08-09: alphaclamp[0.1,0.9] hai COUNT nhat quan '
                            '-2.14pp vi COUNT can alpha gan 1, trong khi tran giup OBJECT/LOCATION tranh cuc doan).')
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
    parser.add_argument('--type_moe', action='store_true', help='Type-routed MoE: moi loai 1 expert FFN')
    parser.add_argument('--type_moe_bottleneck', type=int, default=256)
    parser.add_argument('--type_moe_soft', action='store_true', help='MoE soft-routing: tron theo softmax(type_logits) thay vi argmax (robust type-mispredict)')
    parser.add_argument('--vision_backbone_layer', type=int, default=-1,
                        help='Dùng layer L intermediate của SigLIP làm feature CHÍNH cho GCA+TCVG+decoder (ít language-align, giữ local/màu). -1=last_hidden. Vd 6 = L6.')
    parser.add_argument('--gate_vision_layer', type=int, default=-1,
                       help='Gate tinh alpha tu hidden layer L (local structure); -1=output (mac dinh).')
    parser.add_argument('--token_prior_gamma', type=float, default=0.0,
        help='🔬 Ha trong so CE cho token DOAN DUOC tu prior cua tap dap an. '
             'w = 1 - gamma*p_max(token|loai,vi tri), uoc luong TU DU LIEU TRAIN nen tu dong '
             'thich nghi voi bo khac. 0.0 = TAT (mac dinh, trung khop tuyet doi baseline).')
    parser.add_argument('--kl_pretrained_lambda', type=float, default=0.0,
        help='🔬 phat do lech khoi BARTpho pretrained tren token dap an, chuan hoa theo CE. '
             '0.0 = TAT hoan toan (mac dinh, khong doi run cu). Muc tieu ty trong 10-30%%: thu 0.15-0.5')
    parser.add_argument('--type_from_gate_lambda', type=float, default=0.0,
                       help='Them mot dau type doc THI GIAC SAU GATE (gradient di qua alpha). '
                            'type_head thuong doc text_cls nen CE khong chay qua gate; co nay '
                            'bien type_loss thanh muc tieu CUA GATE. 0 = tat.')
    parser.add_argument('--gate_type_blind', action='store_true',
                       help='Ablation: TCVG KHONG dieu kien hoa theo loai o CA train va test '
                            '(ep type_ids=None ca hai nhanh). Dung cho o Tblind cua bang ablation.')
    parser.add_argument('--gate_pertype_net', action='store_true',
                       help='Moi loai cau hoi MOT mang gate rieng tinh alpha (y thiet ke ban dau). '
                            'Khac type_moe (tach FFN HAU-gate) va khac gate_no_type_emb (mot mang '
                            'chung, chi doi query). Ca 4 ban khoi tao GIONG HET nhau.')
    parser.add_argument('--patch_self_attn', action='store_true',
                       help='MODULE THEM VAO: cho cac patch attend LAN NHAU, dat giua Flamingo va '
                            'TCVG (khong doi thu tu module cua bai). Ly do: text2vision co '
                            'query=vision key=text, decoder thi text->vision, nen sau SigLIP dong '
                            'bang cac patch KHONG BAO GIO attend lan nhau. out_proj zero-init.')
    parser.add_argument('--psa_heads', type=int, default=8)
    parser.add_argument('--gate_alpha_budget', action='store_true',
                       help='alpha la mot PHAN BO tren cac patch (softmax over patches x ngan sach) '
                            'thay vi 197 sigmoid DOC LAP. Ep canh tranh: sang cho nay thi toi cho kia. '
                            'Van de goc: khong canh tranh nen model dat het alpha=1 (LOCATION 0.9995).')
    parser.add_argument('--gate_budget_init', type=float, default=0.72,
                       help='Ngan sach alpha trung binh luc khoi tao (mac dinh 0.72 = alpha tb hien tai).')
    parser.add_argument('--gate_blend_vorig', action='store_true',
                       help='Dich tron cua gate = noi suy giua text_pooled va v_orig (patch TRUOC '
                            'Flamingo). Hien (1-alpha) tron ve MOT vector dung chung cho ca 197 patch. '
                            'Moi bien the da thu deu la ham cua v_proj (trong span); v_orig la bieu dien '
                            'decoder khong co duong nao khac de thay. gamma khoi tao ~0 -> non-harm.')
    parser.add_argument('--gate_no_text_cls', action='store_true',
                       help='Ablation bu: query cua gate CHI dung e_type, BO t_cls -> chon loc '
                            'khong gian theo LOAI nhung mu noi dung cau hoi. alpha VAN bien thien '
                            'theo patch qua v_i, nen khac han mot hang so theo loai.')
    parser.add_argument('--gate_no_type_emb', action='store_true',
                       help='ABLATION: gate chi dua tren t_cls (BO e_type), tach co-gating khoi type-conditioned.')
    parser.add_argument('--gate_gca_residual', action='store_true',
                       help='TCVG dieu khien luong GCA (gated = v - gamma*(1-alpha)*(v - v_orig)) thay vi tron ve text_pooled')
    parser.add_argument('--gate_per_channel', action='store_true',
                        help='Gate per patch×channel: alpha [B,P,D] chon feature theo loai (bieu dien manh hon per-patch scalar)')
    parser.add_argument('--gate_blend_l6', action='store_true',
                        help='blend_target = proj(L6) spatial thay text_pooled (can --gate_vision_layer >=0). Patch suppress ve ban L6 cua chinh no.')
    parser.add_argument('--gate_l6_fuse', action='store_true',
                        help='blend_target = text_pooled + l6_fuse([v_L12;L6]): L6 hoc bien sang semantic space roi cong (fix L6-blend). Can --gate_vision_layer >=0.')
    parser.add_argument('--gate_l6_fuse_bottleneck', type=int, default=256)
    parser.add_argument('--vision_l6_enrich', action='store_true',
                        help='PROBE H1/H2: decoder nhan v_L12 + learned(L6) UNCONDITIONAL, no gate (can --gate_vision_layer >=0). Test L6 co signal task.')
    parser.add_argument('--gate_harm_lambda', type=float, default=0.0,
                        help='do-no-harm: phat relu(loss_gate_on - loss_gate_off) per-sample -> gate rut ve identity o sample no lam te (giam break, giu fix). 2x decoder forward.')
    parser.add_argument('--gate_harm_protect', action='store_true',
                        help='EM-aligned harm: chi bao ve token ma gate-off argmax DUNG (chong right->wrong truc tiep, dac tin hieu). Kem --gate_harm_lambda.')
    parser.add_argument('--gate_answer_contrastive_lambda', type=float, default=0.0,
                        help='GAC: InfoNCE gated-vision <-> ANSWER embedding (khong phai question -> pha attractor alpha->0). Giam sat CHINH gate de chon patch du doan dap an, phan biet per-question. Recommend 0.05-0.2.')
    parser.add_argument('--gate_answer_contrastive_temp', type=float, default=0.07,
                        help='Temperature cho GAC (SimCLR default 0.07).')
    parser.add_argument('--gate_answer_contrastive_warmup_epochs', type=int, default=0,
                        help='Ramp gate_answer_contrastive_lambda tuyen tinh tu 0 len target trong N epoch dau (0=bat full luc luon). Tranh GAC va CE loss tranh gradient khi fusion con nhieu luc dau train.')
    parser.add_argument('--gate_diversity_lambda', type=float, default=0.0,
                        help='Thuong do lech chuan alpha THEO PATCH trong tung mau -- chong alpha phang ve gan-hang-so ngay trong khoang [min,max] da kep. Recommend 0.01-0.05.')
    parser.add_argument('--gate_blend_learned', action='store_true',
                       help='Blend target hoc per-patch (MLP([v;t_bar])) thay text-mean tinh chung.')
    parser.add_argument('--type_branch_detach', action='store_true',
                       help='type_loss qua nhanh rieng tren stopgrad(text_cls): lam giau dieu kien gate '
                            '(type_vec) MA KHONG nhieu generation. Tach enrichment khoi interference.')
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
    parser.add_argument('--rdrop_all_pos', action='store_true',
                       help='PHA B3: tinh KL cua R-Drop tren MOI vi tri token, khong chi vi tri 0')
    parser.add_argument('--hard_margin', type=float, default=0.0,
                       help='PHA B2: trong so margin chong lang gieng (0 = tat, hanh vi cu)')
    parser.add_argument('--hard_margin_m', type=float, default=1.0,
                       help='PHA B2: bien do margin giua logit gold va doi thu manh nhat')
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
    parser.add_argument('--warmstart_from', type=str, default=None,
                        help='Init shared weights (encoder/decoder/vision_proj...) tu checkpoint T0 (partial, strict=False), gate/type init fresh, train tu epoch 0. Neo shared tai nghiem T0 -> giam break tai goc.')
    parser.add_argument('--resume_reset_epoch', action='store_true',
                       help='When resuming, reset epoch counter and early-stopping state (fresh schedule with loaded weights)')
    parser.add_argument('--reset_lr', action='store_true',
                       help='When resuming, reset LR to --lr value and reinitialize scheduler (warm restart)')
    parser.add_argument('--save_every', type=int, default=1, help='Save checkpoint every N epochs')
    parser.add_argument('--save_all_epochs', action='store_true',
                       help='Save a lightweight (model-only) checkpoint epoch_{N}.pt every epoch, '
                            'so any epoch can be evaluated later (e.g. to decouple best-checkpoint '
                            'selection from the noisy val-EM metric). ~2GB/epoch.')
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
    # Full CUDA determinism: disable Flash/MemEfficient SDPA (use math backend)
    # so cudnn.deterministic=True no longer triggers CUDNN_STATUS_NOT_INITIALIZED
    # on H100 MIG with MBart SDPA.
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.use_deterministic_algorithms(True, warn_only=True)
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
            max_a_len=args.max_answer_length,
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
                teacher_vision_processor=teacher_vision_processor,
                max_a_len=args.max_answer_length
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
    # 🔬 nhan lop dap an cho answer_head: index trong tap dap an train, -1 neu khong thuoc
    answer_cls_map = None; _n_ans_cls = 0
    if args.answer_cls_lambda > 0:
        import unicodedata as _ud
        import pandas as _pdx
        _tr = _pdx.read_csv(args.train_csv)
        _vocab = sorted({_ud.normalize('NFC', str(a)).strip().lower() for a in _tr['answer']})
        _a2i = {a: i for i, a in enumerate(_vocab)}
        _n_ans_cls = len(_vocab)
        answer_cls_map = torch.tensor(
            [_a2i.get(_ud.normalize('NFC', str(a)).strip().lower(), -1) for a in _tr['answer']],
            dtype=torch.long)
        print(f"[answer_cls] {_n_ans_cls} lop dap an | {(answer_cls_map>=0).sum().item()}/{len(_tr)} hang co nhan")


    model = DeterministicVQA(
        vision_model_name=vision_model,
        bartpho_model_name=bartpho_model,
        bartpho_revision=args.bartpho_revision,
        num_fusion_layers=num_fusion_layers,
        gca_strength=args.gca_strength,
        gca_dropout=args.gca_dropout,
        gca_dropout_types=args.gca_dropout_types,
        text_path_dropout=args.text_path_dropout,
        concat_fusion=args.concat_fusion,
        summary_token=args.summary_token,
        alpha_reg_lambda=args.alpha_reg_lambda,
        gate_layerscale_pertype=args.gate_layerscale_pertype,
        gate_layerscale_init=args.gate_layerscale_init,
        gate_layerscale_l2=args.gate_layerscale_l2,
        slot_attn=args.slot_attn,
        num_slots=args.num_slots,
        slot_init_std=args.slot_init_std,
        slot_tanh_gate=args.slot_tanh_gate,
        slot_stage=args.slot_stage,
        slot_no_type=args.slot_no_type,
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
        use_decoder_lora=args.use_decoder_lora,
        decoder_lora_r=args.decoder_lora_r,
        decoder_lora_alpha=args.decoder_lora_alpha,
        decoder_lora_dropout=args.decoder_lora_dropout,
        use_text_lora=args.use_text_lora,  # 🔥 NEW: LoRA for text encoder
        text_lora_r=args.text_lora_r,  # 🔥 NEW
        text_lora_alpha=args.text_lora_alpha,  # 🔥 NEW
        text_lora_dropout=args.text_lora_dropout,  # 🔥 NEW
        tcvg_type_emb_std=args.tcvg_type_emb_std,
        tcvg_type_emb_init=(
            torch.load(args.tcvg_type_emb_init_path, map_location='cpu', weights_only=False)['prototypes']
            if args.tcvg_type_emb_init_path else None),
        tcvg_gate_mode=args.tcvg_gate_mode,
        tcvg_two_layer=args.tcvg_two_layer,
        tcvg_norm_type_emb=args.tcvg_norm_type_emb,
        tcvg_type_null=args.tcvg_type_null,
        tcvg_type_bias=args.tcvg_type_bias,
        tcvg_type_ctx=args.tcvg_type_ctx,
        tcvg_ln_mode=args.tcvg_ln_mode,
        tcvg_attn_gate=args.tcvg_attn_gate,
        tcvg_refine_gate=args.tcvg_refine_gate,
        tcvg_proto_gate=args.tcvg_proto_gate,
        tcvg_global_scalar_gate=args.tcvg_global_scalar_gate,
        gca_box_tokens=args.gca_box_tokens,
        box_class_n=(args.box_class_n if args.box_class_lambda > 0 else 0),
        box_ground=(args.box_ground_lambda > 0 or args.box_count_lambda > 0),
        num_answer_classes=_n_ans_cls,
        gate_spatial_pertype=([float(x) for x in args.gate_spatial_pertype.split(',')] if args.gate_spatial_pertype else None),
        gate_box_content=args.gate_box_content,
        box_max_inst=args.box_max_inst,
        box_class_vocab=args.box_class_vocab,
        tcvg_spatial_blend=args.tcvg_spatial_blend,
        tcvg_dynamic_peek=args.tcvg_dynamic_peek,
        tcvg_alpha_from_gca=args.tcvg_alpha_from_gca,
        use_type_codebook=args.use_type_codebook,
        codebook_size=args.codebook_size,
        codebook_beta=args.codebook_beta,
        codebook_lambda=args.codebook_lambda,
        decoder_vision_only=args.decoder_vision_only,
        decoder_pool_vision=args.decoder_pool_vision,
        tcvg_topk=args.tcvg_topk,
        tcvg_topk_random=args.tcvg_topk_random,
        tcvg_fusion_gate=args.tcvg_fusion_gate,
        tcvg_fg_2pass=args.tcvg_fg_2pass,
        use_type_task=args.use_type_loss,       # 🔥 type head auxiliary loss only
        use_logits_bias=args.use_logits_bias,   # 🔥 type-aware logits bias (risky, separate flag)
    type_loss_weight=args.type_loss_weight,
    type_branch_detach=args.type_branch_detach,
        use_vision_gate=args.use_vision_gate,
        vision_gate_init=args.vision_gate_init,
        vision_gate_min_alpha=args.vision_gate_min_alpha,
        vision_gate_min_alpha_pertype=(
            [float(x) for x in args.vision_gate_min_alpha_pertype.split(',')]
            if args.vision_gate_min_alpha_pertype else None),
        vision_gate_max_alpha_pertype=(
            [float(x) for x in args.vision_gate_max_alpha_pertype.split(',')]
            if args.vision_gate_max_alpha_pertype else None),
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
        type_moe=args.type_moe,
        type_moe_bottleneck=args.type_moe_bottleneck,
        type_moe_soft=args.type_moe_soft,
        gate_vision_layer=args.gate_vision_layer,      # FIX: cac flag nay truoc day KHONG duoc truyen (no-op)
        vision_backbone_layer=args.vision_backbone_layer,
        gate_blend_learned=args.gate_blend_learned,
        gate_no_type_emb=args.gate_no_type_emb,
        gate_no_text_cls=args.gate_no_text_cls,
        gate_blend_vorig=args.gate_blend_vorig,
        patch_self_attn=args.patch_self_attn,
        psa_heads=args.psa_heads,
        gate_alpha_budget=args.gate_alpha_budget,
        gate_budget_init=args.gate_budget_init,
        gate_pertype_net=args.gate_pertype_net,
        gate_type_blind=args.gate_type_blind,
        type_from_gate_lambda=args.type_from_gate_lambda,
        kl_pretrained_lambda=args.kl_pretrained_lambda,
        gate_per_channel=args.gate_per_channel,
        gate_gca_residual=args.gate_gca_residual,
        gate_blend_l6=args.gate_blend_l6,
        gate_l6_fuse=args.gate_l6_fuse,
        gate_l6_fuse_bottleneck=args.gate_l6_fuse_bottleneck,
        vision_l6_enrich=args.vision_l6_enrich,
        gate_harm_lambda=args.gate_harm_lambda,
        gate_harm_protect=args.gate_harm_protect,
        gate_answer_contrastive_lambda=args.gate_answer_contrastive_lambda,
        gate_answer_contrastive_temp=args.gate_answer_contrastive_temp,
        gate_diversity_lambda=args.gate_diversity_lambda,
    ).to(device)
    
    model.gate_detach_input = args.gate_detach_input
    model.text_only_mode = False  # controlled per-epoch by --text_only_warmup_epochs

    # 🔬 type_emb_init TU DONG (2026-08-09): tinh prototype ngu nghia THAT ngay tai day, dung
    # chinh text encoder cua model (con nguyen pretrained, TRUOC freeze_pretrained/LoRA ben duoi)
    # -- khong can chay script rieng, khong can file .pt trung gian. Tong quat: chi doc cot
    # question/type tu args.train_csv, khong hardcode gia tri/taxonomy nao.
    if getattr(args, 'tcvg_type_emb_init_auto', False) and getattr(model, 'use_vision_gate', False):
        import pandas as _pd
        print(f"\n[TypeEmbInit] Tinh prototype ngu nghia tu {args.train_csv} (encoder con pretrained)...")
        _df = _pd.read_csv(args.train_csv)
        assert 'question' in _df.columns and 'type' in _df.columns, \
            "--tcvg_type_emb_init_auto can cot 'question' va 'type' trong train_csv"
        _num_types = model.vision_gating.type_embedding.num_embeddings
        _hdim = model.vision_gating.type_embedding.embedding_dim
        _protos = torch.zeros(_num_types, _hdim, device=device)
        model.encoder.eval()
        with torch.no_grad():
            for _t in range(_num_types):
                _qs = _df[_df['type'] == _t]['question'].tolist()
                if not _qs:
                    print(f"  [TypeEmbInit] type={_t}: khong co cau hoi, giu prototype = 0")
                    continue
                _sum = torch.zeros(_hdim, device=device)
                _bs = 64
                for _i in range(0, len(_qs), _bs):
                    _batch = _qs[_i:_i + _bs]
                    _enc = model.tokenizer(_batch, truncation=True, padding=True, max_length=32,
                                            return_tensors='pt').to(device)
                    _out = model.encoder(input_ids=_enc['input_ids'], attention_mask=_enc['attention_mask'])
                    _mask = _enc['attention_mask'].unsqueeze(-1).float()
                    _pooled = (_out.last_hidden_state * _mask).sum(dim=1) / _mask.sum(dim=1).clamp(min=1)
                    _sum += _pooled.sum(dim=0)
                _protos[_t] = _sum / len(_qs)
                print(f"  [TypeEmbInit] type={_t}: n={len(_qs)}  norm={_protos[_t].norm().item():.4f}")
        model.vision_gating.type_embedding.weight.data.copy_(_protos)
        model.train()
        print("[TypeEmbInit] Da khoi tao type_embedding tu prototype ngu nghia (tu dong).")

    # 🔬 WARM-START: neo shared weights tai nghiem T0 (giam break tai goc), gate/type init fresh.
    if args.warmstart_from:
        print(f"\n[WarmStart] Load shared weights tu: {args.warmstart_from}")
        _ck = torch.load(args.warmstart_from, map_location=device, weights_only=False)
        _sd = _ck['model_state_dict'] if 'model_state_dict' in _ck else _ck
        _own = model.state_dict()
        # chi load key CO trong ca hai + shape khop + KHONG phai gate/type (de fresh)
        _skip = ('vision_gating', 'type_head', 'type_branch', 'type_experts', 'l6_fuse', 'l6_enrich', 'gate_layer_proj')
        _load = {k: v for k, v in _sd.items()
                 if k in _own and _own[k].shape == v.shape and not any(s in k for s in _skip)}
        _miss = [k for k in _own if k not in _load]
        model.load_state_dict(_load, strict=False)
        print(f"[WarmStart] loaded {len(_load)}/{len(_own)} tensors (shared). fresh: {len(_miss)} (gate/type/moe). Train tu epoch 0.")
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
        unfreeze_decoder=unfreeze_decoder,
        freeze_lm_head=getattr(args, 'freeze_lm_head', False)
    )

    # 🔬 IQG: bat decoder sinh lai cau hoi tu thi giac truoc hop nhat.
    if args.iqg_lambda > 0:
        model.iqg_lambda = float(args.iqg_lambda)
        model.iqg_mask = float(args.iqg_mask)
        model.iqg_check_every = int(args.iqg_check_every)
        if args.iqg_mask <= 0:
            print("  ⚠️  IQG voi iqg_mask=0: teacher forcing se TU GIAI nhiem vu, ket qua vo nghia.")
        print(f"  🔬 IQG: decoder phai SINH lai cau hoi tu bo nho thi giac TRUOC hop nhat "
              f"(lambda={args.iqg_lambda}) — 0 tham so moi, dung chinh decoder + lm_head")

    # 🔬 QGND: dung tu vung NEO tu CAU HOI train (khong phai tu dap an).
    # Bo N token pho bien nhat (tu chuc nang: la/gi/nao/co/trong...) vi chung co o moi cau nen
    # khong mang thong tin thi giac; giu token xuat hien >= min_freq de nhan khong qua thua.
    if args.qgnd_lambda > 0:
        import collections as _co
        import pandas as _pdq
        _cnt = _co.Counter()
        _qs = _pdq.read_csv(args.train_csv).question.astype(str).tolist()
        for _q in _qs:
            _cnt.update(model.tokenizer(_q, add_special_tokens=False)['input_ids'])
        _ranked = [t for t, _ in _cnt.most_common()]
        _drop = set(_ranked[:args.qgnd_drop_top])
        _ids = [t for t, c in _cnt.items() if c >= args.qgnd_min_freq and t not in _drop]
        _sample = model.tokenizer.convert_ids_to_tokens(_ranked[:8])
        print(f"[QGND] {len(_cnt)} token trong cau hoi -> giu {len(_ids)} "
              f"(bo {args.qgnd_drop_top} pho bien nhat, vd {_sample}; nguong tan suat {args.qgnd_min_freq})")
        model.set_qgnd_vocab(_ids, lam=args.qgnd_lambda, temp=args.qgnd_temp)
        model = model.to(device)

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
    # 🔥 type_embedding tach rieng: no duoc init N(0,1) nen norm ~32, trong khi buoc AdamW
    # chi ~lr moi toa do. Do duoc: sau 40 epoch (21320 buoc) vector chi xoay 5-10 mrad
    # (0.3-0.6 do) — tuc ma loai gan nhu DONG BANG o gia tri ngau nhien luc init, va
    # "loai nao bi gating" thua huong tu init chu khong duoc hoc lai.
    # --type_emb_lr_multiplier cho phep no hoc du nhanh de thoat khoi ma ngau nhien do.
    temb_named = [(n, p) for n, p in model.named_parameters()
                  if p.requires_grad and 'vision_gating.type_embedding' in n]
    gate_named = [(n, p) for n, p in model.named_parameters()
                  if p.requires_grad and 'vision_gating' in n
                  and 'vision_gating.type_embedding' not in n]
    # 🔬 decoder LR thap (neo mem tai T0 khi warm-start): decoder troi cham -> giu gan T0 (giam break)
    #    ma van train duoc (khac freeze cung). Chi tach khi --decoder_lr_multiplier != 1.
    _dec_split = abs(args.decoder_lr_multiplier - 1.0) > 1e-9
    dec_named = [(n, p) for n, p in model.named_parameters()
                 if p.requires_grad and 'vision_gating' not in n and n.startswith('decoder.')] if _dec_split else []
    _dec_set = {id(p) for _, p in dec_named}
    other_named = [(n, p) for n, p in model.named_parameters()
                   if p.requires_grad and 'vision_gating' not in n and id(p) not in _dec_set]
    gate_lr = learning_rate * args.gate_lr_multiplier
    gate_wd = 0.0 if args.gate_no_weight_decay else weight_decay
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Optimizer] Base LR = {learning_rate:.2e} | trainable params: {total_trainable/1e6:.2f}M")
    if args.gate_lr_multiplier != 1.0 or args.gate_no_weight_decay:
        print(f"[Optimizer] Gate group ({len(gate_named)} tensors): "
              f"LR={gate_lr:.2e} (×{args.gate_lr_multiplier}), WD={gate_wd}")

    temb_lr = learning_rate * args.type_emb_lr_multiplier
    if args.type_emb_lr_multiplier != 1.0:
        print(f"[Optimizer] type_embedding group ({len(temb_named)} tensors): "
              f"LR={temb_lr:.2e} (x{args.type_emb_lr_multiplier})")
    optimizer = torch.optim.AdamW(
        [
            {'params': [p for _, p in other_named], 'lr': learning_rate, 'weight_decay': weight_decay},
            {'params': [p for _, p in gate_named],  'lr': gate_lr,       'weight_decay': gate_wd},
            {'params': [p for _, p in temb_named],  'lr': temb_lr,       'weight_decay': gate_wd},
            {'params': [p for _, p in dec_named],   'lr': learning_rate * args.decoder_lr_multiplier, 'weight_decay': weight_decay},
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

    # 🔬 region_map_lookup (tcvg_spatial_blend_region_map): tai 1 lan, dung lai moi epoch/batch.
    region_map_lookup = None
    if getattr(args, 'tcvg_spatial_blend_region_map', None):
        print(f"\n[RegionMap] Dang tai {args.tcvg_spatial_blend_region_map}...")
        region_map_lookup = pickle.load(open(args.tcvg_spatial_blend_region_map, 'rb'))
        print(f"[RegionMap] Da tai {len(region_map_lookup)} img_id co annotation COCO that.")

    # 🔬 GATE DISTILL (A3): nhan alpha oracle, sinh boi eval.py --dump_oracle_alpha tren CHINH
    # train_csv nay (DataLoader shuffle=False -> hang thu i cua .npz = hang thu i cua CSV).
    gate_distill_alpha = None
    if getattr(args, 'gate_distill_path', None) and args.gate_distill_lambda > 0:
        _z = np.load(args.gate_distill_path)
        gate_distill_alpha = torch.from_numpy(_z['alpha'].astype('float32'))
        _n_csv = len(train_dataset)
        if gate_distill_alpha.size(0) != _n_csv:
            raise SystemExit(
                f"[gate_distill] nhan co {gate_distill_alpha.size(0)} hang nhung train_csv co "
                f"{_n_csv} hang -> KHONG khop, se distill sai mau. Sinh lai nhan tren dung file "
                f"{args.train_csv}.")
        print(f"[gate_distill] Da tai {tuple(gate_distill_alpha.shape)} tu {args.gate_distill_path}, "
              f"lambda={args.gate_distill_lambda} mode={args.gate_distill_mode}")

    _gac_lambda_target = args.gate_answer_contrastive_lambda
    _gac_warmup = args.gate_answer_contrastive_warmup_epochs

    # ---- Val decoding khớp test (tùy chọn) ----------------------------------
    # Dùng lại NGUYÊN các hàm của eval.py chứ không viết lại, để val và test không
    # bao giờ trôi khỏi nhau. eval.py nằm cùng thư mục src/ nhưng 'eval' trùng tên
    # builtin nên nạp bằng đường dẫn file.
    val_decode_cfg = None
    if args.val_match_test_decoding:
        import importlib.util as _ilu
        _spec = _ilu.spec_from_file_location(
            '_vivqa_eval', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'eval.py'))
        _em = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_em)
        val_decode_cfg = {
            'num_beams': args.val_decode_beams,
            'repetition_penalty': args.val_decode_rep_penalty,
            'max_length': args.val_decode_max_length,
            'prefix_trie': _em.build_answer_trie(args.val_trie_csv, model.tokenizer),
            'valid_answers_set': _em.build_valid_answers_set(args.val_trie_csv),
            'snap_fn': _em.snap_to_valid_answer,
            'use_synonyms': True,
            'norm_fn': lambda t: _em._normalize_vn(t, True),
        }
        print(f"[val-decode] Val EM dùng đúng bộ decoding của test: beam={args.val_decode_beams} "
              f"rep_penalty={args.val_decode_rep_penalty} max_len={args.val_decode_max_length} "
              f"trie={args.val_trie_csv} +snap +synonyms, bật từ epoch {args.val_decode_from_epoch}")

    # 🔬 OGM-GE: trang thai song qua CAC epoch (ratio_ema phai lien tuc, khong reset moi epoch —
    # reset se lam he so nhay lung tung o dau moi epoch). None khi tat -> hoan toan tro.
    ogm_state = {} if (args.ogm_ge > 0 or args.gge_diag > 0 or args.gge > 0) else None
    if ogm_state is not None:
        print(f"[OGM-GE] BAT: alpha={args.ogm_ge} noise={args.ogm_ge_noise} every={args.ogm_ge_every} "
              f"ema={args.ogm_ge_ema} epoch[{args.ogm_ge_start_epoch},{args.ogm_ge_end_epoch}) "
              f"-> ham gradient RIENG flamingo_fusion. Forward va suy luan KHONG doi.")

    # 🔬 bang prior token, uoc TU train csv (khong hardcode token nao -> chay duoc tren bo khac)
    _token_prior_table = None
    if getattr(args, 'token_prior_gamma', 0.0) > 0:
        _token_prior_table = _build_token_prior(args.train_csv, model.tokenizer)
        _det = sum(1 for v in _token_prior_table.values() if v > 0.95)
        print(f"  🔬 token_prior_gamma={args.token_prior_gamma}: {len(_token_prior_table)} "
              f"(loai,vi tri), {_det} vi tri XAC DINH (p_max>0.95)")

    for epoch in range(start_epoch, stage3_epochs + 1):
        # 🔬 GAC curriculum: ramp gate_answer_contrastive_lambda tuyen tinh 0 -> target trong
        # N epoch dau, tranh GAC va CE loss tranh gradient luc fusion con nhieu (xem
        # --gate_answer_contrastive_warmup_epochs).
        if _gac_warmup and _gac_warmup > 0:
            _gac_ramp = min(1.0, epoch / float(_gac_warmup))
            model.gate_answer_contrastive_lambda = _gac_lambda_target * _gac_ramp
            if epoch <= _gac_warmup:
                print(f"[GAC curriculum] E{epoch}/{_gac_warmup}: lambda={model.gate_answer_contrastive_lambda:.4f} "
                      f"(target={_gac_lambda_target})")

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
            gate_types_no_typeloss=args.gate_types_no_typeloss,
            gate_sparsity_lambda=args.gate_sparsity_lambda,
            kl_pretrained_lambda=args.kl_pretrained_lambda,
            token_prior_gamma=args.token_prior_gamma,
            token_prior_table=_token_prior_table,
            gate_distill_alpha=gate_distill_alpha,
            gate_distill_lambda=args.gate_distill_lambda,
            gate_distill_mode=args.gate_distill_mode,
            slot_bind_lambda=args.slot_bind_lambda,
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
            rdrop_all_pos=args.rdrop_all_pos,
            use_cdw_ce=args.use_cdw_ce,
            cdw_lambda=args.cdw_lambda,
            cdw_ordinal_weights=cdw_ordinal_weights,
            hard_margin=args.hard_margin,
            hard_margin_m=args.hard_margin_m,
            box_ground_lambda=args.box_ground_lambda,
            box_count_lambda=args.box_count_lambda,
            box_count_typed=args.box_count_typed,
            box_class_lambda=args.box_class_lambda,
            answer_cls_lambda=args.answer_cls_lambda,
            answer_cls_map=answer_cls_map,
            ema_model=ema_model,
            ema_decay=args.ema_decay,
            region_map_lookup=region_map_lookup,
            ogm_ge=args.ogm_ge,
            ogm_ge_noise=args.ogm_ge_noise,
            ogm_ge_every=args.ogm_ge_every,
            ogm_ge_ema=args.ogm_ge_ema,
            ogm_ge_start_epoch=args.ogm_ge_start_epoch,
            ogm_ge_end_epoch=args.ogm_ge_end_epoch,
            ogm_state=ogm_state,
            gge_diag=args.gge_diag,
            gge=args.gge,
            gge_floor=args.gge_floor,
            gge_start_epoch=args.gge_start_epoch,
        )

        print(f"  TRAIN -> Loss: {train_metrics['loss']:.4f} | Answer: {train_metrics['answer_loss']:.4f}")
        # 🔬 CHAN DOAN GGE: nhan co rong khong. top10_mass = 0.10 nghia la trai deu (GGE = ha lr,
        # null tu gay ra); cang gan 1.0 thi tai trong so cang that.
        # 🔬 IQG: in do lon SONG. Loss dung im hoac ~0 ngay tu dau = decoder dang CHEP cau hoi
        # tu dau vao chu khong sinh tu thi giac -> can thiet ke sai, khong duoc doc ket qua.
        if args.iqg_lambda > 0:
            _i = getattr(model, 'last_iqg', None)
            if _i:
                _g=_i.get('gap_no_vision')
                _gs=f" | chenh khi BO ANH {_g:+.4f}" + ("  <<< ~0 = VO NGHIA" if _g is not None and _g<0.15 else "") if _g is not None else ""
                print(f"  IQG -> loss {_i['loss']:.4f} | token cau hoi moi mau {_i['tok_per_sample']:.1f}{_gs}")
            else:
                print("  IQG -> !! khong co so lieu: mat mat KHONG chay, dung lai")

        # 🔬 QGND: in do lon SONG. Mot mat mat dung im hoac so token duong = 0 nghia la can thiep
        # KHONG chay, va moi ket qua doc duoc luc do la vo nghia (bai hoc --hard_margin).
        if args.qgnd_lambda > 0:
            _q = getattr(model, 'last_qgnd', None)
            if _q:
                print(f"  QGND -> loss {_q['loss']:.4f} | so token duong tren mot mau "
                      f"{_q['pos_per_sample']:.2f}")
            else:
                print("  QGND -> !! khong co so lieu: mat mat KHONG chay, dung lai")

        # 🔬 GGE dang chay that: in do lon SONG cua trong so. Neu no ve ~1.0 deu thi can thiep
        # da tro (bieu thuc tu chuan hoa), va moi ket qua doc duoc luc do la vo nghia.
        if args.gge > 0 and ogm_state is not None and ogm_state.get('ggew_n'):
            _n = ogm_state.pop('ggew_n'); _s = ogm_state.pop('ggew_sum')
            print(f"  GGE -> trong so trung binh {_s/_n:.4f} (1.0 = tro hoan toan)")
        if args.gge_diag > 0 and ogm_state is not None and ogm_state.get('gge_n'):
            _n = ogm_state.pop('gge_n')
            _g = {k[4:]: ogm_state.pop(k) / _n for k in list(ogm_state) if k.startswith('gge_')}
            print(f"  GGE-diag -> w tb {_g['w_mean']:.4f} | trung vi {_g['w_med']:.4f} | "
                  f"khoi luong top10% {_g['top10_mass']:.3f} | ty le w>0.5 {_g['frac_gt_05']:.3f}")
        # 🔬 OGM-GE: in do lon SONG cua can thiep moi epoch. Bat buoc — mot he so nam im o 1.0
        # hoac mot ty so nam im o 1.00 nghia la can thiep KHONG chay, va ket qua am/0 luc do
        # khong noi len dieu gi ve gia thuyet. Da mat GPU vi khong do truoc dieu nay.
        if args.ogm_ge > 0 and ogm_state is not None:
            _rn = ogm_state.pop('ratio_raw_n', 0)
            _rs = ogm_state.pop('ratio_raw_sum', 0.0)
            _cn = ogm_state.pop('coeff_n', 0)
            _cs = ogm_state.pop('coeff_sum', 0.0)
            print(f"  OGM-GE -> ratio tb {_rs/max(_rn,1):.4f} (ema {ogm_state.get('ratio_ema', float('nan')):.4f}) "
                  f"| coeff tb {_cs/max(_cn,1):.4f} | so buoc bi ham {_cn} "
                  f"| grad_norm tb {train_metrics.get('grad_norm_mean', 0.0):.4f}")
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
            gate_types_no_typeloss=args.gate_types_no_typeloss,
            vision_dropout_rate=0.0,
            amp_dtype=amp_dtype,
            current_epoch=epoch,
            log_file=train_log_file,
            region_map_lookup=region_map_lookup,
        )

        print(f"  VAL   -> Loss: {val_metrics['loss']:.4f} | Answer: {val_metrics['answer_loss']:.4f}")

        # 🔥 Tính EM/F1 trên TOÀN BỘ val set mỗi epoch (dùng EMA model nếu có)
        _eval_model = ema_model if ema_model is not None else model
        _vcfg = val_decode_cfg if (val_decode_cfg is not None
                                   and epoch >= args.val_decode_from_epoch) else None
        full_val = evaluate_full_val(_eval_model, val_loader, model.tokenizer, device,
                                     decode_cfg=_vcfg)
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
            'learning_rate': optimizer.param_groups[0]['lr'],
            'train_grad_norm_mean': train_metrics.get('grad_norm_mean', 0.0),
            'train_grad_norm_max': train_metrics.get('grad_norm_max', 0.0),
            'train_nan_loss_steps': train_metrics.get('nan_loss_steps', 0),
            'train_nan_grad_steps': train_metrics.get('nan_grad_steps', 0),
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

        # 🔥 Per-epoch lightweight checkpoint (model-only) — for later eval of ANY epoch,
        # decoupling checkpoint selection from the noisy val-EM metric.
        if getattr(args, 'save_all_epochs', False):
            epoch_ckpt = {
                'epoch': epoch,
                'stage': stage,
                'model_state_dict': (ema_model.state_dict() if ema_model is not None else model.state_dict()),
                'train_loss': train_metrics['loss'],
                'val_loss': val_metrics['loss'],
                'args': vars(args),
            }
            epoch_path = os.path.join(output_dir, f'epoch_{epoch}.pt')
            try:
                torch.save(epoch_ckpt, epoch_path)
                print(f"  💾 Saved per-epoch checkpoint: {epoch_path}")
            except OSError as e:
                print(f"  ⚠️  Failed to save per-epoch ckpt: {e}")

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
        swa_val = evaluate_full_val(swa_model.module, val_loader, model.tokenizer, device,
                                    decode_cfg=val_decode_cfg)
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
