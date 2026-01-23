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
import json
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from collections import Counter
import csv
import matplotlib.pyplot as plt
from dataset import VQAGenDataset

from model_no_latent import DeterministicVQA

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
# UTILITIES
# ============================================================================

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=5, min_delta=0.001, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
    
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            if self.verbose:
                print(f"  📉 Validation loss improved: {self.best_loss:.4f} → {val_loss:.4f}")
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.verbose:
                print(f"  ⚠️  No improvement for {self.counter}/{self.patience} epochs")
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
    pred_tokens = prediction.lower().split()
    gt_tokens = ground_truth.lower().split()
    
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
    is_training=True, max_norm=1.0, stage=3, gradient_accumulation_steps=1
):
    """
    Run one epoch for deterministic model (no KL diagnostics needed!)
    
    Args:
        gradient_accumulation_steps: Accumulate gradients over multiple batches
                                     for effective larger batch size
    
    Returns:
        dict with metrics: loss, answer_loss
    """
    model.train() if is_training else model.eval()
    
    total_loss = 0.0
    total_answer_loss = 0.0
    num_batches = 0
    
    with torch.set_grad_enabled(is_training):
        pbar = tqdm(dataloader, desc=f"{'Train' if is_training else 'Val'} Stage {stage}")
        
        for batch_idx, batch in enumerate(pbar):
            pixel_values = batch['pixel_values'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass with mixed precision
            with autocast(enabled=(scaler is not None)):
                outputs = model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    stage=stage
                )
                
                loss = outputs.total_loss
                
                # Scale loss for gradient accumulation
                if is_training and gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps
            
            if is_training and loss is not None:
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                # Update weights after accumulating gradients
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                        optimizer.step()
                    
                    optimizer.zero_grad()
            
            # Accumulate metrics (use original loss, not scaled)
            if loss is not None:
                actual_loss = loss.item() * gradient_accumulation_steps if gradient_accumulation_steps > 1 else loss.item()
                total_loss += actual_loss
                total_answer_loss += outputs.answer_loss.item()
                num_batches += 1
                
                pbar.set_postfix({
                    'loss': f"{actual_loss:.3f}",
                    'ans': f"{outputs.answer_loss.item():.3f}"
                })
    
    if num_batches == 0:
        return {
            'loss': 0.0,
            'answer_loss': 0.0
        }
    
    return {
        'loss': total_loss / num_batches,
        'answer_loss': total_answer_loss / num_batches
    }


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
                max_length=20,
                num_beams=3  # Use beam search!
            )
            
            # Decode labels
            label_texts = []
            for label in labels:
                label_tokens = label[label != -100].cpu().tolist()
                label_text = tokenizer.decode(label_tokens, skip_special_tokens=True)
                label_texts.append(label_text)
            
            # Decode questions
            question_texts = []
            for inp in input_ids:
                question_text = tokenizer.decode(inp, skip_special_tokens=True)
                question_texts.append(question_text)
            
            # Compute metrics
            for q, pred, gt in zip(question_texts, predictions, label_texts):
                # Exact match
                em = 1.0 if pred.strip().lower() == gt.strip().lower() else 0.0
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
    
    # Model
    parser.add_argument('--dinov2_model', type=str, default='facebook/dinov2-base', help='DINOv2 model')
    parser.add_argument('--bartpho_model', type=str, default='vinai/bartpho-syllable', help='BARTpho model')
    parser.add_argument('--num_fusion_layers', type=int, default=2, help='Number of Flamingo fusion layers')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    
    # Training
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--max_norm', type=float, default=1.0, help='Gradient clipping max norm')
    parser.add_argument('--no_amp', action='store_true', help='Disable automatic mixed precision')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, 
                       help='Number of gradient accumulation steps (for effective larger batch size)')
    
    # LR scheduler & early stopping
    parser.add_argument('--scheduler', type=str, default='plateau', choices=['none', 'plateau', 'cosine'],
                       help='LR scheduler type')
    parser.add_argument('--scheduler_patience', type=int, default=3, help='Patience for ReduceLROnPlateau')
    parser.add_argument('--scheduler_factor', type=float, default=0.5, help='Factor for ReduceLROnPlateau')
    parser.add_argument('--early_stopping', action='store_true', help='Enable early stopping')
    parser.add_argument('--early_stopping_patience', type=int, default=5, help='Early stopping patience')
    
    # Freezing
    parser.add_argument('--unfreeze_encoder_layers', type=int, default=3, help='Number of text encoder layers to unfreeze')
    parser.add_argument('--freeze_decoder', action='store_true', help='Freeze decoder (default: unfrozen)')
    
    # Checkpointing
    parser.add_argument('--output_dir', type=str, default='./checkpoints_no_latent', help='Output directory for checkpoints')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--save_every', type=int, default=1, help='Save checkpoint every N epochs')
    parser.add_argument('--sample_every', type=int, default=3, help='Sample predictions every N epochs')
    
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
    
    # Set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # ========================================================================
    # CONFIG (from args)
    # ========================================================================
    
    # Data
    data_dir = args.data_dir
    batch_size = args.batch_size
    num_workers = args.num_workers
    
    # Model
    dinov2_model = args.dinov2_model
    bartpho_model = args.bartpho_model
    num_fusion_layers = args.num_fusion_layers
    
    # Training (Stage 3 ONLY)
    stage3_epochs = args.epochs
    learning_rate = args.lr
    weight_decay = args.weight_decay
    max_norm = args.max_norm
    use_amp = not args.no_amp
    
    # Freezing strategy
    unfreeze_encoder_layers = args.unfreeze_encoder_layers
    unfreeze_decoder = not args.freeze_decoder
    
    # Checkpointing
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Device] Using: {device}")
    
    print("\n" + "="*80)
    print("TRAINING CONFIGURATION")
    print("="*80)
    print(f"  Data dir: {data_dir}")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {stage3_epochs}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Weight decay: {weight_decay}")
    print(f"  Gradient clipping: {max_norm}")
    print(f"  Mixed precision: {use_amp}")
    print(f"  Fusion layers: {num_fusion_layers}")
    print(f"  Unfreeze encoder layers: {unfreeze_encoder_layers}")
    print(f"  Unfreeze decoder: {unfreeze_decoder}")
    print(f"  Output dir: {output_dir}")
    print(f"  Random seed: {args.seed}")
    print("="*80 + "\n")
    
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
        
        vision_processor = AutoProcessor.from_pretrained(dinov2_model)
        
        # Load full training dataset
        full_train_dataset = VQAGenDataset(
            csv_path=args.train_csv,
            image_folder=args.image_dir,
            vision_processor=vision_processor,
            tokenizer_name=bartpho_model
        )
        
        # Check if val_csv provided
        if args.val_csv:
            print(f"[Data] Using provided validation CSV: {args.val_csv}")
            val_dataset = VQAGenDataset(
                csv_path=args.val_csv,
                image_folder=args.image_dir,
                vision_processor=vision_processor,
                tokenizer_name=bartpho_model
            )
            train_dataset = full_train_dataset
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
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
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
        dinov2_model_name=dinov2_model,
        bartpho_model_name=bartpho_model,
        num_fusion_layers=num_fusion_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        gradient_checkpointing=not args.no_gradient_checkpointing
    ).to(device)
    
    model.freeze_pretrained(
        unfreeze_encoder_layers=unfreeze_encoder_layers,
        unfreeze_decoder=unfreeze_decoder
    )
    
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"[Model] Total params: {total_params:.1f}M")
    print(f"[Model] Trainable params: {trainable_params:.1f}M ({trainable_params/total_params*100:.1f}%)")
    
    # ========================================================================
    # OPTIMIZER
    # ========================================================================
    
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Mixed precision scaler (use new API to avoid deprecation warning)
    if use_amp:
        try:
            from torch.amp import GradScaler as NewGradScaler
            scaler = NewGradScaler('cuda')
        except (ImportError, AttributeError):
            # Fallback to old API for older PyTorch versions
            scaler = GradScaler()
    else:
        scaler = None
    
    # 🔥 LR Scheduler
    scheduler = None
    if args.scheduler == 'plateau':
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=args.scheduler_factor,
            patience=args.scheduler_patience,
            min_lr=1e-7
        )
        print(f"[Scheduler] ReduceLROnPlateau (patience={args.scheduler_patience}, factor={args.scheduler_factor})")
    elif args.scheduler == 'cosine':
        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
        print(f"[Scheduler] CosineAnnealingLR (T_max={args.epochs})")
    else:
        print(f"[Scheduler] None (fixed LR)")
    
    # 🔥 Early Stopping
    early_stopping = None
    if args.early_stopping:
        early_stopping = EarlyStopping(
            patience=args.early_stopping_patience,
            min_delta=0.001,
            verbose=True
        )
        print(f"[Early Stopping] Enabled (patience={args.early_stopping_patience})")
    
    # Resume from checkpoint if specified
    start_epoch = 1
    best_val_loss = float('inf')
    
    if args.resume:
        print(f"\n[Resume] Loading checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        if scaler and 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"[Resume] Resuming from epoch {start_epoch}, best val loss: {best_val_loss:.4f}")
    
    # ========================================================================
    # STAGE 3: END-TO-END TRAINING (NO STAGES 1/2!)
    # ========================================================================
    
    print("\n" + "="*80)
    print("STAGE 3: END-TO-END TRAINING (No Latent/KL Warmup)")
    print("="*80)
    print(f"  • Epochs: {stage3_epochs} (starting from {start_epoch})")
    print(f"  • Learning rate: {learning_rate}")
    print(f"  • Focus: Direct optimization for accuracy")
    print("="*80 + "\n")
    
    stage = 3
    
    # Training history for plots and CSV
    training_history = []
    
    for epoch in range(start_epoch, stage3_epochs + 1):
        print(f"\n[Stage 3 | Epoch {epoch}/{stage3_epochs}]")
        
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
            gradient_accumulation_steps=args.gradient_accumulation_steps
        )
        
        print(f"  TRAIN -> Loss: {train_metrics['loss']:.4f} | Answer: {train_metrics['answer_loss']:.4f}")
        
        # Validation
        val_metrics = run_one_epoch_deterministic(
            model=model,
            dataloader=val_loader,
            optimizer=None,
            scaler=None,
            device=device,
            is_training=False,
            stage=stage
        )
        
        print(f"  VAL   -> Loss: {val_metrics['loss']:.4f} | Answer: {val_metrics['answer_loss']:.4f}")
        
        # Track metrics in history
        epoch_metrics = {
            'epoch': epoch,
            'train_loss': train_metrics['loss'],
            'train_answer_loss': train_metrics['answer_loss'],
            'val_loss': val_metrics['loss'],
            'val_answer_loss': val_metrics['answer_loss'],
            'learning_rate': optimizer.param_groups[0]['lr']
        }
        
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
        
        # 🔥 LR Scheduler step
        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_metrics['loss'])
            else:
                scheduler.step()
            
            # Print current LR
            current_lr = optimizer.param_groups[0]['lr']
            print(f"  📊 Learning Rate: {current_lr:.2e}")
        
        # Sample predictions every N epochs
        if epoch % args.sample_every == 0:
            print("\n  [Sample Predictions with Metrics]")
            samples, sample_metrics = sample_predictions(
                model, val_loader, model.tokenizer, device, num_samples=5, compute_metrics=True
            )
            
            if sample_metrics:
                print(f"    📊 Metrics: EM={sample_metrics['exact_match']:.1f}% | F1={sample_metrics['f1_score']:.1f}%", end="")
                if ROUGE_AVAILABLE:
                    print(f" | ROUGE-1={sample_metrics['rouge1']:.1f}% | ROUGE-L={sample_metrics['rougeL']:.1f}%")
                else:
                    print()
                
                # Add to epoch metrics
                epoch_metrics['exact_match'] = sample_metrics['exact_match']
                epoch_metrics['f1_score'] = sample_metrics['f1_score']
                if ROUGE_AVAILABLE:
                    epoch_metrics['rouge1'] = sample_metrics['rouge1']
                    epoch_metrics['rougeL'] = sample_metrics['rougeL']
                
                # 🔥 Log sample metrics to W&B
                if args.use_wandb:
                    wandb_log.update({
                        'val/exact_match': sample_metrics['exact_match'],
                        'val/f1_score': sample_metrics['f1_score']
                    })
                    if ROUGE_AVAILABLE:
                        wandb_log.update({
                            'val/rouge1': sample_metrics['rouge1'],
                            'val/rougeL': sample_metrics['rougeL']
                        })
            
            for i, s in enumerate(samples, 1):
                em_symbol = "✓" if s['exact_match'] == 1.0 else "✗"
                rouge_str = f" | R1={s['rouge1']:.2f} | RL={s['rougeL']:.2f}" if ROUGE_AVAILABLE else ""
                print(f"    {i}. {em_symbol} Q: {s['question']}")
                print(f"       Pred: {s['prediction']} | GT: {s['ground_truth']}")
                print(f"       Metrics: F1={s['f1_score']:.2f}{rouge_str}")
        
        # 🔥 Send W&B log
        if args.use_wandb:
            wandb.log(wandb_log)
        
        # Add to training history
        training_history.append(epoch_metrics)
        
        # 🔥 Early stopping check
        if early_stopping is not None:
            if early_stopping(val_metrics['loss']):
                print(f"\n🛑 Early stopping at epoch {epoch}!")
                break
        
        # Save best model checkpoint
        is_best = val_metrics['loss'] < best_val_loss
        if is_best:
            best_val_loss = val_metrics['loss']
            print(f"  ✅ NEW BEST! Saving checkpoint...")
            
            checkpoint = {
                'epoch': epoch,
                'stage': stage,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_metrics['loss'],
                'val_loss': val_metrics['loss'],
                'best_val_loss': best_val_loss,
                'args': vars(args)
            }
            
            if scaler is not None:
                checkpoint['scaler_state_dict'] = scaler.state_dict()
            
            if scheduler is not None:
                checkpoint['scheduler_state_dict'] = scheduler.state_dict()
            
            best_path = os.path.join(output_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            print(f"  💾 Saved to: {best_path}")
        
        # 🔥 ALWAYS save last model (for resume)
        last_checkpoint = {
            'epoch': epoch,
            'stage': stage,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_metrics['loss'],
            'val_loss': val_metrics['loss'],
            'best_val_loss': best_val_loss,
            'training_history': training_history,  # Include history for resume
            'args': vars(args)
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
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"  Best validation loss: {best_val_loss:.4f}")
    print(f"  Checkpoints saved to: {output_dir}")
    print("="*80)
    
    # 🔥 Finish W&B run
    if args.use_wandb:
        wandb.finish()
        print("\n[W&B] Run finished and synced!")


if __name__ == '__main__':
    main()
