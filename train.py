#!/usr/bin/env python3
"""
Full 3-Stage Training Pipeline (Continuous)
============================================

Run all 3 stages in a single continuous training session
Automatically switches stages based on epoch milestones

Usage:
    python train.py --csv_path <path> --image_folder <path>
"""

import torch
import os
import csv
import matplotlib.pyplot as plt
import pandas as pd
from dataclasses import dataclass
from train_utils import (
    FixedTrainConfig, set_seed, run_one_epoch
)
from model import FixedLatentReasoningVQA, TeacherEvaluator, TrainingCurriculum
from dataset import VQAGenDataset
from transformers import AutoImageProcessor
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler
from tqdm import tqdm


def get_current_stage(epoch: int, stage1_epochs: int, stage2_epochs: int):
    """Determine current stage based on epoch number"""
    if epoch < stage1_epochs:
        return 1
    elif epoch < stage1_epochs + stage2_epochs:
        return 2
    else:
        return 3


def plot_training_curves(csv_path: str, save_dir: str):
    """
    Plot training curves from CSV log
    
    Creates 4 subplots:
    1. Total Loss (train vs val)
    2. Answer Loss (train vs val)
    3. KL Loss (train vs val with raw KL)
    4. Overfitting Ratio + KL Weight
    """
    print(f"\n[PLOTTING] Loading training log from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Training Curves - ViVQA Latent Reasoning', fontsize=16, fontweight='bold')
    
    # Plot 1: Total Loss
    ax1 = axes[0, 0]
    ax1.plot(df['epoch'], df['train_total'], 'b-', label='Train Total', linewidth=2)
    ax1.plot(df['epoch'], df['val_total'], 'r-', label='Val Total', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Total Loss', fontsize=12)
    ax1.set_title('Total Loss (Answer + KL + Ortho + Teacher)', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Add stage boundaries
    for stage_name, stage_epoch in zip(['Stage 1→2', 'Stage 2→3'], 
                                       df[df['stage'].diff() != 0]['epoch'].tolist()):
        ax1.axvline(x=stage_epoch, color='gray', linestyle='--', alpha=0.5, label=stage_name)
    
    # Plot 2: Answer Loss (more important!)
    ax2 = axes[0, 1]
    ax2.plot(df['epoch'], df['train_answer'], 'b-', label='Train Answer', linewidth=2)
    ax2.plot(df['epoch'], df['val_answer'], 'r-', label='Val Answer', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Answer Loss', fontsize=12)
    ax2.set_title('Answer Loss (Without Regularization)', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Highlight best epoch
    best_epoch = df.loc[df['val_answer'].idxmin(), 'epoch']
    best_val = df['val_answer'].min()
    ax2.scatter([best_epoch], [best_val], color='red', s=100, marker='*', 
                label=f'Best: Epoch {best_epoch}', zorder=5)
    ax2.legend(loc='upper right', fontsize=10)
    
    # Plot 3: KL Loss (with raw KL)
    ax3 = axes[1, 0]
    ax3.plot(df['epoch'], df['train_kl'], 'b-', label='Train KL (after free bits)', linewidth=2)
    ax3.plot(df['epoch'], df['val_kl'], 'r-', label='Val KL (after free bits)', linewidth=2)
    if 'train_kl_raw' in df.columns:
        ax3.plot(df['epoch'], df['train_kl_raw'], 'b--', label='Train KL (raw)', linewidth=1, alpha=0.7)
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('KL Loss', fontsize=12)
    ax3.set_title('KL Divergence (Raw vs After Free Bits)', fontsize=14, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # Add healthy KL range
    ax3.axhspan(0.03, 0.08, alpha=0.2, color='green', label='Healthy KL Range')
    ax3.legend(loc='upper right', fontsize=10)
    
    # Plot 4: Overfitting Ratio + KL Weight
    ax4 = axes[1, 1]
    ax4_twin = ax4.twinx()
    
    # Overfitting ratio
    line1 = ax4.plot(df['epoch'], df['overfitting_ratio'], 'purple', 
                     label='Overfitting Ratio (Val/Train)', linewidth=2)
    ax4.axhline(y=2.0, color='orange', linestyle='--', alpha=0.5, label='Warning (2.0x)')
    ax4.axhline(y=2.5, color='red', linestyle='--', alpha=0.5, label='High (2.5x)')
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Overfitting Ratio', fontsize=12, color='purple')
    ax4.tick_params(axis='y', labelcolor='purple')
    
    # KL weight
    line2 = ax4_twin.plot(df['epoch'], df['kl_weight'], 'green', 
                          label='KL Weight', linewidth=2, linestyle='-.')
    ax4_twin.set_ylabel('KL Weight', fontsize=12, color='green')
    ax4_twin.tick_params(axis='y', labelcolor='green')
    
    ax4.set_title('Overfitting Monitor + KL Weight Schedule', fontsize=14, fontweight='bold')
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines] + ['Warning (2.0x)', 'High (2.5x)']
    ax4.legend(lines + [ax4.lines[1], ax4.lines[2]], labels, loc='upper left', fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    # Add stage annotations
    stages = df['stage'].unique()
    for stage in stages:
        stage_df = df[df['stage'] == stage]
        if len(stage_df) > 0:
            mid_epoch = stage_df['epoch'].iloc[len(stage_df)//2]
            ax4.text(mid_epoch, ax4.get_ylim()[1] * 0.95, 
                    f'Stage {stage}', 
                    ha='center', va='top', fontsize=10, 
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save plot
    plot_path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"[PLOTTING] ✅ Saved training curves to {plot_path}")
    
    # Also save as PDF for papers
    pdf_path = os.path.join(save_dir, 'training_curves.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    print(f"[PLOTTING] ✅ Saved PDF version to {pdf_path}")
    
    plt.close()
    
    # Print summary statistics
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    print(f"Total epochs: {len(df)}")
    print(f"Best val answer loss: {df['val_answer'].min():.4f} (Epoch {df.loc[df['val_answer'].idxmin(), 'epoch']})")
    print(f"Best val total loss: {df['val_total'].min():.4f} (Epoch {df.loc[df['val_total'].idxmin(), 'epoch']})")
    print(f"Final overfitting ratio: {df['overfitting_ratio'].iloc[-1]:.2f}x")
    if 'train_kl_raw' in df.columns:
        print(f"Final KL raw: {df['train_kl_raw'].iloc[-1]:.4f}")
    print(f"Final KL after: {df['train_kl'].iloc[-1]:.4f}")
    print("="*80 + "\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Run full 3-stage training in one session")
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--image_folder", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--teacher_type", type=str, default="rule_based", 
                       choices=["rule_based", "vlm"])
    parser.add_argument("--stage1_epochs", type=int, default=0)  # 🚨 SKIP Stage 1 - too restrictive
    parser.add_argument("--stage2_epochs", type=int, default=15)  # Extended warmup
    parser.add_argument("--stage3_epochs", type=int, default=20)
    parser.add_argument("--num_reasoning_samples", type=int, default=3)
    parser.add_argument("--max_kl_weight", type=float, default=0.15,
                       help="� CRITICAL: 0.6→0.15 (KL raw=0.223 too high! Target: 0.03-0.08)")
    parser.add_argument("--early_stopping_patience", type=int, default=5,
                       help="Stop if val loss doesn't improve for N epochs")
    
    args = parser.parse_args()
    
    # Configuration
    cfg = FixedTrainConfig()
    cfg.csv_path = args.csv_path
    cfg.image_folder = args.image_folder
    cfg.batch_size = args.batch_size
    cfg.teacher_type = args.teacher_type
    cfg.num_reasoning_samples = args.num_reasoning_samples
    cfg.use_teacher = True  # 🚨 FIX: Enable teacher for Stage 3!
    
    # Add missing attributes
    cfg.learning_rate = cfg.base_lr  # Add learning_rate alias
    
    # Total epochs
    total_epochs = args.stage1_epochs + args.stage2_epochs + args.stage3_epochs
    stage1_end = args.stage1_epochs
    stage2_end = args.stage1_epochs + args.stage2_epochs
    
    print("="*80)
    print("CONTINUOUS 3-STAGE TRAINING")
    print("="*80)
    print(f"\nStage boundaries:")
    print(f"  Stage 1 (Baseline): Epochs 0-{stage1_end-1}")
    print(f"  Stage 2 (Warmup):   Epochs {stage1_end}-{stage2_end-1}")
    print(f"  Stage 3 (Full):     Epochs {stage2_end}-{total_epochs-1}")
    print(f"  TOTAL: {total_epochs} epochs")
    print(f"\nKL weight config:")
    print(f"  Max KL weight: {args.max_kl_weight} (effective = {args.max_kl_weight * 0.01:.3f} due to 0.01 factor in loss)")
    print(f"  Stage 1: KL weight = 0.0")
    print(f"  Stage 2: KL weight = 0.0 → {args.max_kl_weight} (linear warmup)")
    print(f"  Stage 3: KL weight = {args.max_kl_weight}")
    print("="*80 + "\n")
    
    # Setup
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Model
    print("[1/6] Loading model...")
    model = FixedLatentReasoningVQA(
        num_reasoning_tokens=cfg.num_reasoning_tokens,
        latent_dim=cfg.latent_dim,
        num_reasoning_layers=cfg.num_reasoning_layers,
        num_fusion_layers=cfg.num_fusion_layers,
        free_bits=cfg.free_bits,
        ortho_weight=cfg.ortho_weight,
        token_dropout_prob=cfg.token_dropout_prob,
        gradient_checkpointing=True
    )
    
    # Freeze with decoder unfrozen (will handle per-stage later)
    model.freeze_pretrained(unfreeze_encoder_layers=3, unfreeze_decoder=True)
    model = model.to(device)
    
    # Dataset
    print("\n[2/6] Loading dataset...")
    vision_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
    full_dataset = VQAGenDataset(
        csv_path=cfg.csv_path,
        image_folder=cfg.image_folder,
        vision_processor=vision_processor,
        tokenizer_name='vinai/bartpho-syllable',  # Pass name, not object
        max_q_len=32,  # Same as train.py
        max_a_len=32   # Same as train.py
    )
    
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # 🚨 FIXED: Add generator for seeded shuffle
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.batch_size, 
        shuffle=True,
        generator=torch.Generator().manual_seed(42)  # Reproducible shuffle
    )
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Teacher (initialize once, use in Stage 3)
    teacher_evaluator = None
    if args.teacher_type:
        print(f"\n[3/6] Setting up {args.teacher_type} teacher...")
        teacher_evaluator = TeacherEvaluator(
            teacher_type=args.teacher_type,
            device=device,
            tokenizer=model.tokenizer  # Use model's tokenizer
        )
    else:
        print("\n[3/6] No teacher (teacher_type not specified)")
    
    # Optimizer & Scheduler - ✅ GROUPED LR (critical!)
    print("\n[4/6] Setting up optimizer with grouped LR...")
    
    # ✅ Group parameters by component
    fusion_params = []
    decoder_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'flamingo_fusion' in name:
            fusion_params.append(param)
        elif 'decoder' in name or 'lm_head' in name:
            decoder_params.append(param)
        else:
            other_params.append(param)
    
    param_groups = [
        {'params': fusion_params, 'lr': cfg.fusion_lr, 'name': 'fusion'},
        {'params': decoder_params, 'lr': cfg.decoder_lr, 'name': 'decoder'},
        {'params': other_params, 'lr': cfg.base_lr, 'name': 'other'}
    ]
    
    print(f"  Fusion params: {len(fusion_params)} (LR={cfg.fusion_lr:.2e})")
    print(f"  Decoder params: {len(decoder_params)} (LR={cfg.decoder_lr:.2e})")
    print(f"  Other params: {len(other_params)} (LR={cfg.base_lr:.2e})")
    
    optimizer = AdamW(param_groups, weight_decay=cfg.weight_decay)
    
    scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=total_epochs,
        eta_min=1e-6  # 🔥 FIX: Add minimum LR to prevent oscillation at high overfit
    )
    scaler = GradScaler(enabled=cfg.use_amp)
    
    # 🚨 CRITICAL FIX: Curriculum với epoch-based KL warmup (không phải batch-based!)
    curriculum = TrainingCurriculum(
        total_steps_per_stage=args.stage2_epochs,  # ✅ FIXED: Số EPOCHS (không phải batches!)
        max_kl_weight=args.max_kl_weight  # Tunable KL weight
    )
    
    # Training loop
    print("\n[5/6] Starting continuous training...")
    print("="*80 + "\n")
    
    best_val_loss = float('inf')
    patience_counter = 0  # 🚨 FIX: Early stopping để tránh overfitting
    best_val_answer_loss = float('inf')  # 🚨 NEW: Track answer-only loss (more reliable than total)
    
    # 🚨 NEW: CSV logging for training curves
    csv_log_path = os.path.join(cfg.save_dir, 'training_log.csv')
    # Ensure save directory exists before opening the log file
    os.makedirs(cfg.save_dir, exist_ok=True)
    try:
        csv_file = open(csv_log_path, 'w', newline='')
    except Exception as e:
        raise RuntimeError(f"Failed to open CSV log file at {csv_log_path}: {e}")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        'epoch', 'stage', 
        'train_total', 'train_answer', 'train_kl', 'train_kl_raw', 'train_ortho', 'train_teacher',
        'val_total', 'val_answer', 'val_kl', 'val_kl_raw', 'val_ortho',
        'learning_rate', 'kl_weight', 'effective_kl_weight',
        'overfitting_ratio', 'answer_gap',
        'best_val_answer', 'patience',
        'text_gate'  # 🚀 NEW: Track text gate value
    ])
    print(f"[LOGGING] Training log will be saved to: {csv_log_path}")
    print(f"[LOGGING] Training curves will be plotted after each epoch\n")
    
    for epoch in range(total_epochs):
        # Determine current stage
        current_stage = get_current_stage(epoch, args.stage1_epochs, args.stage2_epochs)
        
        # 🚨 CRITICAL: Adjust learning rates based on stage (not hard freeze!)
        # Stage 1: LR=0 for decoder (no update but gradient flows)
        # Stage 2+: Normal LR for decoder
        if current_stage == 1:
            # Set decoder LR to 0 (no weight update but allows gradient flow)
            for param_group in optimizer.param_groups:
                if param_group['name'] == 'decoder':
                    param_group['lr'] = 0.0
        else:
            # Restore decoder LR for Stage 2+
            for param_group in optimizer.param_groups:
                if param_group['name'] == 'decoder':
                    param_group['lr'] = cfg.decoder_lr * scheduler.get_last_lr()[0] / cfg.base_lr
        
        # Stage transition announcements
        if epoch == 0:
            if current_stage == 1:
                print("\n" + "="*80)
                print("⚠️  STAGE 1: BASELINE (DEPRECATED)")
                print("⚠️  WARNING: Stage 1 is deprecated! Direct Stage 2 start recommended.")
                print("    Reason: Decoder frozen (LR=0) is too restrictive, no benefit")
                print("    Better: Skip to Stage 2 with freeze vision strategy")
                print("🔒 Decoder LR=0 (gradient flows, no update)")
                print("="*80 + "\n")
            else:
                # Direct to Stage 2
                print("\n" + "="*80)
                print("🟡 STAGE 2: WARMUP (Reasoning KL Warmup)")
                print("🎯 Strategy: Keep vision encoder FROZEN (DINOv2 + small dataset)")
                print("="*80 + "\n")
                curriculum.warmup_epochs = 0
        elif epoch == stage1_end and stage1_end > 0:
            print("\n" + "="*80)
            print("🟡 STAGE 2: WARMUP (Reasoning KL Warmup)")
            print("🎯 Strategy: Keep vision encoder FROZEN (DINOv2 + small dataset)")
            print("="*80 + "\n")
            curriculum.warmup_epochs = 0  # Track warmup progress
        # ❌ REMOVED: Vision unfreezing (causes gradient explosion with DINOv2)
        # elif epoch == stage1_end + 3:
        #     for param in model.vision_encoder.parameters():
        #         param.requires_grad = True
        # 👉 Vision stays FROZEN throughout training (correct for 11-15K samples)
        elif epoch == stage2_end:
            print("\n" + "="*80)
            print("🟢 STAGE 3: FULL (Complete + Teacher)")
            print("="*80 + "\n")
        
        # 🚨 FIXED: Epoch-based KL warmup (Stage 2)
        # Note: curriculum.current_step not needed - pass epoch_progress directly!
        if current_stage == 2:
            epoch_in_stage2 = epoch - stage1_end
            curriculum.warmup_epochs = epoch_in_stage2
        else:
            epoch_in_stage2 = 0  # For Stage 1 or 3
        
        # Determine if teacher should be used
        use_teacher_this_epoch = (current_stage == 3)
        
        print(f"EPOCH {epoch+1}/{total_epochs} (Stage {current_stage})")
        print("="*80)
        
        # Train
        train_losses = run_one_epoch(
            model, train_loader, optimizer, scaler, device, cfg,
            curriculum, current_stage,
            teacher_evaluator=teacher_evaluator if use_teacher_this_epoch else None,
            scheduler=scheduler,
            train=True
        )
        
        # Validation
        with torch.no_grad():
            val_losses = run_one_epoch(
                model, val_loader, optimizer, scaler, device, cfg,
                curriculum, current_stage,
                teacher_evaluator=None,
                train=False
            )
        
        # Logging
        current_lr = scheduler.get_last_lr()[0]
        kl_weight = curriculum.get_kl_weight(current_stage, epoch_progress=epoch_in_stage2 / args.stage2_epochs)
        
        # 🚨 FIX: Hiển thị effective KL weight (×0.2, không phải ×0.03!)
        effective_kl = kl_weight * 0.2  # Match model.py loss calculation
        
        # 🚨 NEW: Monitor answer-only loss (without KL regularization)
        val_answer_only = val_losses['answer']
        train_answer_only = train_losses['answer']
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train - Total: {train_losses['total']:.4f}, Answer: {train_losses['answer']:.4f}, "
              f"KL: {train_losses['kl']:.4f}, Teacher: {train_losses['teacher']:.4f}")
        print(f"  Val   - Total: {val_losses['total']:.4f}, Answer: {val_losses['answer']:.4f}, "
              f"KL: {val_losses['kl']:.4f}")
        
        # 🚀 NEW: Show text gate value
        text_gate_value = getattr(model, 'last_text_gate', 0.0)
        print(f"  LR: {current_lr:.6f}, KL weight: {effective_kl:.4f} (raw={kl_weight:.1f}), Stage: {current_stage}, Text Gate: {text_gate_value:.4f}")
        
        # 🚨 FIXED: Monitor overfitting với edge case handling
        # If train loss too small → ratio explodes (misleading!)
        if train_losses['total'] < 1e-4:
            overfitting_ratio = 1.0  # Too small to compare reliably
            print(f"  ⚠️ Train loss too small ({train_losses['total']:.6f}) - skipping overfitting check")
        else:
            overfitting_ratio = val_losses['total'] / train_losses['total']
        
        answer_gap = val_answer_only / train_answer_only if train_answer_only > 1e-4 else 1.0
        print(f"  📊 Overfitting: {overfitting_ratio:.2f}x total, {answer_gap:.2f}x answer-only", end="")
        
        # 🔥 NEW: Adaptive LR reduction on high overfitting
        if answer_gap > 2.5 and current_stage == 3:
            # Halve LR when answer overfitting is severe
            for param_group in optimizer.param_groups:
                old_lr = param_group['lr']
                param_group['lr'] *= 0.5
            new_lr = optimizer.param_groups[0]['lr']
            print(f" ⚠️  HIGH! 🔥 LR reduced: {old_lr:.2e} → {new_lr:.2e}")
        elif overfitting_ratio > 2.5:
            print(" ⚠️  HIGH!")
        elif overfitting_ratio > 2.0:
            print(" 🟡 MODERATE")
        else:
            print(" ✅ OK")
        
        # 🚨 NEW: Log to CSV
        text_gate_value = getattr(model, 'last_text_gate', 0.0)
        csv_writer.writerow([
            epoch + 1,
            current_stage,
            train_losses['total'],
            train_losses['answer'],
            train_losses['kl'],
            train_losses.get('kl_raw', train_losses['kl']),
            train_losses['ortho'],
            train_losses['teacher'],
            val_losses['total'],
            val_losses['answer'],
            val_losses['kl'],
            val_losses.get('kl_raw', val_losses['kl']),
            val_losses['ortho'],
            current_lr,
            kl_weight,
            effective_kl,
            overfitting_ratio,
            answer_gap,
            best_val_answer_loss,
            patience_counter,
            text_gate_value  # 🚀 NEW: Text gate value
        ])
        csv_file.flush()  # Write immediately
        
        # Stage 2 warmup progress
        if current_stage == 2:
            warmup_pct = (curriculum.warmup_epochs / args.stage2_epochs) * 100
            print(f"  📈 Stage 2 Warmup: {curriculum.warmup_epochs}/{args.stage2_epochs} epochs ({warmup_pct:.1f}%)")
        
        # 🚨 FIXED: KL diagnostics using kl_raw from return dict
        if current_stage >= 2:
            kl_raw = train_losses.get('kl_raw', train_losses['kl'])  # Fallback if not available
            kl_after = train_losses['kl']
            penalty_reduction = ((kl_raw - kl_after) / kl_raw * 100) if kl_raw > 1e-6 else 0
            
            print(f"  🔍 KL Diagnostics: raw={kl_raw:.4f}, after_free_bits={kl_after:.4f}, penalty_reduction={penalty_reduction:.1f}%")
            
            # 🚨 EMPIRICAL UPDATE: Health checks for free_bits=0.005
            # Observed: KLr = 0.02-0.024 (lower than theoretical 0.05-0.08!)
            if kl_after == 0 and kl_raw > 0.005:
                print(f"     ⚠️  FREE BITS TOO HIGH! All KL becomes free (kl_raw={kl_raw:.3f}). Reduce from 0.005!")
            elif kl_raw < 0.01:
                print(f"     ⚠️  KL COLLAPSE! raw < 0.01. Increase KL weight!")
            elif kl_raw < 0.015 and epoch_in_stage2 > 10:
                print(f"     🟡 KL raw low (<0.015) after 10 epochs - monitor for collapse")
            elif kl_raw > 0.10:
                print(f"     ⚠️  KL TOO HIGH! raw > 0.10. Risk over-regularization - reduce KL weight!")
            elif penalty_reduction < 10:
                print(f"     🟡 Free bits tight (<{penalty_reduction:.0f}% reduction). Consider increasing to 0.008.")
            elif penalty_reduction > 50:
                print(f"     ⚠️  Free bits generous (>{penalty_reduction:.0f}% reduction). Reduce from 0.005 to 0.003!")
            elif 0.02 <= kl_raw <= 0.04 and 15 <= penalty_reduction <= 35:
                print(f"     ✅ KL healthy range! (target: 0.03-0.08, penalty: 20-40%)")
        
        # 🚨 NEW: Teacher loss diagnostics (Stage 3 only)
        if current_stage == 3 and train_losses['teacher'] > 0:
            teacher_loss_val = train_losses['teacher']
            # Analyze teacher contribution to total loss
            teacher_contribution = (teacher_loss_val * cfg.teacher_weight) / train_losses['total'] * 100
            
            print(f"  🎓 Teacher Diagnostics: loss={teacher_loss_val:.4f}, weight={cfg.teacher_weight}, contribution={teacher_contribution:.1f}%")
            
            # Adaptive suggestions (NOT automatic - manual tuning recommended)
            if teacher_loss_val < 0.05:
                print(f"     🟡 Teacher loss weak (<0.05). Contribution only {teacher_contribution:.1f}%.")
                print(f"        💡 Suggestion: Increase teacher_weight from {cfg.teacher_weight} to 0.5 for stronger signal.")
            elif teacher_loss_val > 0.5:
                print(f"     ⚠️  Teacher loss strong (>0.5). Contribution {teacher_contribution:.1f}% may dominate!")
                print(f"        💡 Suggestion: Decrease teacher_weight from {cfg.teacher_weight} to 0.15 to avoid over-reliance.")
            elif 0.05 <= teacher_loss_val <= 0.2 and 3 <= teacher_contribution <= 10:
                print(f"     ✅ Teacher healthy! (loss: 0.05-0.2, contribution: 3-10%)")
        
        # Prepare checkpoint dict
        os.makedirs(cfg.save_dir, exist_ok=True)
        checkpoint = {
            'epoch': epoch + 1,
            'stage': current_stage,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
            'config': cfg.__dict__
        }
        
        # Save best model (always check and update)
        # 🚨 NEW: Use answer-only loss for best model (more stable than total)
        current_val_answer = val_losses['answer']
        
        if current_val_answer < best_val_answer_loss:
            best_val_answer_loss = current_val_answer
            best_val_loss = val_losses['total']  # Also track total for logging
            patience_counter = 0  # 🚨 FIX: Reset patience
            best_path = os.path.join(cfg.save_dir, "best.pt")
            torch.save(checkpoint, best_path)
            print(f"  ✅ New best model saved! (val_answer: {best_val_answer_loss:.4f}, val_total: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"  ⏳ No improvement ({patience_counter}/{args.early_stopping_patience}) - best_answer: {best_val_answer_loss:.4f}")
            
            # Early stopping
            if patience_counter >= args.early_stopping_patience:
                print("\n" + "="*80)
                print(f"🛑 EARLY STOPPING: Val answer loss hasn't improved for {args.early_stopping_patience} epochs")
                print(f"   Best val answer loss: {best_val_answer_loss:.4f}")
                print(f"   Current val answer loss: {current_val_answer:.4f}")
                print("="*80)
                break
        
        # Save last checkpoint (overwrite each epoch to save disk space)
        last_path = os.path.join(cfg.save_dir, "last.pt")
        torch.save(checkpoint, last_path)
        
        # Sample predictions every 5 epochs
        if (epoch + 1) % 5 == 0:
            print("\n" + "="*80)
            print(f"📝 SAMPLE PREDICTIONS (Epoch {epoch+1}, Stage {current_stage})")
            print("="*80)
            model.eval()
            with torch.no_grad():
                # Get 3 random samples from validation set
                import random
                sample_indices = random.sample(range(len(val_loader.dataset)), min(3, len(val_loader.dataset)))
                
                for i, idx in enumerate(sample_indices):
                    # Get sample from dataset
                    sample = val_loader.dataset[idx]
                    pixel_values = sample[0].unsqueeze(0).to(device)
                    input_ids = sample[1].unsqueeze(0).to(device)
                    attention_mask = sample[2].unsqueeze(0).to(device)
                    labels = sample[3].unsqueeze(0)
                    
                    # Get ground truth
                    gt_tokens = labels[0][labels[0] != -100]
                    ground_truth = model.tokenizer.decode(gt_tokens, skip_special_tokens=True)
                    
                    # Get question
                    question = model.tokenizer.decode(input_ids[0], skip_special_tokens=True)
                    
                    # Forward pass to get reasoning info
                    outputs = model(
                        pixel_values=pixel_values,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=None,
                        deterministic_reasoning=True,
                        kl_weight=0.0,
                        stage=current_stage  # 🚨 CRITICAL: Pass stage for Stage 1 bypass
                    )
                    
                    # Generate prediction from reasoning latents
                    prediction = model.generate_from_reasoning(
                        reasoning_latents=outputs.reasoning_latents,
                        max_length=10,
                        num_beams=1
                    )[0]
                    
                    # Check match
                    match = prediction.lower().strip() == ground_truth.lower().strip()
                    partial_match = ground_truth.lower().strip() in prediction.lower().strip() or \
                                   prediction.lower().strip() in ground_truth.lower().strip()
                    
                    print(f"\n📋 Sample {i+1}:")
                    print(f"  ❓ Question: {question}")
                    print(f"  ✓ Ground Truth: {ground_truth}")
                    print(f"  🤖 Prediction: {prediction}")
                    if outputs.kl_loss is not None:
                        print(f"  📊 KL: {outputs.kl_loss.item():.4f}")
                    if match:
                        print(f"  ✅ EXACT MATCH")
                    elif partial_match:
                        print(f"  🟡 PARTIAL MATCH")
                    else:
                        print(f"  ❌ WRONG")
            
            print("="*80 + "\n")
            model.train()
        
        scheduler.step()
        curriculum.step()  
        
        # 🚨 NEW: Plot training curves every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == total_epochs - 1:
            try:
                csv_file.flush()  # Ensure data is written
                plot_training_curves(csv_log_path, cfg.save_dir)
            except Exception as e:
                print(f"  ⚠️  Failed to plot curves: {e}")
        
        print()
    
    # Close CSV file
    csv_file.close()
    
    print("\n" + "="*80)
    print("✅ ALL 3 STAGES COMPLETED!")
    print("="*80)
    print(f"\nCheckpoints saved in: {cfg.save_dir}/")
    print(f"  - best.pt (best validation model - use for inference)")
    print(f"  - last.pt (last epoch checkpoint - use for resume)")
    print(f"\nTraining log and curves:")
    print(f"  - {csv_log_path} (CSV log)")
    print(f"  - {os.path.join(cfg.save_dir, 'training_curves.png')} (PNG)")
    print(f"  - {os.path.join(cfg.save_dir, 'training_curves.pdf')} (PDF for papers)")
    
    # Final plot
    print(f"\n[FINAL PLOTTING] Generating final training curves...")
    try:
        plot_training_curves(csv_log_path, cfg.save_dir)
    except Exception as e:
        print(f"  ❌ Failed to generate final plot: {e}")
    
    print()


if __name__ == "__main__":
    main()
