"""
TRAINING UTILITIES
==================
Shared utilities for training scripts (train.py, run_3stage.py)
Contains run_one_epoch and other training functions
"""

import random
import os
import torch
from dataclasses import dataclass
from torch.amp import autocast
import torch.nn.functional as F
from tqdm import tqdm


def set_seed(seed=42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.benchmark = True


@dataclass
class FixedTrainConfig:
    """Configuration with all fixes"""
    # Data
    csv_path: str = "/kaggle/input/vivqa/ViVQA-main/ViVQA-main/train.csv"
    image_folder: str = "/kaggle/input/vivqa/drive-download-20220309T020508Z-001/train"
    save_dir: str = "/kaggle/working/checkpoints_fixed"
    
    # Training
    stage: int = 1  # 1: Baseline, 2: Warmup, 3: Full
    batch_size: int = 4
    accum_steps: int = 8
    num_epochs: int = 10
    val_split: float = 0.1
    num_workers: int = 4
    
    # Optimization - ✅ FIXED LR for frozen setup
    base_lr: float = 5e-5
    fusion_lr: float = 5e-4  # 🚨 INCREASED: 2e-4 → 5e-4 (fusion needs higher LR)
    decoder_lr: float = 5e-4  # 🚨 CRITICAL FIX: 2e-4 → 5e-4 (decoder cần học NHANH để adapt reasoning!)
    encoder_lr: float = 5e-6  # Lower LR for encoder finetuning
    weight_decay: float = 0.05
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.06
    use_amp: bool = True
    
    # Model
    num_reasoning_tokens: int = 6
    latent_dim: int = 512  # 🔥 CRITICAL FIX: 256→512 (KL=0.17 vì capacity không đủ!)
    # Root cause: 6×256=1536 capacity < 1792 input (vision 768 + text 1024)
    # New: 6×512=3072 capacity > 1792 → KL should drop to 0.03-0.08
    num_reasoning_layers: int = 2
    num_fusion_layers: int = 2
    free_bits: float = 0.15  # 🔥🔥🔥 CRITICAL: 0.005→0.15 (KL=0.27, need AGGRESSIVE clamp!)
    # Current issue: KL raw=0.268, free_bits=0.005 only removes 2% (USELESS!)
    # Target: Remove ~50% → 0.268-0.15=0.118 (acceptable range)
    # This is NOT cheating - free bits = "minimum KL to maintain" (β-VAE theory)
    ortho_weight: float = 0.05  # 🔥 REDUCED: 0.1→0.05 (ortho pushes tokens apart → higher KL)
    token_dropout_prob: float = 0.3
    unfreeze_encoder_layers: int = 0
    
    # Teacher distillation
    use_teacher: bool = False
    teacher_type: str = 'rule_based'
    teacher_weight: float = 0.3  # 🚨 REFINED: 0.5→0.3 (giảm coupling với KL + high decoder LR)
    num_reasoning_samples: int = 5
    reasoning_temperature: float = 0.6  # 🚨 CRITICAL: Train temp=0.6 (exploration), Val=0.5 (deterministic)
    reasoning_temperature_val: float = 0.5  # 🚨 NEW: Separate temp for validation
    preference_margin: float = 0.1
    
    # Early stopping
    es_patience: int = 6
    es_min_delta: float = 1e-4


def run_one_epoch(
    model, loader, optimizer, scaler, device, cfg,
    curriculum, stage, teacher_evaluator=None, scheduler=None, train=True
):
    """
    🚨 FIXED: Run one epoch with stage control and teacher
    
    CRITICAL FIXES:
    - Pass `stage` parameter to model.forward() for Stage 1 bypass
    - Handle teacher distillation in Stage 3
    """
    if train:
        model.train()
    else:
        model.eval()
    
    total_loss = 0.0
    answer_loss_sum = 0.0
    kl_loss_sum = 0.0
    kl_loss_raw_sum = 0.0  # 🚨 NEW: Track raw KL before free bits
    ortho_loss_sum = 0.0
    teacher_loss_sum = 0.0
    num_batches = 0
    
    if train:
        optimizer.zero_grad()
    
    pbar = tqdm(loader, desc="Train" if train else "Val", ncols=120)
    
    for batch_idx, (pixel_values, input_ids, attention_mask, labels) in enumerate(pbar):
        pixel_values = pixel_values.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        
        # Get curriculum parameters
        kl_weight = curriculum.get_kl_weight(stage)
        stop_grad = curriculum.get_stop_gradient(stage)
        
        # 🚨 CLARIFIED: Temperature handling
        # Train: deterministic=False, temperature=0.6 (exploration)
        # Val:   deterministic=False, temperature=0.5 (lower noise, more stable)
        # Note: deterministic=True only for intervention tests (not during training/val)
        temperature = cfg.reasoning_temperature if train else cfg.reasoning_temperature_val
        
        with autocast(device_type='cuda', enabled=cfg.use_amp):
            # 🚨 CRITICAL: Pass stage parameter for Stage 1 bypass!
            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                deterministic_reasoning=not train,
                stop_gradient_to_latent=stop_grad,
                kl_weight=kl_weight,
                temperature=temperature,  # 🚨 NEW: Use train/val-specific temperature
                stage=stage  # 🚨 FIXED!
            )
            
            loss = outputs.total_loss
            
            # 🚨 NEW: NaN detection and early abort
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n❌ NaN/Inf detected at batch {batch_idx}!")
                print(f"   Answer loss: {outputs.answer_loss.item() if outputs.answer_loss is not None else 'N/A'}")
                print(f"   KL loss: {outputs.kl_loss.item() if outputs.kl_loss is not None else 'N/A'}")
                print(f"   KL raw: {outputs.kl_loss_raw.item() if outputs.kl_loss_raw is not None else 'N/A'}")
                print(f"   Ortho loss: {outputs.ortho_loss.item() if outputs.ortho_loss is not None else 'N/A'}")
                print(f"   KL weight: {kl_weight:.4f}")
                print(f"   Stage: {stage}")
                raise RuntimeError(f"Training stopped due to NaN/Inf loss at batch {batch_idx}")
            
            # Teacher distillation (Stage 3 only)
            teacher_loss = torch.tensor(0.0, device=device)
            
            if cfg.use_teacher and teacher_evaluator is not None and train and stage == 3:
                # Get ground truths
                ground_truths = []
                for i in range(labels.size(0)):
                    label_ids = labels[i][labels[i] != -100]
                    gt = model.tokenizer.decode(label_ids, skip_special_tokens=True).strip()
                    ground_truths.append(gt)
                
                # 🚨 OPTIMIZED: Sample all reasoning paths in single forward pass
                # Old: Loop num_samples times → B × N forward passes
                # New: Expand batch → single forward with (B × N) batch size
                batch_size = pixel_values.size(0)
                num_samples = cfg.num_reasoning_samples
                
                # Expand inputs: [B, ...] → [B * N, ...]
                expanded_pixels = pixel_values.repeat_interleave(num_samples, dim=0)
                expanded_input_ids = input_ids.repeat_interleave(num_samples, dim=0)
                expanded_attention_mask = attention_mask.repeat_interleave(num_samples, dim=0)
                
                # Single forward for all samples
                sample_out = model(
                    pixel_values=expanded_pixels,
                    input_ids=expanded_input_ids,
                    attention_mask=expanded_attention_mask,
                    labels=None,
                    deterministic_reasoning=False,
                    temperature=cfg.reasoning_temperature,
                    stop_gradient_to_latent=stop_grad,
                    kl_weight=0.0,
                    stage=stage
                )
                
                # Generate answers from all samples at once
                with torch.no_grad():
                    all_answers = model.generate_from_reasoning(
                        reasoning_latents=sample_out.reasoning_latents,
                        max_length=10,
                        num_beams=1
                    )
                
                # Reshape back to [B, N]
                candidate_answers = []
                for i in range(batch_size):
                    answers_for_example = all_answers[i * num_samples:(i + 1) * num_samples]
                    candidate_answers.append(answers_for_example)
                
                # Teacher evaluates (still per-example for scoring)
                batch_teacher_losses = []
                
                for batch_i in range(batch_size):
                    # Get N answers for this example
                    answers_for_example = candidate_answers[batch_i]
                    
                    teacher_scores = teacher_evaluator.evaluate_answers(
                        answers_for_example,
                        [ground_truths[batch_i]] * num_samples,
                        images=pixel_values[batch_i:batch_i+1].repeat(num_samples, 1, 1, 1) if cfg.teacher_type == 'vlm' else None,
                        questions=None
                    )
                    
                    # Preference loss
                    best_idx = teacher_scores.argmax()
                    worst_idx = teacher_scores.argmin()
                    
                    preference_loss = F.margin_ranking_loss(
                        teacher_scores[best_idx].unsqueeze(0),
                        teacher_scores[worst_idx].unsqueeze(0),
                        target=torch.ones(1, device=device),
                        margin=cfg.preference_margin
                    )
                    
                    batch_teacher_losses.append(preference_loss)
                
                teacher_loss = torch.stack(batch_teacher_losses).mean()
                loss = loss + cfg.teacher_weight * teacher_loss
            
            if train:
                loss = loss / cfg.accum_steps
        
        # Backward
        if train:
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % cfg.accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        
        # Accumulate
        total_loss += loss.item() * cfg.accum_steps if train else loss.item()
        answer_loss_sum += outputs.answer_loss.item() if outputs.answer_loss is not None else 0
        kl_loss_sum += outputs.kl_loss.item() if outputs.kl_loss is not None else 0
        kl_loss_raw_sum += outputs.kl_loss_raw.item() if outputs.kl_loss_raw is not None else 0  # 🚨 NEW
        ortho_loss_sum += outputs.ortho_loss.item() if outputs.ortho_loss is not None else 0
        teacher_loss_sum += teacher_loss.item() if isinstance(teacher_loss, torch.Tensor) else 0
        num_batches += 1
        
        # 🚨 FIXED: Realtime KL diagnostics in progress bar
        if hasattr(outputs, 'kl_loss_raw') and outputs.kl_loss_raw is not None:
            kl_raw_value = outputs.kl_loss_raw.item()
            kl_after_value = outputs.kl_loss.item() if outputs.kl_loss is not None else 0
            penalty_red = ((kl_raw_value - kl_after_value) / kl_raw_value * 100) if kl_raw_value > 1e-6 else 0
            
            # 🚀 NEW: Add gate value monitoring
            gate_value = getattr(model, 'last_text_gate', 0.0)
            
            pbar.set_postfix({
                'L': f"{loss.item() * cfg.accum_steps if train else loss.item():.3f}",
                'A': f"{outputs.answer_loss.item():.3f}",
                'KLr': f"{kl_raw_value:.3f}",  # Raw KL before free bits
                'KLa': f"{kl_after_value:.3f}",  # After free bits
                'fb%': f"{penalty_red:.0f}%",  # Free bits reduction
                'gate': f"{gate_value:.3f}",  # 🚀 NEW: Text gate value
                'T': f"{teacher_loss.item():.3f}",
                'KLw': f"{kl_weight * 0.2:.4f}"
            })
        else:
            # Fallback
            pbar.set_postfix({
                'L': f"{loss.item() * cfg.accum_steps if train else loss.item():.3f}",
                'A': f"{outputs.answer_loss.item():.3f}",
                'KL': f"{outputs.kl_loss.item():.3f}",
                'O': f"{outputs.ortho_loss.item():.3f}",
                'T': f"{teacher_loss.item():.3f}",
                'KLw': f"{kl_weight * 0.2:.4f}"
            })
    
    return {
        'total': total_loss / num_batches,
        'answer': answer_loss_sum / num_batches,
        'kl': kl_loss_sum / num_batches,
        'kl_raw': kl_loss_raw_sum / num_batches,  # 🚨 NEW: Include raw KL
        'ortho': ortho_loss_sum / num_batches,
        'teacher': teacher_loss_sum / num_batches
    }
