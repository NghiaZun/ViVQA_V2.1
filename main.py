"""
ViVQA Fine-tune Loop v2 — pipeline chính.

Sơ đồ mỗi loop:
  Eval → F1/EM từng type
  → Stop check (EM variance + delta + gap)
  → Compute generation plan (budget theo deficit, style theo gap)
  → Generate images (FLUX.1-schnell + Qwen/YOLO verify)
  → Merge dataset
  → Fine-tune (resume từ checkpoint trước)
"""

import json
import os
import subprocess
import sys

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from config import CFG, TYPES
from evaluator import run_eval, should_stop
from generation import (
    compute_generation_plan, print_plan,
    load_generation_models, unload_generation_models,
    generate_images_for_type,
)
from pipeline_utils import (
    combined_image_dir, rebuild_combined_image_dir,
    merge_dataset, recompute_answer_weights,
)


# ── Helpers ───────────────────────────────────────────────────────

def run_train(train_csv, val_csv, image_dir, output_dir, epochs=None,
              lr=None, warmup_epochs=None, resume=None, seed=None,
              early_stopping=None, early_stopping_patience=None):
    # Defaults từ config.py — cho phép override qua tham số
    epochs                  = CFG['epochs']                   if epochs                  is None else epochs
    lr                      = CFG['lr']                       if lr                      is None else lr
    warmup_epochs           = CFG['warmup_epochs']            if warmup_epochs           is None else warmup_epochs
    seed                    = CFG['seed']                     if seed                    is None else seed
    early_stopping          = CFG['early_stopping']           if early_stopping          is None else early_stopping
    early_stopping_patience = CFG['early_stopping_patience']  if early_stopping_patience is None else early_stopping_patience

    parts = [
        'python train.py',
        f'--train_csv {train_csv}',
        f'--val_csv {val_csv}',
        f'--image_dir {image_dir}',
        f'--output_dir {output_dir}',
        f'--vision_model {CFG["vision_model"]}',
        f'--bartpho_model {CFG["bartpho_model"]}',
        f'--fusion_type {CFG["fusion_type"]}',
        f'--num_fusion_layers {CFG["num_fusion_layers"]}',
        f'--epochs {epochs}',
        f'--lr {lr}',
        f'--weight_decay {CFG["weight_decay"]}',
        f'--dropout {CFG["dropout"]}',
        f'--label_smoothing {CFG["label_smoothing"]}',
        f'--gradient_accumulation_steps {CFG["gradient_accumulation_steps"]}',
        f'--scheduler {CFG["scheduler"]}',
        f'--warmup_epochs {warmup_epochs}',
        f'--early_stopping_metric {CFG["early_stopping_metric"]}',
        f'--sample_every {CFG["sample_every"]}',
        f'--answer_weights {CFG["answer_weights"]}',
        f'--seed {seed}',
    ]
    if CFG.get('pk_sampling'):
        parts += [
            '--pk_sampling',
            f'--pk_p {CFG["pk_p"]}',
            f'--pk_k {CFG["pk_k"]}',
        ]
    if CFG.get('use_text_lora'):
        parts += [
            '--use_text_lora',
            f'--text_lora_r {CFG["text_lora_r"]}',
            f'--text_lora_alpha {CFG["text_lora_alpha"]}',
        ]
    if CFG.get('use_vision_gate'):
        parts.append('--use_vision_gate')
    if CFG.get('use_type_loss'):
        parts.append('--use_type_loss')
    if CFG.get('use_contrastive'):
        parts += [
            '--use_contrastive',
            f'--contrastive_lambda {CFG["contrastive_lambda"]}',
            f'--contrastive_temp {CFG["contrastive_temp"]}',
        ]
    if resume:
        parts += [f'--resume {resume}', '--reset_lr']
    if early_stopping:
        parts += ['--early_stopping', f'--early_stopping_patience {early_stopping_patience}']

    cmd = ' '.join(parts)
    subprocess.run(cmd, shell=True, check=False)


def resolve_checkpoint(ckpt_dir):
    best = os.path.join(ckpt_dir, 'best_model.pt')
    last = os.path.join(ckpt_dir, 'last_model.pt')
    if os.path.exists(best):
        return best
    if os.path.exists(last):
        print(f'⚠️  best_model.pt không có, dùng last_model.pt')
        return last
    return None


def plot_history(history, overall_final, per_type_final, out_path):
    loops = [h['loop'] for h in history] + [len(history)]
    colors = {'COUNT': '#e74c3c', 'LOCATION': '#2ecc71', 'OBJECT': '#f39c12', 'COLOR': '#3498db'}

    def get_metric(metric, type_name=None):
        vals = []
        for h in history:
            vals.append(h['overall'][metric] if type_name is None
                        else h['per_type'].get(type_name, {}).get(metric, 0))
        vals.append(overall_final[metric] if type_name is None
                    else per_type_final.get(type_name, {}).get(metric, 0))
        return vals

    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(loops, get_metric('F1'), 'o-', color='steelblue',  lw=2, label='Overall F1')
    ax1.plot(loops, get_metric('EM'), 's--', color='darkorange', lw=2, label='Overall EM')
    ax1.set_title('Overall F1 vs EM per loop')
    ax1.set_xlabel('Loop'); ax1.set_ylabel('%'); ax1.legend(); ax1.grid(alpha=0.3)
    ax1.set_xticks(loops)

    ax2 = fig.add_subplot(gs[0, 1])
    for t in TYPES:
        ax2.plot(loops, get_metric('F1', t), 'o-', color=colors[t], lw=2, label=t)
    ax2.set_title('Per-type F1 improvement')
    ax2.set_xlabel('Loop'); ax2.set_ylabel('F1 (%)'); ax2.legend(); ax2.grid(alpha=0.3)
    ax2.set_xticks(loops)

    ax3 = fig.add_subplot(gs[1, 0])
    for t in TYPES:
        ax3.plot(loops, get_metric('EM', t), 's--', color=colors[t], lw=2, label=t)
    ax3.set_title('Per-type EM improvement  ← stop condition')
    ax3.set_xlabel('Loop'); ax3.set_ylabel('EM (%)'); ax3.legend(); ax3.grid(alpha=0.3)
    ax3.set_xticks(loops)

    ax4 = fig.add_subplot(gs[1, 1])
    for t in TYPES:
        f1s  = get_metric('F1', t)
        ems  = get_metric('EM', t)
        gaps = [f - e for f, e in zip(f1s, ems)]
        ax4.plot(loops, gaps, 'o-', color=colors[t], lw=2, label=t)
    ax4.axhline(CFG['stop_gap_max'], linestyle='--', color='gray', alpha=0.7,
                label=f'Stop threshold ({CFG["stop_gap_max"]})')
    ax4.set_title('F1 - EM gap per type  ← sinh ảnh style')
    ax4.set_xlabel('Loop'); ax4.set_ylabel('Gap (F1 - EM)'); ax4.legend(); ax4.grid(alpha=0.3)
    ax4.set_xticks(loops)

    plt.suptitle('ViVQA Fine-tune Loop v2', fontsize=14, fontweight='bold')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'✓ Plot saved → {out_path}')
    matplotlib.pyplot.close(fig)


# ── Main pipeline ─────────────────────────────────────────────────

def main():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB')

    current_train_csv = CFG['train_csv']
    current_ckpt      = None
    best_ckpt         = None
    best_em           = -1.0
    best_loop         = -1
    prev_overall_em   = 0.0
    history           = []

    # ── Loop 0: baseline training ─────────────────────────────────
    print('\n' + '=' * 60)
    print('LOOP 0 — Training baseline')
    print('=' * 60)

    recompute_answer_weights(current_train_csv)
    rebuild_combined_image_dir()

    ckpt_dir_0 = os.path.join(CFG['ckpt_dir'], 'loop0')
    os.makedirs(ckpt_dir_0, exist_ok=True)

    run_train(
        train_csv=current_train_csv,
        val_csv=CFG['val_csv'],
        image_dir=combined_image_dir,
        output_dir=ckpt_dir_0,
        epochs=CFG['initial_epochs'],
    )

    current_ckpt = resolve_checkpoint(ckpt_dir_0)
    if current_ckpt is None:
        print('⚠️  Không tìm thấy checkpoint sau loop 0, thoát.')
        sys.exit(1)
    best_ckpt = current_ckpt
    print(f'✓ Baseline checkpoint: {current_ckpt}')

    # ── Loops 1…max_loops ─────────────────────────────────────────
    for loop in range(1, CFG['max_loops'] + 1):
        print()
        print('=' * 65)
        print(f'LOOP {loop}')
        print('=' * 65)

        # A: Eval
        print(f'\n[{loop}A] Evaluating...')
        if not os.path.exists(current_ckpt):
            print(f'  ⚠️  Checkpoint không tồn tại: {current_ckpt}')
            overall, per_type = {'F1': 0.0, 'EM': 0.0}, {}
        else:
            result_csv = os.path.join(CFG['work_dir'], f'result_loop{loop-1}.csv')
            overall, per_type = run_eval(current_ckpt, result_csv)

        print(f'\n  Overall  F1={overall["F1"]:.2f}%  EM={overall["EM"]:.2f}%')
        print(f'  {"Type":<12} {"F1":>6}  {"EM":>6}  {"Gap":>6}')
        print(f'  {"-"*40}')
        for t in TYPES:
            if t in per_type:
                f1, em = per_type[t]['F1'], per_type[t]['EM']
                print(f'  {t:<12} {f1:>6.2f}  {em:>6.2f}  {f1-em:>6.2f}')

        history.append({'loop': loop - 1, 'overall': overall, 'per_type': per_type})

        # Track best checkpoint xuyên loop theo EM
        if per_type and overall['EM'] > best_em:
            best_em   = overall['EM']
            best_ckpt = current_ckpt
            best_loop = loop - 1
            print(f'  ★ New best EM={best_em:.2f}% tại loop {best_loop} → {best_ckpt}')

        # B: Stop check
        if per_type and should_stop(overall, per_type, prev_overall_em):
            print(f'\n✅ Tất cả điều kiện EM đạt. Dừng loop tại loop {loop-1}.')
            break

        if per_type:
            prev_overall_em = overall['EM']

        # C: Generation plan — đọc result.csv để target answer hay sai
        print(f'\n[{loop}C] Computing generation plan...')
        result_csv = os.path.join(CFG['work_dir'], f'result_loop{loop-1}.csv')
        plan = compute_generation_plan(result_csv=result_csv, train_csv=current_train_csv)
        print_plan(plan)

        # D: Generate
        print(f'\n[{loop}D] Loading generation models...')
        gen_models = load_generation_models()

        new_rows_all = []
        for type_name, plan_entry in plan.items():
            new_rows = generate_images_for_type(
                type_name, plan_entry, current_train_csv, gen_models, loop,
            )
            new_rows_all.extend(new_rows)

        unload_generation_models(gen_models)

        # E: Merge — chỉ dùng aug của loop này + original data
        # KHÔNG tích lũy aug từ loop trước để tránh noise compounding
        print(f'\n[{loop}E] Merging dataset (loop {loop} aug only)...')
        loop_train_csv = merge_dataset(CFG['train_csv'], new_rows_all, loop)

        import shutil
        loop_img_dir = os.path.join(CFG['aug_root'], f'result_{loop}')
        loop_combined = os.path.join(CFG['work_dir'], f'combined_loop{loop}')
        if os.path.exists(loop_combined):
            shutil.rmtree(loop_combined)
        os.makedirs(loop_combined)
        for f in os.listdir(CFG['image_dir']):
            os.symlink(os.path.join(CFG['image_dir'], f),
                       os.path.join(loop_combined, f))
        if os.path.isdir(loop_img_dir):
            for f in os.listdir(loop_img_dir):
                dst = os.path.join(loop_combined, f)
                if not os.path.exists(dst):
                    shutil.copy2(os.path.join(loop_img_dir, f), dst)
        print(f'✓ Loop {loop} combined: {len(os.listdir(loop_combined))} images')

        recompute_answer_weights(loop_train_csv)

        # F: Fine-tune từ loop 0 baseline — không resume từ checkpoint đã degraded
        print(f'\n[{loop}F] Fine-tuning từ baseline checkpoint...')
        ckpt_dir_loop = os.path.join(CFG['ckpt_dir'], f'loop{loop}')
        os.makedirs(ckpt_dir_loop, exist_ok=True)

        run_train(
            train_csv=loop_train_csv,
            val_csv=CFG['val_csv'],
            image_dir=loop_combined,
            output_dir=ckpt_dir_loop,
            epochs=CFG['initial_epochs'],
            resume=best_ckpt,          # luôn resume từ best checkpoint (loop 0 ban đầu)
            early_stopping=True,
            early_stopping_patience=CFG['finetune_es_patience'],
        )

        ckpt = resolve_checkpoint(ckpt_dir_loop)
        if ckpt:
            current_ckpt = ckpt
            print(f'\n✓ Loop {loop} done. Checkpoint: {current_ckpt}')
        else:
            print(f'\n⚠️  Không có checkpoint nào trong loop {loop}, giữ: {current_ckpt}')

    else:
        print('\n⛔ Đạt max_loops.')

    # ── Final eval + summary ──────────────────────────────────────
    print('\n' + '=' * 65)
    print('FINAL EVAL')
    print('=' * 65)
    print(f'Best checkpoint: loop {best_loop}, EM={best_em:.2f}% → {best_ckpt}')

    final_result_csv = os.path.join(CFG['work_dir'], 'result_final.csv')
    overall_final, per_type_final = run_eval(best_ckpt, final_result_csv)

    print('\n' + '=' * 75)
    print('RESULTS SUMMARY')
    print('=' * 75)
    header = f'{"Loop":<6}  {"Overall F1":>10}  {"Overall EM":>10}  '
    for t in TYPES:
        header += f'{t+" F1":>10}  {t+" EM":>10}  '
    print(header)
    print('-' * 75)

    for h in history:
        row = f'{h["loop"]:<6}  {h["overall"]["F1"]:>10.2f}  {h["overall"]["EM"]:>10.2f}  '
        for t in TYPES:
            if t in h['per_type']:
                row += f'{h["per_type"][t]["F1"]:>10.2f}  {h["per_type"][t]["EM"]:>10.2f}  '
            else:
                row += f'{"N/A":>10}  {"N/A":>10}  '
        print(row)

    final_row = f'{"FINAL":<6}  {overall_final["F1"]:>10.2f}  {overall_final["EM"]:>10.2f}  '
    for t in TYPES:
        if t in per_type_final:
            final_row += f'{per_type_final[t]["F1"]:>10.2f}  {per_type_final[t]["EM"]:>10.2f}  '
    print(final_row)

    history_path = os.path.join(CFG['work_dir'], 'pipeline_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    print(f'\n✓ History saved → {history_path}')

    plot_history(
        history, overall_final, per_type_final,
        out_path=os.path.join(CFG['work_dir'], 'finetune_loop_v2.png'),
    )

    print('\n' + '=' * 65)
    print('PIPELINE COMPLETE')
    print(f'Best checkpoint (loop {best_loop}, EM={best_em:.2f}%): {best_ckpt}')
    print('=' * 65)


if __name__ == '__main__':
    main()
