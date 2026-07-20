"""
BIỂU ĐỒ TCVG (seed 42) — số liệu FIX CỨNG, không cần train lại / không cần checkpoint.

2 panel:
  • Trái : α trung bình MỖI LOẠI (mức giữ đặc trưng thị giác của cổng TCVG)
           → thấp = cổng CHỌN LỌC (dập nền); ≈1.0 = giữ toàn cảnh.
  • Phải : Exact Match Baseline (T0) vs TCVG (T2) theo loại + Δ tăng
           → TCVG cải thiện mọi loại, mạnh nhất ở Màu sắc/Số lượng.

Toàn bộ số liệu là kết quả seed = 42 (run87), được HARD-CODE bên dưới nên script
chạy độc lập chỉ với matplotlib (không cần dataset/ảnh/checkpoint — vốn không kèm repo).

Chạy:
  python visualize_alpha_chart.py --output figs/fig_tcvg_chart.png
"""
import os
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────────────────────
# SỐ LIỆU SEED 42 (run87) — FIX CỨNG
#   loại_id: (alpha_trung_bình, EM_baseline_T0, EM_tcvg_T2)
#   Δ tăng = EM_tcvg − EM_baseline  (tính tự động bên dưới để luôn nhất quán)
# ─────────────────────────────────────────────────────────────────────────────
SEED42 = {
    2: (0.44, 67.82, 72.77),   # Màu sắc  / COLOR    → Δ +4.95
    1: (0.54, 62.25, 67.08),   # Số lượng / COUNT    → Δ +4.83
    0: (1.00, 70.02, 74.10),   # Đối tượng/ OBJECT   → Δ +4.08
    3: (1.00, 66.71, 71.51),   # Vị trí   / LOCATION → Δ +4.80
}
ORDER = [2, 1, 0, 3]           # COLOR, COUNT, OBJECT, LOCATION (theo mức chọn lọc α)

TYPE_VI = {0: 'Đối tượng', 1: 'Số lượng', 2: 'Màu sắc', 3: 'Vị trí'}
TYPE_EN = {0: 'OBJECT', 1: 'COUNT', 2: 'COLOR', 3: 'LOCATION'}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--output', default='figs/fig_tcvg_chart.png')
    p.add_argument('--dpi', type=int, default=200)
    return p.parse_args()


def main():
    args = parse_args()

    labels = [f"{TYPE_VI[t]}\n{TYPE_EN[t]}" for t in ORDER]
    alpha = [SEED42[t][0] for t in ORDER]
    base  = [SEED42[t][1] for t in ORDER]
    tcvg  = [SEED42[t][2] for t in ORDER]
    delta = [tcvg[i] - base[i] for i in range(len(ORDER))]

    print("[chart] Số liệu seed 42 (fix cứng):")
    for i, t in enumerate(ORDER):
        print(f"  {TYPE_EN[t]:9s} α={alpha[i]:.2f}  EM {base[i]:.2f} → {tcvg[i]:.2f}  (Δ +{delta[i]:.2f})")

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(16, 6.2))

    # ── Panel trái: α selectivity ────────────────────────────────────────────
    y = np.arange(len(ORDER))
    barcolors = ['#2166ac' if a < 0.75 else '#b2182b' for a in alpha]
    axL.barh(y, alpha, color=barcolors, height=0.6)
    for i, a in enumerate(alpha):
        axL.text(a + 0.02, y[i], f"{a:.2f}", va='center', fontsize=15, fontweight='bold')
    axL.set_yticks(y); axL.set_yticklabels(labels, fontsize=14)
    axL.set_xlim(0, 1.15); axL.set_xlabel('α trung bình (mức giữ đặc trưng thị giác)', fontsize=14)
    axL.axvline(0.75, ls='--', color='gray', lw=1)
    axL.set_title('(a) Cổng TCVG theo loại câu hỏi\nthấp = CHỌN LỌC (dập nền) · ≈1.0 = giữ toàn cảnh',
                  fontsize=15, fontweight='bold')
    axL.invert_yaxis()

    # ── Panel phải: EM Baseline vs TCVG + Δ ──────────────────────────────────
    x = np.arange(len(ORDER)); w = 0.38
    axR.bar(x - w/2, base, w, label='Baseline (không TCVG)', color='#bdbdbd')
    axR.bar(x + w/2, tcvg, w, label='TCVG (ViGCT-VQA)', color='#2166ac')
    for i in range(len(ORDER)):
        axR.text(x[i] + w/2, tcvg[i] + 0.4, f"+{delta[i]:.2f}",
                 ha='center', fontsize=13, fontweight='bold', color='#1a7a1a')
    axR.set_xticks(x); axR.set_xticklabels(labels, fontsize=14)
    axR.set_ylim(55, 80); axR.set_ylabel('Exact Match (%)', fontsize=14)
    axR.set_title('(b) Độ chính xác theo loại (seed 42)\nTCVG cải thiện mọi loại',
                  fontsize=15, fontweight='bold')
    axR.legend(fontsize=13, loc='lower right')

    fig.suptitle('TCVG điều tiết cổng thị giác theo loại câu hỏi — cải thiện chọn lọc (seed 42)',
                 fontsize=18, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches='tight')
    print(f"\n✅ Đã lưu: {args.output}")


if __name__ == '__main__':
    main()
