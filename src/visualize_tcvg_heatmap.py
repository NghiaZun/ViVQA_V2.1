"""
HÌNH 1 (khóa luận) — "TCVG Vision Gate Visualization" (tái tạo Fig. 2 của paper).

Bố cục 2 hàng × 4 cột (mỗi cột 1 LOẠI câu hỏi, mỗi cột 1 ẢNH riêng khớp câu hỏi):
  • Hàng trên  — "Baseline (No TCVG)" : ảnh gốc, cổng đồng đều → không highlight.
  • Hàng dưới  — "TCVG (init = 1.0)"  : heatmap α (jet) overlay trên nền xám tối,
                 điểm nóng = patch mà TCVG cho model TẬP TRUNG.

α_i ∈ [0,1] là cổng thị giác per-patch của VisionGating (TCVG). Cùng model run87
nhưng loại câu hỏi khác nhau → phân bố α khác nhau (type-conditioned).

Mặc định dùng ĐÚNG 4 ảnh/câu hỏi như Fig. 2 của paper + checkpoint run87.

Chạy:
  /home/user/workspace/all_env/vivqa/bin/python3 visualize_tcvg_heatmap.py \
      --output figs/fig_tcvg_heatmap.png
  # thang α thật [0,1] (trung thực, không kéo giãn từng panel):
  ... --raw
"""
import os
import math
import argparse
import textwrap

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoImageProcessor, BartphoTokenizer

from viz_utils import build_model_from_checkpoint, TYPE_NAMES


# 4 cột đúng như Fig. 2 của paper: (type_id, img_id, câu hỏi VI, dịch EN)
PAPER_COLUMNS = [
    (0, '549817', 'những gì trong phòng với đường ống tiếp xúc',
        'What is in the room with exposed pipes'),
    (1, '30387', 'có bao nhiêu người đàn ông mặc tạp dề đang tạo dáng bên thức ăn của họ',
        'How many men in aprons are posing by their food'),
    (2, '557067', 'màu của miếng vá là gì',
        'What is the color of the patch'),
    (3, '149185', 'cậu bé đang làm thủ thuật ván trượt ở đâu',
        'Where is the boy doing the skateboard trick'),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', default='checkpoints_run87/best_model.pt')
    p.add_argument('--image_folder', default='archive/data/images/test')
    p.add_argument('--vision_model', default='google/siglip-base-patch16-224')
    p.add_argument('--max_q_len', type=int, default=32)
    p.add_argument('--stretch', action='store_true',
                   help='Kéo giãn min-max TỪNG panel (mặc định: dùng thang α thật [0,1] '
                        '→ OBJECT/LOCATION đỏ đều vì α≈1.0)')
    p.add_argument('--overlay', type=float, default=0.62, help='Độ đậm tối đa của heatmap')
    p.add_argument('--desat', type=float, default=0.85, help='Giảm sắc độ màu (1=gốc, nhỏ=nhạt)')
    p.add_argument('--output', default='figs/fig_tcvg_heatmap.png')
    return p.parse_args()


def heatmap_rgba(grid14, size=224, raw=False, overlay=0.55, vmin=0.0, vmax=1.0, desat=0.72):
    """14×14 α → RGBA 224×224 (jet, giảm sắc độ cho đỡ chói).
    Cold = trong suốt (lộ ảnh), hot = đậm vừa phải."""
    cmap = plt.get_cmap('jet')
    if raw:
        norm = np.clip((grid14 - vmin) / max(vmax - vmin, 1e-6), 0, 1)
    else:
        # kéo giãn min-max từng panel (giống paper) để thấy cấu trúc không gian
        lo, hi = grid14.min(), grid14.max()
        norm = (grid14 - lo) / max(hi - lo, 1e-6)
    up = np.asarray(Image.fromarray((norm * 255).astype(np.uint8))
                    .resize((size, size), Image.BILINEAR)) / 255.0
    rgba = cmap(up)
    # Giảm sắc độ (đỡ chói mắt): pha màu jet về mức xám cùng độ sáng
    rgb = rgba[..., :3]
    lum = rgb.mean(axis=-1, keepdims=True)
    rgba[..., :3] = rgb * desat + lum * (1.0 - desat)
    rgba[..., 3] = np.clip(0.06 + overlay * up, 0, 1)   # alpha theo độ nóng
    return rgba


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model, info = build_model_from_checkpoint(args.checkpoint, args.vision_model, device)
    if not info['has_vision_gate']:
        raise SystemExit("❌ Checkpoint không có vision_gating (TCVG).")
    num_patches = info['num_patches']
    grid = int(round(math.sqrt(num_patches)))
    init_bias = float(info['saved_args'].get('vision_gate_init', 1.0))

    processor = AutoImageProcessor.from_pretrained(args.vision_model)
    tokenizer = BartphoTokenizer.from_pretrained('vinai/bartpho-syllable')

    cols = PAPER_COLUMNS
    pils, px_list = [], []
    for _, img_id, _, _ in cols:
        path = os.path.join(args.image_folder, f"{img_id}.jpg")
        if not os.path.exists(path):
            raise SystemExit(f"❌ Không thấy ảnh: {path}")
        pil = Image.open(path).convert('RGB')
        pils.append(pil.resize((224, 224), Image.BILINEAR))
        px_list.append(processor(images=pil, return_tensors='pt')['pixel_values'])

    pixel_values = torch.cat(px_list, 0).to(device)
    q_enc = tokenizer([c[2] for c in cols], truncation=True, padding='max_length',
                      max_length=args.max_q_len, return_tensors='pt')
    input_ids = q_enc['input_ids'].to(device)
    attention_mask = q_enc['attention_mask'].to(device)
    question_types = torch.tensor([c[0] for c in cols], device=device)

    # Hook bắt α
    captured = {}

    def hook_fn(module, inp, out):
        captured['alpha'] = out[1].detach().float().cpu()

    h = model.vision_gating.register_forward_hook(hook_fn)
    with torch.no_grad():
        model(pixel_values=pixel_values, input_ids=input_ids,
              attention_mask=attention_mask, question_types=question_types)
    h.remove()

    alpha = captured['alpha']
    if alpha.size(1) == num_patches + 1 and info['use_siglip_pooler']:
        alpha = alpha[:, 1:]
    alpha = alpha[:, :num_patches]
    grids = alpha.reshape(len(cols), grid, grid).numpy()

    print("\n[viz] α theo loại (TCVG, run87):")
    for i, (t, iid, q, _) in enumerate(cols):
        g = grids[i]
        print(f"  {TYPE_NAMES[t]:9s} img={iid}  μ={g.mean():.3f} σ={g.std():.3f} "
              f"min={g.min():.3f} max={g.max():.3f}")

    # ── Vẽ ───────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 4, figsize=(18, 9.4))
    for i, (t, iid, q_vi, q_en) in enumerate(cols):
        disp = np.asarray(pils[i])
        gray = np.asarray(pils[i].convert('L'))
        dark = (np.stack([gray] * 3, -1) * 0.62).astype(np.uint8)   # nền xám (sáng vừa)

        # Hàng trên — Baseline (ảnh gốc, không highlight)
        ax0 = axes[0, i]
        ax0.imshow(disp)
        q_vi_w = '\n'.join(textwrap.wrap(q_vi, width=30))
        q_en_w = '\n'.join(textwrap.wrap(q_en, width=32))
        title = f"{TYPE_NAMES[t]}\n{q_vi_w}\n\n{q_en_w}"
        ax0.set_title(title, fontsize=13, fontweight='bold')
        ax0.axis('off')

        # Hàng dưới — TCVG heatmap (mặc định thang α thật [0,1])
        ax1 = axes[1, i]
        ax1.imshow(dark)
        rgba = heatmap_rgba(grids[i], size=224, raw=(not args.stretch),
                            overlay=args.overlay, desat=args.desat)
        ax1.imshow(rgba, interpolation='bilinear')
        ax1.set_title(f"α: μ={grids[i].mean():.2f}, σ={grids[i].std():.2f}", fontsize=13.5)
        ax1.axis('off')

    # Nhãn hàng bên trái
    axes[0, 0].text(-0.11, 0.5, 'Baseline\nkhông TCVG', transform=axes[0, 0].transAxes,
                    rotation=90, va='center', ha='center', fontsize=17,
                    fontweight='bold', color='black')
    axes[1, 0].text(-0.11, 0.5, f'TCVG\ninit = {init_bias:.1f}', transform=axes[1, 0].transAxes,
                    rotation=90, va='center', ha='center', fontsize=17,
                    fontweight='bold', color='#c81e1e')

    note = '  — α chuẩn hoá tương phản theo panel' if args.stretch else ''
    fig.suptitle('TCVG — Cổng giữ đặc trưng thị giác theo patch' + note,
                 fontsize=21, fontweight='bold')
    fig.subplots_adjust(left=0.06, right=0.99, top=0.74, bottom=0.12, hspace=0.10, wspace=0.05)

    # Colorbar chung — chỉ có ý nghĩa định lượng khi dùng thang α thật [0,1]
    if not args.stretch:
        import matplotlib as mpl
        sm = mpl.cm.ScalarMappable(cmap=plt.get_cmap('jet'),
                                   norm=mpl.colors.Normalize(vmin=0.0, vmax=1.0))
        cbar_ax = fig.add_axes([0.30, 0.06, 0.40, 0.024])
        cb = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
        cb.set_label('α — mức giữ đặc trưng thị giác của patch: '
                     '1 = giữ làm bằng chứng, 0 = thay bằng ngữ cảnh câu hỏi',
                     fontsize=12.5)
        cb.ax.tick_params(labelsize=11)

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches='tight')
    print(f"\n✅ Đã lưu: {args.output}")


if __name__ == '__main__':
    main()
