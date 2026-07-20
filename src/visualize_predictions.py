"""
HÌNH 2 & 3 (khóa luận) — ví dụ định tính: dự đoán ĐÚNG và dự đoán SAI.

Sinh ra 2 hình (bố cục mỗi ví dụ MỘT HÀNG: ảnh bên trái | chú thích bên phải):
  • fig_correct.png : ví dụ model trả lời ĐÚNG (khung xanh, ✓)
  • fig_wrong.png   : ví dụ model trả lời SAI  (khung đỏ, ✗) + nhận xét lỗi

Ảnh, câu hỏi, đáp án đúng lấy CÙNG MỘT DÒNG trong test.csv nên luôn khớp nhau.
Inference chạy đúng pipeline eval.py (beam search + constrained decoding trên trie
train_split) để con số khớp bảng kết quả trong khóa luận.

Chọn ví dụ SAI "dễ giải thích": tự chấm điểm ưu tiên các lỗi có ý nghĩa
(COUNT lệch 1 đơn vị, COLOR nhầm màu cụ thể, OBJECT/LOC đoán vật liên quan) và
LOẠI BỎ các dòng câu hỏi quá dài / nhãn nhiễu (dự đoán rỗng).

Cách chạy (mặc định run87):
  /home/user/workspace/all_env/vivqa/bin/python3 visualize_predictions.py \
      --n 3 --seed 42 --scan 600 --outdir figs

  # cố định ví dụ theo img_id:
  ... --correct_ids 557067 436394 --wrong_ids 527073 ...
"""
import os
import re
import argparse
import random
import textwrap

import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from transformers import AutoImageProcessor, BartphoTokenizer

from viz_utils import build_model_from_checkpoint, TYPE_NAMES, TYPE_NAMES_VI
from eval import (
    build_answer_trie, build_valid_answers_set, snap_to_valid_answer,
    compute_exact_match, _normalize_vn,
)


# ── Tiện ích tiếng Việt để chấm điểm / nhận xét lỗi ─────────────────────────
_NUM_WORDS = {
    'không': 0, 'một': 1, 'hai': 2, 'ba': 3, 'bốn': 4, 'tư': 4, 'năm': 5,
    'lăm': 5, 'sáu': 6, 'bảy': 7, 'tám': 8, 'chín': 9, 'mười': 10,
    'mười một': 11, 'mười hai': 12,
}
_COLORS = {'đỏ', 'cam', 'vàng', 'xanh', 'xanh lá', 'xanh dương', 'lục', 'lam',
           'tím', 'hồng', 'nâu', 'đen', 'trắng', 'xám', 'be', 'bạc', 'tía',
           'vàng kim', 'nâu vàng'}

# ── Ví dụ CỐ ĐỊNH (curate) cho khóa luận — tái lập được, không phụ thuộc quét ──
# Mỗi mục ghim (img_id, chuỗi-khớp-câu-hỏi) để lấy ĐÚNG dòng (1 ảnh có nhiều câu hỏi).
CURATED_CORRECT = [
    ('445048', 'trượt tuyết xuống tuyết bao phủ'),   # OBJECT → núi
    ('168217', 'gấu bông đang ngồi trên sàn nhà'),    # COUNT  → hai
    ('418479', 'màu của con chim'),                    # COLOR  → xám
]
CURATED_WRONG = [
    ('545250', 'xếp hàng',
     'Xác định sai đối tượng: dãy tủ lạnh cũ bị vứt ngoài bãi, model bắt nhầm '
     'sang chiếc máy xúc/xe cơ giới màu vàng ở phía sau nên trả lời "xe tải" '
     'thay vì tủ lạnh.'),
    ('317391', 'hươu cao cổ',
     'Đếm sai do che khuất: có 4 con hươu cao cổ nhưng chúng đứng sát và che '
     'khuất nhau dưới tán cây, một con bị thân con bên cạnh che gần hết nên '
     'model chỉ đếm được 3.'),
    ('327841', 'màu của xe tải',
     'Nhầm màu sang đối tượng khác: ảnh có nhiều xe tải, model bắt màu đỏ của '
     'chiếc xe/khung phía sau thay vì màu xanh lá của xe tải chính ở tiền cảnh.'),
]


def _to_num(s):
    s = _normalize_vn(s).replace('màu', '').strip()
    if s.isdigit():
        return int(s)
    return _NUM_WORDS.get(s, None)


def _is_color(s):
    s = _normalize_vn(s).replace('màu', '').strip()
    return s in _COLORS


def score_wrong(q, gt, pred, t):
    """Điểm cao = lỗi DỄ GIẢI THÍCH cho khóa luận. Kèm nhận xét ngắn."""
    if not pred.strip():
        return -1, ''                       # dự đoán rỗng → loại
    n_words = len(q.split())
    penalty = 8 if n_words > 16 else 0       # câu quá dài thường là nhãn nhiễu
    if t == 1:                               # COUNT
        a, b = _to_num(gt), _to_num(pred)
        if a is not None and b is not None:
            if abs(a - b) == 1:
                return 100 - penalty, f"Đếm lệch 1 đơn vị ({gt} → {pred})"
            return 65 - penalty, f"Đếm sai ({gt} → {pred})"
        return 30 - penalty, ''
    if t == 2:                               # COLOR
        if _is_color(gt) and _is_color(pred):
            return 90 - penalty, f"Nhầm màu ({gt} → {pred})"
        return 45 - penalty, ''
    if t == 0:                               # OBJECT
        return 55 - penalty, f"Nhận nhầm đối tượng ({gt} → {pred})"
    if t == 3:                               # LOCATION
        return 50 - penalty, f"Xác định sai vị trí ({gt} → {pred})"
    return 20, ''


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', default='checkpoints_run87/best_model.pt')
    p.add_argument('--csv_path', default='archive/test.csv')
    p.add_argument('--image_folder', default='archive/data/images/test')
    p.add_argument('--vision_model', default='google/siglip-base-patch16-224')
    p.add_argument('--train_csv_for_trie', default='archive/train_split.csv')
    p.add_argument('--n', type=int, default=3, help='Số ví dụ mỗi hình')
    p.add_argument('--max_q_len', type=int, default=32)
    p.add_argument('--max_length', type=int, default=10)
    p.add_argument('--num_beams', type=int, default=3)
    p.add_argument('--repetition_penalty', type=float, default=1.3)
    p.add_argument('--use_synonyms', action='store_true', default=True)
    p.add_argument('--scan', type=int, default=600,
                   help='Số mẫu quét để gom ứng viên (0 = quét hết test.csv)')
    p.add_argument('--correct_ids', nargs='*', default=None)
    p.add_argument('--wrong_ids', nargs='*', default=None)
    p.add_argument('--rescan', action='store_true',
                   help='Quét test.csv tự chọn ví dụ (mặc định: dùng ví dụ curate cố định)')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--outdir', default='figs')
    return p.parse_args()


def load_sample(row, image_folder, processor, tokenizer, max_q_len, device):
    img_id = str(row['img_id'])
    pil = Image.open(os.path.join(image_folder, f"{img_id}.jpg")).convert('RGB')
    px = processor(images=pil, return_tensors='pt')['pixel_values'].to(device)
    q = str(row['question'])
    enc = tokenizer(q, truncation=True, padding='max_length',
                    max_length=max_q_len, return_tensors='pt')
    return {
        'pixel_values': px,
        'input_ids': enc['input_ids'].to(device),
        'attention_mask': enc['attention_mask'].to(device),
        'pil': pil, 'img_id': img_id, 'question': q,
        'gt': str(row['answer']),
        'type': int(row['type']) if 'type' in row else 0,
    }


@torch.no_grad()
def predict(model, s, args, prefix_trie, valid_set):
    out = model.generate(
        pixel_values=s['pixel_values'], input_ids=s['input_ids'],
        attention_mask=s['attention_mask'], max_length=args.max_length,
        num_beams=args.num_beams, repetition_penalty=args.repetition_penalty,
        prefix_trie=prefix_trie,
    )
    pred = out[0]
    if valid_set is not None:
        pred = snap_to_valid_answer(pred, valid_set)
    return pred


def _wrap(txt, w=42):
    return '\n'.join(textwrap.wrap(txt, width=w)) or txt


def draw_grid(examples, outpath, correct, title):
    """Mỗi ví dụ 1 hàng: [ảnh | chú thích]."""
    n = len(examples)
    if n == 0:
        print(f"[viz] ⚠️ Không có ví dụ cho: {title} — bỏ qua {outpath}")
        return
    edge = '#1a9850' if correct else '#d73027'

    fig, axes = plt.subplots(n, 2, figsize=(12, 3.9 * n),
                             gridspec_kw={'width_ratios': [1.0, 1.15]},
                             squeeze=False)
    for i, e in enumerate(examples):
        ax_img, ax_txt = axes[i, 0], axes[i, 1]
        # Ảnh + khung màu
        ax_img.imshow(e['pil'])
        ax_img.add_patch(Rectangle((0.005, 0.005), 0.99, 0.99, transform=ax_img.transAxes,
                                   fill=False, edgecolor=edge, linewidth=6))
        ax_img.set_xticks([]); ax_img.set_yticks([])
        for sp in ax_img.spines.values():
            sp.set_visible(False)

        # Chú thích
        ax_txt.axis('off')
        tname = f"{TYPE_NAMES_VI[e['type']]}  ({TYPE_NAMES[e['type']]})"
        lines = [
            ('Loại câu hỏi', tname),
            ('Câu hỏi', _wrap(e['question'])),
            ('Đáp án đúng', e['gt']),
            ('Dự đoán', e['pred']),
        ]
        y = 0.93
        for label, val in lines:
            ax_txt.text(0.0, y, f"{label}:", fontsize=16, fontweight='bold',
                        va='top', transform=ax_txt.transAxes)
            ax_txt.text(0.36, y, val, fontsize=16, va='top',
                        color=(edge if label.startswith('Dự đoán') else 'black'),
                        transform=ax_txt.transAxes)
            y -= 0.165 + 0.085 * val.count('\n')
        if (not correct) and e.get('note'):
            note = '\n'.join(textwrap.wrap('Giải thích lỗi: ' + e['note'], width=44))
            ax_txt.text(0.0, y - 0.03, note, fontsize=14.5, style='italic',
                        color=edge, va='top', transform=ax_txt.transAxes)

    fig.suptitle(title, fontsize=20, fontweight='bold', color=edge, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    os.makedirs(os.path.dirname(outpath) or '.', exist_ok=True)
    fig.savefig(outpath, dpi=190, bbox_inches='tight')
    print(f"✅ Đã lưu: {outpath}  ({n} ví dụ)")


def pick_diverse(cands, n):
    """cands: list dict có 'type'. Trải đều 4 loại (round-robin)."""
    by_t = {t: [] for t in range(4)}
    for c in cands:
        by_t[c['type']].append(c)
    picked, ti = [], 0
    while len(picked) < n and any(by_t.values()):
        t = ti % 4
        if by_t[t]:
            picked.append(by_t[t].pop(0))
        ti += 1
    return picked[:n]


def main():
    args = parse_args()
    random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model, info = build_model_from_checkpoint(args.checkpoint, args.vision_model, device)
    processor = AutoImageProcessor.from_pretrained(args.vision_model)
    tokenizer = BartphoTokenizer.from_pretrained('vinai/bartpho-syllable')

    print("[viz] Dựng answer trie (constrained decoding)...")
    prefix_trie = build_answer_trie(args.train_csv_for_trie, model.tokenizer)
    valid_set = build_valid_answers_set(args.train_csv_for_trie)

    import pandas as pd
    df = pd.read_csv(args.csv_path)

    def run_row(row, note=''):
        s = load_sample(row, args.image_folder, processor, tokenizer, args.max_q_len, device)
        s['pred'] = predict(model, s, args, prefix_trie, valid_set)
        s['em'] = compute_exact_match(s['pred'], s['gt'], args.use_synonyms)
        s['note'] = note
        return s

    def find_row(iid, qsub=None):
        sub = df[df['img_id'].astype(str) == str(iid)]
        if qsub:
            sub = sub[sub['question'].str.contains(qsub, case=False, na=False, regex=False)]
        if sub.empty:
            print(f"[viz] ⚠️ không thấy img_id={iid} (qsub={qsub!r})")
            return None
        return sub.iloc[0]

    if args.rescan:
        # ── Quét test.csv, chấm điểm chọn ví dụ (fallback) ───────────────────
        order = list(range(len(df)))
        random.shuffle(order)
        limit = args.scan if args.scan > 0 else len(order)
        corr_c, wrong_c = [], []
        for idx in order[:limit]:
            row = df.iloc[idx]
            try:
                s = run_row(row)
            except Exception:
                continue
            sc, _ = score_wrong(s['question'], s['gt'], s['pred'], s['type'])
            if s['em'] == 1.0:
                s['score'] = 50 - max(0, len(s['question'].split()) - 12)
                corr_c.append(s)
            elif sc > 0:
                s['score'] = sc
                wrong_c.append(s)
        corr_c.sort(key=lambda x: -x['score'])
        wrong_c.sort(key=lambda x: -x['score'])
        corr = pick_diverse(corr_c, args.n)
        wrong = pick_diverse(wrong_c, args.n)
    elif args.correct_ids or args.wrong_ids:
        # ── img_id chỉ định thủ công ─────────────────────────────────────────
        corr, wrong = [], []
        for iid in (args.correct_ids or []):
            r = find_row(iid)
            if r is not None:
                s = run_row(r)
                (corr if s['em'] == 1.0 else wrong).append(s)
        for iid in (args.wrong_ids or []):
            r = find_row(iid)
            if r is not None:
                s = run_row(r)
                (wrong if s['em'] == 0.0 else corr).append(s)
    else:
        # ── MẶC ĐỊNH: ví dụ curate cố định + giải thích bám ảnh ──────────────
        corr, wrong = [], []
        for iid, qsub in CURATED_CORRECT:
            r = find_row(iid, qsub)
            if r is not None:
                s = run_row(r)
                if s['em'] == 1.0:
                    corr.append(s)
                else:
                    print(f"[viz] ⚠️ {iid} giờ dự đoán ĐÚNG (pred={s['pred']}) — bỏ khỏi 'đúng'")
        for iid, qsub, expl in CURATED_WRONG:
            r = find_row(iid, qsub)
            if r is not None:
                s = run_row(r, note=expl)
                if s['em'] == 0.0:
                    wrong.append(s)
                else:
                    print(f"[viz] ⚠️ {iid} giờ dự đoán ĐÚNG (pred={s['pred']}) — không còn là 'sai'")

    draw_grid(corr, os.path.join(args.outdir, 'fig_correct.png'), True,
              'Ví dụ dự đoán ĐÚNG')
    draw_grid(wrong, os.path.join(args.outdir, 'fig_wrong.png'), False,
              'Ví dụ dự đoán SAI')


if __name__ == '__main__':
    main()
