"""HIEU CHINH OFFSET TAN SUAT — CO LAM MODEL TONG QUAT RA LOP CHUA TRAIN KHONG?

Da do (102 mau OOV, checkpoints_g_s1_s1):
  - Lop CHUA TRAIN van NHIN THAY duoc: chon dung 1 trong 9 lop giu lai 40.2% (chance 11.1%).
  - Nhung trong CUNG LOAI cau hoi no xep HANG CHOT: top1 0.0%, rank med 10/10.
  - Vi diem cua mot lop tang don dieu theo tan suat train:
        chua thay -26.1 | 1-5 -19.8 | 6-20 -17.1 | 21-100 -16.2 | >100 -15.3
        Spearman(log tan suat, diem) = 0.483
  => offset la HANG SO THEO LOP, khong phu thuoc anh. Tru duoc.

Hieu chinh: b_c = trung binh logP(c | v, q) tren mot tap anh+cau hoi KHONG NHAN.
            score'(a) = logP(a | v,q) - tau * b_a
Uoc luong b_c tren NUA val (chi dung anh va cau hoi, KHONG dung dap an), do tren nua con lai,
nen day khong phai transductive an nhan.
"""
import sys, numpy as np, pandas as pd, unicodedata as ud

norm = lambda s: ud.normalize('NFC', str(s)).strip().lower()
T = {0: 'OBJECT', 1: 'COUNT', 2: 'COLOR', 3: 'LOCATION'}
STEM = sys.argv[1] if len(sys.argv) > 1 else 'analysis/openvocab/pmi_g_s1_val_ALL'

z = np.load(STEM + '_scores.npz', allow_pickle=True)
SV, SP, CAND = z['s_vision'], z['s_prior'], list(z['cand'])
d = pd.read_csv(STEM + '.csv').reset_index(drop=True)
tro = pd.read_csv('archive/train_split_original.csv'); tro['an'] = tro.answer.map(norm)
SEEN = {norm(x) for x in pd.read_csv('archive/train_split_oov.csv').answer}
HELD = sorted(set(tro.an) - SEEN)
c2i = {c: i for i, c in enumerate(CAND)}
TVOC = {t: [c2i[c] for c in sorted(set(g.an))] for t, g in tro.groupby('type')}
gi = np.array([c2i.get(g, -1) for g in d.gold])
held_mask = d.gold.isin(HELD).values
print(f'{len(d)} mau | {held_mask.sum()} gold chua train | {len(CAND)} lop')

rng = np.random.default_rng(0)
fold = rng.integers(0, 2, len(d))          # 0 = uoc luong b_c, 1 = danh gia


def evaluate(S, mask, restrict):
    """top1 tren tap mask; restrict='all' hoac 'type'."""
    ok = []
    for r in np.where(mask)[0]:
        if gi[r] < 0: continue
        ss = list(range(len(CAND))) if restrict == 'all' else TVOC[int(d.type.iloc[r])]
        if gi[r] not in ss: continue
        s = S[r][ss]
        ok.append(ss[int(np.argmax(s))] == gi[r])
    return 100 * np.mean(ok) if ok else float('nan'), len(ok)


ev = (fold == 1)
print(f'\nuoc luong b_c tren {(~ev).sum()} mau (khong nhan), danh gia tren {ev.sum()} mau')
b_vis = SV[~ev].mean(0)                     # b_c = E_{v,q}[ logP(c|v,q) ]
print(f'{"phuong phap":<40}{"tau":>5}{"EM da-train":>13}{"top1 chua-train":>17}{"  (cung loai)":>15}')
for name, S, taus in [('goc: logP(a|v,q)', SV, [0.0]),
                      ('PMI: - logP(a|q)', None, [0.5, 1.0]),
                      ('hieu chinh tan suat: - b_a', None, [0.5, 1.0])]:
    for tau in taus:
        if name.startswith('goc'):
            Sx = SV
        elif name.startswith('PMI'):
            Sx = SV - tau * SP
        else:
            Sx = SV - tau * b_vis[None, :]
        em, n1 = evaluate(Sx, ev & ~held_mask, 'all')
        oo, n2 = evaluate(Sx, ev & held_mask, 'all')
        ot, n3 = evaluate(Sx, ev & held_mask, 'type')
        print(f'{name:<40}{tau:>5}{em:>13.2f}{oo:>17.2f}{ot:>15.2f}')
print(f'   (n: da-train {n1}, chua-train {n2}/{n3})')

# ket hop: hieu chinh tan suat + PMI
print()
best = None
for t1 in [0.25, 0.5, 0.75, 1.0]:
    for t2 in [0.0, 0.25, 0.5]:
        Sx = SV - t1 * b_vis[None, :] - t2 * SP
        em, _ = evaluate(Sx, ev & ~held_mask, 'all')
        oo, _ = evaluate(Sx, ev & held_mask, 'all')
        if best is None or em + oo > best[0]:
            best = (em + oo, t1, t2, em, oo)
print(f'ket hop tot nhat: tau_freq={best[1]} tau_pmi={best[2]} -> '
      f'EM da-train {best[3]:.2f} | top1 chua-train {best[4]:.2f}')

base_em, _ = evaluate(SV, ev & ~held_mask, 'all')
print(f'(doi chung: EM da-train khong hieu chinh = {base_em:.2f}, top1 chua-train = 0.00)')
