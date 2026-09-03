"""THEM TYPE THI DUOC GI? — kiem gia thuyet cua giang vien bang du lieu DA CO, khong can GPU.

Gia thuyet cua giang vien (muc 7): T0 tra loi SAI LOAI (hoi mau ma tra ve so luong),
va them type thi sua duoc. Nghi van cua tac gia: "can ban no chi tra loi sai chu dau tra loi
lon loai dau".

Do truc tiep. Nhan LOAI cho tung CHUOI DAP AN duoc suy tu chinh du lieu train (dap an nay
thuong di voi loai cau hoi nao), khong gan tay:
    answer -> loai chiem da so trong train_split_original.csv
Sau do voi moi loi cua T0: dap an du doan co thuoc DUNG loai cua cau hoi khong?
    - "sai loai"  = du doan thuoc mot loai khac (hoi mau -> tra ve so)
    - "dung loai, sai gia tri" = du doan dung ho nhung sai (hoi mau -> mau khac)

Neu ti le "sai loai" gan 0 thi type-conditioning KHONG CO GI DE SUA, va do la loi giai thich
cho T1-T0 = -0.14 (n=10, p=0.35).

So sanh T0 (khong gate, khong type_loss) voi T2 (co ca hai) tren CUNG encoder SigLIP1,
cung seed — day la cap duy nhat co hieu so chi khac dung phan can hoi.
"""
import pandas as pd, numpy as np, unicodedata as ud, os
from collections import Counter

T = {0: 'OBJECT', 1: 'COUNT', 2: 'COLOR', 3: 'LOCATION'}
norm = lambda s: ud.normalize('NFC', str(s)).strip().lower()

tr = pd.read_csv('archive/train_split_original.csv')
tr['a'] = tr.answer.map(norm)
# loai CHIEM DA SO cua moi chuoi dap an, suy tu du lieu train
maj = tr.groupby('a').type.agg(lambda s: Counter(s).most_common(1)[0][0]).to_dict()
pur = tr.groupby('a').type.agg(lambda s: Counter(s).most_common(1)[0][1] / len(s)).to_dict()
print(f'{len(maj)} chuoi dap an | do thuan loai trung binh {np.mean(list(pur.values()))*100:.1f}%')
print(f'  so dap an thuan 100% mot loai: {sum(1 for v in pur.values() if v > 0.999)}/{len(pur)}')

SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 42]
rows = []
for s in SEEDS:
    f0, f2 = f'analysis/T0/T0_seed{s}.csv', f'beam3fixed/seed{s}_ep40.csv'
    if not (os.path.exists(f0) and os.path.exists(f2)):
        continue
    d0, d2 = pd.read_csv(f0), pd.read_csv(f2)
    if len(d0) != len(d2):
        continue
    gt_t = d0.question_type.map({v: k for k, v in T.items()})
    for tag, d in (('T0', d0), ('T2', d2)):
        p = d.prediction.map(norm)
        pt = p.map(lambda a: maj.get(a, -1))
        err = d.exact_match.values <= 0.5
        known = pt.values >= 0
        wrong_type = err & known & (pt.values != gt_t.values)
        rows.append(dict(seed=s, arm=tag, n_err=int(err.sum()),
                         n_wrong_type=int(wrong_type.sum()),
                         pct=100 * wrong_type.sum() / max(err.sum(), 1),
                         n_unknown=int((err & ~known).sum())))
R = pd.DataFrame(rows)
if R.empty:
    raise SystemExit('khong ghep duoc cap T0/T2 nao')

print(f'\n=== TRONG SO LOI, BAO NHIEU LA "SAI LOAI" ===')
g = R.groupby('arm').agg(loi=('n_err', 'mean'), sai_loai=('n_wrong_type', 'mean'),
                         pct=('pct', 'mean'), pct_sd=('pct', 'std'), n=('seed', 'count'))
print(g.round(2).to_string())

print(f'\n=== CHI TIET THEO SEED ===')
print(R.pivot(index='seed', columns='arm', values='pct').round(2).to_string())

# phan ra theo loai cau hoi, tren T0
print(f'\n=== T0: ti le "sai loai" trong loi, TACH THEO LOAI CAU HOI ===')
acc = {t: [] for t in T.values()}
for s in SEEDS:
    f0 = f'analysis/T0/T0_seed{s}.csv'
    if not os.path.exists(f0):
        continue
    d = pd.read_csv(f0)
    gt_t = d.question_type.map({v: k for k, v in T.items()})
    pt = d.prediction.map(norm).map(lambda a: maj.get(a, -1))
    err = d.exact_match.values <= 0.5
    for ti, tn in T.items():
        m = err & (gt_t.values == ti) & (pt.values >= 0)
        if m.sum():
            acc[tn].append(100 * (pt.values[m] != ti).sum() / m.sum())
print(f'{"loai":10s} {"% loi la sai loai":>20s}   {"n seed":>7s}')
for tn, v in acc.items():
    if v:
        print(f'{tn:10s} {np.mean(v):19.2f}%   {len(v):7d}')

print('\n=> Neu ti le nay nho thi loi CHU YEU la "dung ho, sai gia tri", va type-conditioning')
print('   khong co gi de sua — khop voi T1-T0 = -0.14 (p=0.35, n=10).')
