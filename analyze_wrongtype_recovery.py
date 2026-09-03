"""CAU HOI CUA THAY: neu T0 LON LOAI thi T2 co go duoc khong?

Day la mot XAC SUAT CO DIEU KIEN, khong phai ti le tong:
    P(T2 dung | T0 lon loai)

Va no chi doc duoc khi dat canh MOC SO SANH DUNG:
    P(T2 dung | T0 sai nhung DUNG loai)      <- lo?i thu?ng
Neu hai con so nay bang nhau thi type KHONG go duoc gi RIENG cho lon loai — T2 chi go
duoc mot ti le loi chung chung, khong phan biet loi do thuoc kieu nao.

Va con mot moc nua bat buoc, vi trong du an nay hai lan chay khac seed da churn 230 mau:
    P(T0' dung | T0 lon loai)   voi T0' la MOT SEED T0 KHAC — cung cau hinh, khong co type.
Neu T2 khong hon con so nay thi "go duoc" chi la seed churn.

Nhan loai dap an suy tu train, chi giu chuoi THUAN 100% mot loai.
"""
import pandas as pd, numpy as np, unicodedata as ud, os
from collections import Counter
from itertools import permutations
from scipy import stats

SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 42]
T = ['OBJECT', 'COUNT', 'COLOR', 'LOCATION']
IDX = {t: i for i, t in enumerate(T)}
norm = lambda s: ud.normalize('NFC', str(s)).strip().lower()

tr = pd.read_csv('archive/train_split_original.csv'); tr['a'] = tr.answer.map(norm)
maj = tr.groupby('a').type.agg(lambda s: Counter(s).most_common(1)[0][0]).to_dict()
pur = tr.groupby('a').type.agg(lambda s: Counter(s).most_common(1)[0][1] / len(s)).to_dict()
PURE = {a for a, v in pur.items() if v > 0.999}

def load(arm, s):
    f = f'analysis/T0/T0_seed{s}.csv' if arm == 'T0' else f'beam3fixed/seed{s}_ep40.csv'
    return pd.read_csv(f) if os.path.exists(f) else None

D = {}
for arm in ('T0', 'T2'):
    for s in SEEDS:
        d = load(arm, s)
        if d is None:
            continue
        p = d.prediction.map(norm)
        D[(arm, s)] = dict(
            ok=d.exact_match.values > .5,
            qt=d.question_type.map(IDX).values,
            pt=p.map(lambda a: maj.get(a, -1)).values,
            pure=p.isin(PURE).values)

def rates(a_key, b_key):
    """tra ve (n_lon_loai, ti le b go duoc | a lon loai, n_thuong, ti le b go duoc | a sai dung loai)"""
    A, B = D[a_key], D[b_key]
    err = ~A['ok'] & A['pure']
    wrong_t = err & (A['pt'] != A['qt'])
    right_t = err & (A['pt'] == A['qt'])
    return (int(wrong_t.sum()), 100 * B['ok'][wrong_t].mean() if wrong_t.sum() else np.nan,
            int(right_t.sum()), 100 * B['ok'][right_t].mean() if right_t.sum() else np.nan)

print('=== T2 go duoc gi, TUY THEO T0 sai kieu nao (ghep cung seed) ===')
print(f'{"seed":>5} {"n lon loai":>11} {"T2 go duoc":>12} {"n sai thuong":>13} {"T2 go duoc":>12}')
rw, rr, nw = [], [], []
for s in SEEDS:
    if ('T0', s) not in D or ('T2', s) not in D:
        continue
    a, b, c, d = rates(('T0', s), ('T2', s))
    print(f'{s:5d} {a:11d} {b:11.1f}% {c:13d} {d:11.1f}%')
    rw.append(b); rr.append(d); nw.append(a)
rw, rr = np.array(rw), np.array(rr)
print(f'\n  trung binh: lon loai {rw.mean():.1f}% +/- {rw.std(ddof=1):.1f}'
      f'   |  sai thuong {rr.mean():.1f}% +/- {rr.std(ddof=1):.1f}')
t = stats.ttest_rel(rw, rr)
print(f'  hieu = {rw.mean()-rr.mean():+.1f} pp | p={t.pvalue:.3f} | lon loai cao hon o '
      f'{int((rw>rr).sum())}/{len(rw)} seed')
print(f'  (moi seed chi co trung binh {np.mean(nw):.1f} ca lon loai — mau rat nho)')

print('\n=== DOI CHUNG: mot seed T0 KHAC go duoc bao nhieu (khong he co type) ===')
cw, cr = [], []
for i, j in permutations(SEEDS, 2):
    if ('T0', i) not in D or ('T0', j) not in D:
        continue
    a, b, c, d = rates(('T0', i), ('T0', j))
    if a: cw.append(b)
    if c: cr.append(d)
print(f'  lon loai   : {np.mean(cw):.1f}% +/- {np.std(cw,ddof=1):.1f}   ({len(cw)} cap)')
print(f'  sai thuong : {np.mean(cr):.1f}% +/- {np.std(cr,ddof=1):.1f}')
print(f'\n  => T2 go lon loai {rw.mean():.1f}% vs mot seed T0 khac go {np.mean(cw):.1f}%')
print('     Neu hai con so nay tuong duong thi "them type go duoc lon loai" KHONG dung —')
print('     do chi la muc churn ma bat ky lan chay lai nao cung co.')
