"""MA TRAN PHAM TRU DAP AN theo loai cau hoi, T0 vs T2, 9 seed — muc "still planned #1".

Cau hoi: cau hoi loai X co bi tra loi bang dap an thuoc pham tru KHAC khong (vi du hoi mau ma
tra ra mot con so), va gate co sua duoc kieu loi do khong.

Pham tru cua mot dap an duoc suy tu TRAIN: dap an do xuat hien cung loai cau hoi nao nhieu nhat.
CANH BAO da biet: OBJECT va LOCATION dung chung ~81% tu vung, nen hai pham tru do KHONG tach
duoc sach. Moi dap an co ty le loai cao nhat < 0.70 bi danh dau NHAP NHANG va tach rieng ra,
khong duoc gop im lang vao mot pham tru.
"""
import os, unicodedata as ud, numpy as np, pandas as pd
from scipy import stats

norm = lambda s: ud.normalize('NFC', str(s)).strip().lower()
T = ['OBJECT', 'COUNT', 'COLOR', 'LOCATION']
SEEDS = list(range(10))

tr = pd.read_csv('archive/train_split.csv')
tr['an'] = tr.answer.map(norm)
TMAP = {0: 'OBJECT', 1: 'COUNT', 2: 'COLOR', 3: 'LOCATION'}
tr['tname'] = tr.type.map(TMAP)
share = tr.groupby(['an', 'tname']).size().unstack(fill_value=0)
frac = share.div(share.sum(1), axis=0)
CAT = {}
for a in frac.index:
    top = frac.loc[a].idxmax()
    CAT[a] = top if frac.loc[a, top] >= 0.70 else 'NHAP NHANG'
amb = sum(1 for v in CAT.values() if v == 'NHAP NHANG')
print(f'{len(CAT)} dap an trong train | {amb} nhap nhang (>=2 loai, khong loai nao >=70%)')
print('  vi du nhap nhang:', [a for a, v in CAT.items() if v == 'NHAP NHANG'][:8])
for t in T:
    print(f'  pham tru {t:<9}: {sum(1 for v in CAT.values() if v==t):>4} dap an')


def load(arm, s):
    f = f'analysis/T0/T0_seed{s}.csv' if arm == 'T0' else f'beam3fixed/seed{s}_ep40.csv'
    if not os.path.exists(f):
        return None
    d = pd.read_csv(f)
    d['hit'] = (d.exact_match > 0.5).astype(int)
    d['pcat'] = d.prediction.map(lambda x: CAT.get(norm(x), 'NGOAI TU VUNG'))
    d['gcat'] = d.ground_truth.map(lambda x: CAT.get(norm(x), 'NGOAI TU VUNG'))
    return d


A = {(a, s): load(a, s) for a in ('T0', 'T2') for s in SEEDS}
A = {k: v for k, v in A.items() if v is not None}
pairs = [s for s in SEEDS if ('T0', s) in A and ('T2', s) in A]
print(f'\n{len(pairs)} cap seed ghep duoc\n')
COLS = T + ['NHAP NHANG', 'NGOAI TU VUNG']

for arm in ('T0', 'T2'):
    M = np.zeros((len(T), len(COLS)))
    for s in pairs:
        d = A[(arm, s)]
        for i, t in enumerate(T):
            sub = d[d.question_type == t]
            for j, c in enumerate(COLS):
                M[i, j] += (sub.pcat == c).sum()
    M /= len(pairs)
    print(f'=== {arm}: cau hoi loai (hang) -> pham tru dap an sinh ra (cot), tb {len(pairs)} seed ===')
    print(f'{"":<10}' + ''.join(f'{c:>13}' for c in COLS))
    for i, t in enumerate(T):
        print(f'{t:<10}' + ''.join(f'{100*M[i,j]/M[i].sum():>12.2f}%' for j in range(len(COLS))))
    print()

print('=== SAI PHAM TRU (dap an sinh ra khac pham tru cua GOLD), % trong loai ===')
print(f'{"loai":<10}{"T0":>8}{"T2":>8}{"delta":>9}{"p":>8}{"am":>7}')
for t in T + ['TAT CA']:
    r = {}
    for arm in ('T0', 'T2'):
        v = []
        for s in pairs:
            d = A[(arm, s)]
            sub = d if t == 'TAT CA' else d[d.question_type == t]
            sub = sub[sub.gcat != 'NGOAI TU VUNG']
            v.append(100 * (sub.pcat != sub.gcat).mean())
        r[arm] = np.array(v)
    dd = r['T2'] - r['T0']
    p = stats.ttest_rel(r['T2'], r['T0']).pvalue
    print(f'{t:<10}{r["T0"].mean():>8.2f}{r["T2"].mean():>8.2f}{dd.mean():>+9.2f}{p:>8.3f}'
          f'{int((dd<0).sum()):>4}/{len(dd)}')

print('\n=== TRONG SO CAU SAI: loi la cung pham tru hay khac pham tru ===')
print(f'{"loai":<10}{"T0 sai":>9}{"cung pt":>10}{"T2 sai":>9}{"cung pt":>10}')
for t in T:
    row = []
    for arm in ('T0', 'T2'):
        ne, sa = [], []
        for s in pairs:
            d = A[(arm, s)]
            sub = d[(d.question_type == t) & (d.hit == 0) & (d.gcat != 'NGOAI TU VUNG')]
            ne.append(len(sub)); sa.append(100 * (sub.pcat == sub.gcat).mean())
        row += [np.mean(ne), np.mean(sa)]
    print(f'{t:<10}{row[0]:>9.1f}{row[1]:>9.1f}%{row[2]:>9.1f}{row[3]:>9.1f}%')
