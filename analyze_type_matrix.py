"""CONFUSION MATRIX: LOAI CAU HOI  x  LOAI CUA DAP AN MODEL DUA RA.

Day la ma tran giang vien yeu cau: "cau hoi mau nhung tra loi so luong".
Nhan loai cho tung CHUOI dap an suy tu chinh du lieu train (dap an nay thuong di voi loai
cau hoi nao), chi giu 234/331 chuoi THUAN 100% mot loai de khong dem nhiem nhan.

In 3 ma tran, deu trung binh 10 seed (SigLIP1, T0 va T2 ghep cung seed):
  A. tren TAT CA du doan      -> nhin tong the
  B. chi tren cac mau SAI     -> khi sai thi sai kieu gi (day moi la cho co thong tin)
  C. hieu T2 - T0             -> them gate + type_loss doi gi
"""
import pandas as pd, numpy as np, unicodedata as ud, os
from collections import Counter

T = ['OBJECT', 'COUNT', 'COLOR', 'LOCATION']
IDX = {t: i for i, t in enumerate(T)}
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 42]
norm = lambda s: ud.normalize('NFC', str(s)).strip().lower()

tr = pd.read_csv('archive/train_split_original.csv'); tr['a'] = tr.answer.map(norm)
maj = tr.groupby('a').type.agg(lambda s: Counter(s).most_common(1)[0][0]).to_dict()
pur = tr.groupby('a').type.agg(lambda s: Counter(s).most_common(1)[0][1] / len(s)).to_dict()
PURE = {a for a, v in pur.items() if v > 0.999}
print(f'nhan loai dap an: giu {len(PURE)}/{len(pur)} chuoi thuan 100% mot loai')

def matrices(arm):
    """tra ve (ma tran tren tat ca, ma tran chi tren mau SAI), trung binh cac seed, don vi %"""
    allm, errm = [], []
    for s in SEEDS:
        f = f'analysis/T0/T0_seed{s}.csv' if arm == 'T0' else f'beam3fixed/seed{s}_ep40.csv'
        if not os.path.exists(f):
            continue
        d = pd.read_csv(f)
        p = d.prediction.map(norm)
        keep = p.isin(PURE).values
        pt = p.map(lambda a: maj.get(a, -1)).values
        qt = d.question_type.map(IDX).values
        err = d.exact_match.values <= .5
        A = np.zeros((4, 4)); E = np.zeros((4, 4))
        for i in range(4):
            ma = keep & (qt == i)
            me = ma & err
            for j in range(4):
                A[i, j] = (pt[ma] == j).sum()
                E[i, j] = (pt[me] == j).sum()
        allm.append(A / A.sum(1, keepdims=True).clip(min=1) * 100)
        errm.append(E / E.sum(1, keepdims=True).clip(min=1) * 100)
    return np.mean(allm, 0), np.mean(errm, 0)

def show(M, title, diff=False):
    print(f'\n{title}')
    print(f'{"cau hoi \\ dap an":20s}' + ''.join(f'{t:>11s}' for t in T))
    for i, t in enumerate(T):
        row = f'{t:20s}'
        for j in range(4):
            v = M[i, j]
            row += f'{v:+10.2f} ' if diff else f'{v:10.2f} '
        print(row)

A0, E0 = matrices('T0')
A2, E2 = matrices('T2')
show(A0, 'A1. T0 — TAT CA du doan (% moi hang)')
show(A2, 'A2. T2 — TAT CA du doan (% moi hang)')
show(E0, 'B1. T0 — chi cac mau SAI (% moi hang)')
show(E2, 'B2. T2 — chi cac mau SAI (% moi hang)')
show(E2 - E0, 'C. HIEU T2 - T0 tren cac mau SAI (diem phan tram)', diff=True)

print('\nDuong cheo = tra loi DUNG LOAI (chi sai gia tri). Ngoai duong cheo = tra loi LON LOAI.')
print('Doc B1/B2: ngay ca khi SAI, model van tra loi dung LOAI o hau het truong hop.')
