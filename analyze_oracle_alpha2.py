"""alpha_oracle NEN cai gi? -- theo loai, va co dinh huong "bo bot ban sao" khong.

Phat hien o buoc 1: alpha_oracle NEN patch VAT THE nhieu hon patch NEN (-0.0156, t=-22.8).
Nguoc hoan toan voi truc giac "tap trung vao vat the". Buoc nay tach theo loai va kiem gia
thuyet co che: voi COUNT, nen bot patch trong CUNG mot vung = lam thua vat the (de dem hon).
"""
import numpy as np, pandas as pd, pickle
from math import sqrt

Z = np.load('analysis/oracle_alpha/train_alpha.npz')
P = Z['alpha'].astype('float32')[:, 1:]          # bo token toan cuc
df = pd.read_csv('archive/train_split_original.csv')
rm = pickle.load(open('patch_region_map.pkl', 'rb'))
ids = df['img_id'].astype(int).to_numpy()
types = df['type'].to_numpy()
TN = {0: 'OBJECT', 1: 'COUNT', 2: 'COLOR', 3: 'LOCATION'}

print('=' * 78)
print('A. alpha(vat the) - alpha(nen), TACH THEO LOAI')
print('=' * 78)
print(f'  {"loai":<10} {"n":>6} {"delta":>9} {"t":>8} {"%duong":>8}')
for t in sorted(np.unique(types)):
    d = []
    for i in np.where(types == t)[0]:
        r = rm.get(int(ids[i]))
        if r is None or r.shape[0] != 196: continue
        m = r > 0
        if m.sum() == 0 or (~m).sum() == 0: continue
        d.append(P[i][m].mean() - P[i][~m].mean())
    d = np.array(d)
    if len(d) < 2: continue
    tt = d.mean() / (d.std(ddof=1) / sqrt(len(d)))
    print(f'  {TN.get(int(t), t):<10} {len(d):>6} {d.mean():>+9.4f} {tt:>+8.1f} {(d > 0).mean() * 100:>7.1f}%')

print('\n' + '=' * 78)
print('B. Nen TRONG vung hay nen CA vung? (thua vat the vs bo han vat the)')
print('=' * 78)
print('  Voi moi mau: xet cac vung COCO co >=4 patch. Do ty le patch bi nen trong vung,')
print('  roi xem phan bo: gan 0/1 = bo han hoac giu han ca vung; o giua = THUA BOT trong vung.')
for t in sorted(np.unique(types)):
    fr = []
    for i in np.where(types == t)[0]:
        r = rm.get(int(ids[i]))
        if r is None or r.shape[0] != 196: continue
        for rid in np.unique(r):
            if rid <= 0: continue
            m = r == rid
            if m.sum() < 4: continue
            fr.append((P[i][m] < 0.5).mean())
    fr = np.array(fr)
    if len(fr) < 10: continue
    part = ((fr > 0.2) & (fr < 0.8)).mean()
    print(f'  {TN.get(int(t), t):<10} n_vung={len(fr):>6}  ty le vung bi nen MOT PHAN (20-80%)='
          f'{part * 100:>5.1f}%   nen han(>80%)={100 * (fr > 0.8).mean():>5.1f}%   '
          f'giu han(<20%)={100 * (fr < 0.2).mean():>5.1f}%')

print('\n' + '=' * 78)
print('C. COUNT: so vung SONG SOT co lien quan den dap an (so luong) khong?')
print('=' * 78)
mask_c = types == 1
surv, gold = [], []
for i in np.where(mask_c)[0]:
    r = rm.get(int(ids[i]))
    if r is None or r.shape[0] != 196: continue
    try:
        g = int(str(df['answer'].iloc[i]).strip())
    except Exception:
        continue
    n_s = sum(1 for rid in np.unique(r) if rid > 0 and (r == rid).sum() >= 2
              and P[i][r == rid].mean() > 0.5)
    surv.append(n_s); gold.append(g)
surv, gold = np.array(surv), np.array(gold)
if len(surv) > 20:
    n_reg = []
    for i in np.where(mask_c)[0][:len(surv)]:
        r = rm.get(int(ids[i]))
        if r is not None and r.shape[0] == 196:
            n_reg.append(sum(1 for rid in np.unique(r) if rid > 0 and (r == rid).sum() >= 2))
    n_reg = np.array(n_reg[:len(surv)])
    print(f'  n={len(surv)} mau COUNT co annotation')
    print(f'  tuong quan Pearson(so vung song sot, dap an) = {np.corrcoef(surv, gold)[0, 1]:+.3f}')
    print(f'  tuong quan Pearson(TONG so vung,      dap an) = {np.corrcoef(n_reg, gold)[0, 1]:+.3f}')
    print(f'  -> neu "song sot" tuong quan CAO HON "tong so vung": oracle dang THUA vung ve dung')
    print(f'     so luong can dem => co che that la DE-DUPLICATION, va do la dieu co the giam sat.')
