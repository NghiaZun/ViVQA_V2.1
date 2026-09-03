"""MA TRAN 2x2 T0 vs T2 — dung/sai chuyen dich the nao khi them gate + type_loss.

    T0 dung & T2 dung   : ca hai lam duoc
    T0 dung & T2 SAI    : T2 LAM HONG
    T0 SAI  & T2 dung   : T2 SUA DUOC
    T0 SAI  & T2 SAI    : ca hai chiu

Ghep theo TUNG SEED (T0 seed s vs T2 seed s, cung SigLIP1) roi lay trung binh 10 seed.

BAT BUOC CO DOI CHUNG: trong du an nay hai lan chay CUNG CAU HINH da tung churn 222-232 mau
voi net tu -28 den +24. Neu khong in muc nhieu do ra thi con so "T2 sua duoc N mau" khong doc
duoc — da mot lan phan ra churn tren MOT seed va ket luan sai vi dieu nay.
Doi chung o day: cap T0 seed i vs T0 seed j (i != j) — cung cau hinh, chi khac seed.
"""
import pandas as pd, numpy as np, os
from itertools import combinations
from scipy import stats

SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 42]
TYPES = ['OBJECT', 'COUNT', 'COLOR', 'LOCATION']

def load(arm, s):
    f = f'analysis/T0/T0_seed{s}.csv' if arm == 'T0' else f'beam3fixed/seed{s}_ep40.csv'
    return pd.read_csv(f) if os.path.exists(f) else None

pairs = []
for s in SEEDS:
    a, b = load('T0', s), load('T2', s)
    if a is None or b is None or len(a) != len(b):
        continue
    pairs.append((s, a.exact_match.values > .5, b.exact_match.values > .5, a.question_type.values))
print(f'{len(pairs)} cap seed ghep duoc (T0 vs T2, cung SigLIP1, cung seed)')

def mat(e0, e2):
    return (int((e0 & e2).sum()), int((e0 & ~e2).sum()),
            int((~e0 & e2).sum()), int((~e0 & ~e2).sum()))

M = np.array([mat(e0, e2) for _, e0, e2, _ in pairs])          # [n,4]
n = M[0].sum()
print(f'\n=== MA TRAN 2x2, trung binh {len(pairs)} seed (tren {n} mau test) ===')
print(f'{"":22s} {"T2 DUNG":>12s} {"T2 SAI":>12s}')
print(f'{"T0 DUNG":22s} {M[:,0].mean():12.1f} {M[:,1].mean():12.1f}')
print(f'{"T0 SAI":22s} {M[:,2].mean():12.1f} {M[:,3].mean():12.1f}')
churn = M[:, 1] + M[:, 2]
net = M[:, 2] - M[:, 1]
print(f'\n  T2 SUA DUOC   : {M[:,2].mean():6.1f} +/- {M[:,2].std(ddof=1):.1f}')
print(f'  T2 LAM HONG   : {M[:,1].mean():6.1f} +/- {M[:,1].std(ddof=1):.1f}')
print(f'  NET           : {net.mean():+6.1f} +/- {net.std(ddof=1):.1f}   (= {net.mean()/n*100:+.2f} pp EM)')
print(f'  TONG CHUYEN DICH: {churn.mean():6.1f}  ({churn.mean()/n*100:.1f}% test)')
t = stats.ttest_1samp(net, 0)
print(f'  kiem net != 0 : p={t.pvalue:.3f} | net > 0 o {int((net>0).sum())}/{len(net)} seed')

# ---------- DOI CHUNG: cung cau hinh, khac seed ----------
print(f'\n=== DOI CHUNG (cung cau hinh T0, chi khac seed) ===')
ctrl_ch, ctrl_net = [], []
for i, j in combinations(range(len(pairs)), 2):
    e_i, e_j = pairs[i][1], pairs[j][1]
    _, b, c, _ = mat(e_i, e_j)
    ctrl_ch.append(b + c); ctrl_net.append(c - b)
print(f'  chuyen dich   : {np.mean(ctrl_ch):6.1f} +/- {np.std(ctrl_ch,ddof=1):.1f}   ({len(ctrl_ch)} cap)')
print(f'  net           : {np.mean(ctrl_net):+6.1f} +/- {np.std(ctrl_net,ddof=1):.1f}'
      f'   khoang [{min(ctrl_net):+d}, {max(ctrl_net):+d}]')
print(f'  -> BAT KY net nao nam trong khoang nay deu KHONG phan biet duoc voi nhieu seed.')

# ---------- tach theo loai ----------
print(f'\n=== TACH THEO LOAI CAU HOI ===')
print(f'{"loai":10s} {"n":>6s} {"sua duoc":>10s} {"lam hong":>10s} {"net":>8s} {"pp":>8s} {"p":>7s} {"thang":>6s}')
for t_ in TYPES:
    fix, brk = [], []
    for _, e0, e2, qt in pairs:
        m = qt == t_
        fix.append(int((~e0[m] & e2[m]).sum())); brk.append(int((e0[m] & ~e2[m]).sum()))
    fix, brk = np.array(fix), np.array(brk)
    nt = int((pairs[0][3] == t_).sum()); nn = fix - brk
    p = stats.ttest_1samp(nn, 0).pvalue
    print(f'{t_:10s} {nt:6d} {fix.mean():10.1f} {brk.mean():10.1f} {nn.mean():+8.1f} '
          f'{nn.mean()/nt*100:+8.2f} {p:7.3f} {int((nn>0).sum()):4d}/{len(nn)}')

print('\nDoc: "sua duoc" va "lam hong" deu lon la dau hieu PHEP CHIA LAI, khong phai phep cong.')
print('Chi ket luan khi NET vuot ra ngoai khoang doi chung o tren.')
