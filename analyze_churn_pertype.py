"""CHURN THEO LOAI, 10 SEED, KEM DAI NHIEU — muc "still planned #3" cua TCVG_ABLATION_TRACKING.

Cau hoi: hieu ung cua gate co tap trung o COLOR/COUNT nhu phan tich oracle tien doan khong
(oracle cuu 8.56% COUNT va 5.92% COLOR, so voi 1.92% OBJECT).

Ghep theo TUNG SEED (T0 seed s vs T2 seed s, cung SigLIP1, cung seed) — khong bao gio ghep
cheo seed, va khong bao gio tron SigLIP1 voi SigLIP2.

DOI CHUNG BAT BUOC: cap CUNG CAU HINH khac seed (T0_i vs T0_j, T2_i vs T2_j). Hai run cung
config van churn hang tram mau; neu khong in dai do ra thi con so "gate sua duoc N mau" khong
doc duoc. Da mot lan phan ra churn tren MOT seed va ket luan sai vi thieu dai nay.
"""
import os, itertools, numpy as np, pandas as pd
from scipy import stats

SEEDS = list(range(10))
T = ['OBJECT', 'COUNT', 'COLOR', 'LOCATION']


def load(arm, s):
    f = f'analysis/T0/T0_seed{s}.csv' if arm == 'T0' else f'beam3fixed/seed{s}_ep40.csv'
    if not os.path.exists(f):
        return None
    d = pd.read_csv(f)
    d['hit'] = (d.exact_match > 0.5).astype(int)
    return d


A = {('T0', s): load('T0', s) for s in SEEDS}
A.update({('T2', s): load('T2', s) for s in SEEDS})
A = {k: v for k, v in A.items() if v is not None}
have = {arm: sorted(s for (a, s) in A if a == arm) for arm in ('T0', 'T2')}
print(f'T0 seed {have["T0"]}\nT2 seed {have["T2"]}')

ref = A[('T0', have['T0'][0])]
for k, d in A.items():
    assert len(d) == len(ref), f'{k} khac so dong'
    assert (d.question.values == ref.question.values).all(), f'{k} khac thu tu cau hoi'
TY = ref.question_type.values
n = len(ref)
print(f'{n} mau test, khop thu tu 100% | phan bo loai: '
      f'{ {t: int((TY==t).sum()) for t in T} }\n')


def churn(a, b, mask):
    """a = arm goc, b = arm moi. Tra ve (sua duoc, lam hong, net) tren tap mask."""
    ha, hb = a.hit.values[mask], b.hit.values[mask]
    fix = int(((ha == 0) & (hb == 1)).sum())
    brk = int(((ha == 1) & (hb == 0)).sum())
    return fix, brk, fix - brk


pairs = [s for s in SEEDS if ('T0', s) in A and ('T2', s) in A]
print(f'=== {len(pairs)} cap ghep duoc (T2 - T0, cung seed) ===\n')

rows = []
for lbl, mask in [('TAT CA', np.ones(n, bool))] + [(t, TY == t) for t in T]:
    m = mask
    cross = np.array([churn(A[('T0', s)], A[('T2', s)], m) for s in pairs])
    fix, brk, net = cross[:, 0], cross[:, 1], cross[:, 2]
    # doi chung cung cau hinh: moi cap seed i<j trong CUNG mot arm
    null = []
    for arm in ('T0', 'T2'):
        ss = have[arm]
        for i, j in itertools.combinations(ss, 2):
            null.append(churn(A[(arm, i)], A[(arm, j)], m)[2])
    null = np.array(null, float)
    em_d = 100 * net / m.sum()
    t_, p_ = stats.ttest_1samp(net, 0)
    rows.append(dict(loai=lbl, n=int(m.sum()), sua=fix.mean(), hong=brk.mean(), net=net.mean(),
                     net_sd=net.std(ddof=1), dEM=em_d.mean(), dEM_sd=em_d.std(ddof=1),
                     p=p_, duong=f'{int((net>0).sum())}/{len(net)}',
                     nhieu_sd=null.std(ddof=1), nhieu_p95=np.percentile(np.abs(null), 95),
                     viet_lai=(fix + brk).mean()))

d = pd.DataFrame(rows)
print(f'{"loai":<9}{"n":>6}{"viet lai":>10}{"sua":>8}{"hong":>8}{"net":>9}{"dEM":>8}'
      f'{"p":>8}{"duong":>8}{"nhieu SD":>10}{"|nhieu| p95":>12}')
for _, r in d.iterrows():
    print(f'{r.loai:<9}{r.n:>6}{r.viet_lai:>10.1f}{r.sua:>8.1f}{r.hong:>8.1f}'
          f'{r.net:>+9.1f}{r.dEM:>+8.2f}{r.p:>8.3f}{r.duong:>8}{r.nhieu_sd:>10.1f}{r.nhieu_p95:>12.1f}')

print('\nDOC BANG:')
print('  "viet lai" = so cau gate doi dap an (sua + hong). "net" = sua - hong.')
print('  "nhieu SD"/"|nhieu| p95" = net cua cac cap CUNG cau hinh khac seed. net cua gate phai')
print('  vuot han dai nay moi doc duoc; nam trong dai thi khong phan biet duoc voi doi seed.')

print('\n=== KIEM DINH CHINH: EM theo loai, ghep cap 10 seed ===')
print(f'{"loai":<10}{"T0":>8}{"T2":>8}{"delta":>9}{"sd":>7}{"p":>8}{"duong":>8}')
for lbl, mask in [('TAT CA', np.ones(n, bool))] + [(t, TY == t) for t in T]:
    e0 = np.array([100 * A[('T0', s)].hit.values[mask].mean() for s in pairs])
    e2 = np.array([100 * A[('T2', s)].hit.values[mask].mean() for s in pairs])
    dd = e2 - e0
    t_, p_ = stats.ttest_rel(e2, e0)
    print(f'{lbl:<10}{e0.mean():>8.2f}{e2.mean():>8.2f}{dd.mean():>+9.2f}{dd.std(ddof=1):>7.2f}'
          f'{p_:>8.3f}{int((dd>0).sum())}/{len(dd):<8}')

print('\n=== TY LE TRIET TIEU: bao nhieu phan cua viec viet lai bi huy ===')
for _, r in d.iterrows():
    tot = r.sua + r.hong
    print(f'  {r.loai:<10} viet lai {tot:5.1f} -> net {r.net:+5.1f}  '
          f'({100*(1-abs(r.net)/tot) if tot else 0:.1f}% triet tieu)')
