"""BUOC 1 — phan hoach 4 o (RESCUE/DAMAGE) tu CSV da co, KHONG GPU.

Cau hoi: trong 4 o {base dung/sai} x {TCVG dung/sai}, RESCUE co thuc su nhieu hon DAMAGE khong,
va phan chenh lech do co PHAN BIET DUOC voi viec chi doi seed hay khong?

Doi chung bat buoc: T2_seedA vs T2_seedB — CUNG cau hinh, chi khac seed. Neu churn T0->T2 nam
trong dai churn cua doi chung nay thi phan hoach RESCUE/DAMAGE khong quy duoc cho TCVG.
"""
import numpy as np, pandas as pd, itertools, glob, os
from scipy import stats

SEEDS = [0,1,2,3,4,5,6,7,8,42]
T0 = {s: pd.read_csv(f'analysis/T0/T0_seed{s}.csv') for s in SEEDS
      if os.path.exists(f'analysis/T0/T0_seed{s}.csv')}
T2 = {s: pd.read_csv(f'beam3fixed/seed{s}_ep40.csv') for s in SEEDS
      if os.path.exists(f'beam3fixed/seed{s}_ep40.csv')}
SEEDS = sorted(set(T0) & set(T2))
print(f'seeds khop: {SEEDS}')

ref = T2[SEEDS[0]]
n = len(ref)
for s in SEEDS:
    assert len(T0[s]) == n and len(T2[s]) == n, (s, len(T0[s]), len(T2[s]))
    assert (T0[s].question.values == ref.question.values).all(), f'lech thu tu T0 seed{s}'
    assert (T2[s].question.values == ref.question.values).all(), f'lech thu tu T2 seed{s}'
print(f'{n} dong, thu tu khop tren moi file')

typ = ref.question_type.values
E0 = {s: T0[s].exact_match.values.astype(int) for s in SEEDS}
E2 = {s: T2[s].exact_match.values.astype(int) for s in SEEDS}

def cells(b, t):
    return dict(keep=int(((b==1)&(t==1)).sum()), rescue=int(((b==0)&(t==1)).sum()),
                damage=int(((b==1)&(t==0)).sum()), hard=int(((b==0)&(t==0)).sum()))

print('\n=== A. T0 -> T2, tung seed (cung seed hai ben) ===')
print(f'{"seed":>5} {"keep":>6} {"RESCUE":>7} {"DAMAGE":>7} {"hard":>6} {"net":>6} {"churn%":>7}')
rows=[]
for s in SEEDS:
    c = cells(E0[s], E2[s]); net = c['rescue']-c['damage']
    ch = 100*(c['rescue']+c['damage'])/n
    rows.append(dict(seed=s, **c, net=net, churn=ch))
    print(f'{s:>5} {c["keep"]:>6} {c["rescue"]:>7} {c["damage"]:>7} {c["hard"]:>6} {net:>+6} {ch:>7.2f}')
A = pd.DataFrame(rows)
print(f'{"mean":>5} {A.keep.mean():>6.1f} {A.rescue.mean():>7.1f} {A.damage.mean():>7.1f} '
      f'{A.hard.mean():>6.1f} {A.net.mean():>+6.1f} {A.churn.mean():>7.2f}')
t,p = stats.ttest_rel(A.rescue, A.damage)
print(f'\nRESCUE vs DAMAGE ghep cap theo seed: t={t:.3f} p={p:.4f}  '
      f'{(A.net>0).sum()}/{len(A)} seed duong')
print(f'net trung binh {A.net.mean():+.1f} mau = {100*A.net.mean()/n:+.3f}pp')

print('\n=== B. DOI CHUNG: T2_seedA vs T2_seedB (CUNG cau hinh, chi khac seed) ===')
ctl=[]
for a,b in itertools.combinations(SEEDS,2):
    c = cells(E2[a], E2[b]); ctl.append(dict(a=a,b=b,**c,net=c['rescue']-c['damage'],
                                             churn=100*(c['rescue']+c['damage'])/n))
C = pd.DataFrame(ctl)
print(f'{len(C)} cap. churn {C.churn.mean():.2f}% +/- {C.churn.std():.2f}  '
      f'[{C.churn.min():.2f}, {C.churn.max():.2f}]')
print(f'|net| cua doi chung: mean {C.net.abs().mean():.1f}  sd(net) {C.net.std():.1f}  '
      f'khoang [{C.net.min():+d}, {C.net.max():+d}]')
lo,hi = np.percentile(C.net,[2.5,97.5])
print(f'dai 95% cua net khi CHI doi seed: [{lo:+.1f}, {hi:+.1f}]')
print(f'net that cua T0->T2: {A.net.mean():+.1f}  -> '
      f'{"NGOAI dai (quy duoc cho TCVG)" if A.net.mean()>hi else "TRONG dai doi chung (khong quy duoc)"}')
print(f'churn T0->T2 {A.churn.mean():.2f}% vs churn doi chung {C.churn.mean():.2f}% -> '
      f'ti le {A.churn.mean()/C.churn.mean():.2f}x')

print('\n=== C. theo LOAI cau hoi (gop 10 seed) ===')
print(f'{"type":<10} {"n":>5} {"RESCUE":>7} {"DAMAGE":>7} {"net":>6} {"net/seed":>9} {"p":>8} {"win":>5}')
for t_ in ['OBJECT','COUNT','COLOR','LOCATION']:
    m = typ==t_
    r = np.array([int(((E0[s][m]==0)&(E2[s][m]==1)).sum()) for s in SEEDS])
    d = np.array([int(((E0[s][m]==1)&(E2[s][m]==0)).sum()) for s in SEEDS])
    tt,pp = stats.ttest_rel(r,d)
    print(f'{t_:<10} {m.sum():>5} {r.sum():>7} {d.sum():>7} {r.sum()-d.sum():>+6} '
          f'{(r-d).mean():>+9.1f} {pp:>8.4f} {(r>d).sum():>3}/{len(SEEDS)}')

print('\n=== D. RESCUE/DAMAGE co on dinh qua seed khong? ===')
R = np.stack([( (E0[s]==0)&(E2[s]==1) ).astype(int) for s in SEEDS])   # [S, n]
D = np.stack([( (E0[s]==1)&(E2[s]==0) ).astype(int) for s in SEEDS])
print(f'so mau la RESCUE o >=1 seed : {(R.sum(0)>0).sum()}')
print(f'   trong do o >=5 seed      : {(R.sum(0)>=5).sum()}')
print(f'   o CA 10 seed             : {(R.sum(0)==10).sum()}')
print(f'so mau la DAMAGE o >=1 seed : {(D.sum(0)>0).sum()}')
print(f'   trong do o >=5 seed      : {(D.sum(0)>=5).sum()}')
print(f'   o CA 10 seed             : {(D.sum(0)==10).sum()}')
exp1 = n*(1-(1-R.mean())**len(SEEDS))
print(f'\nneu RESCUE la doc lap ngau nhien moi seed (cung ti le), so mau >=1 seed ky vong ~{exp1:.0f}')
print(f'ty le RESCUE trung binh moi seed: {R.mean():.4f}, DAMAGE: {D.mean():.4f}')
