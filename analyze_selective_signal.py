"""BUOC 2 — co tin hieu TRUOC-CAN-THIEP nao phan biet RESCUE voi DAMAGE khong?

Su kien = mot cap (mau, seed) ma T0 va T2 khac ket qua.  y = 1 RESCUE, y = 0 DAMAGE.
Dac trung: margin/confidence do duoi MOT model THAM CHIEU CO DINH (run87), dung giao thuc cua
dump_test_margin.py. Loai seed 42 khoi phan tich vi run87 CHINH LA T2 seed42 (se ro ri).

Neu margin chi cho biet mau DE LAT (ca hai chieu) chu khong cho biet CHIEU, thi AUROC ~ 0.5
va khong co luat can thiep chon loc nao ton tai.
"""
import numpy as np, pandas as pd, os
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

SEEDS = [0,1,2,3,4,5,6,7,8]          # 42 bi loai: model tham chieu la T2 seed42
ref = pd.read_csv('beam3fixed/seed0_ep40.csv')
n = len(ref); typ = ref.question_type.values
M = pd.read_csv('analysis/evread/s1_run87_meta.csv').set_index('idx')
print(f'model tham chieu run87: {len(M)} / {n} dong co margin (gold rank<=2)')

FEAT = ['margin','s1','s2','margin_ln','ent','s_std','s1_z','nvoc',
        'logf1','logf2','logf_d','qc1','qc2','qc_d']
ev = []
for s in SEEDS:
    e0 = pd.read_csv(f'analysis/T0/T0_seed{s}.csv').exact_match.values.astype(int)
    e2 = pd.read_csv(f'beam3fixed/seed{s}_ep40.csv').exact_match.values.astype(int)
    ch = np.where(e0 != e2)[0]
    for i in ch:
        if i not in M.index: continue
        r = M.loc[i]
        ev.append(dict(idx=i, seed=s, y=int(e2[i]==1), type=typ[i],
                       **{k: float(r[k]) for k in FEAT}))
E = pd.DataFrame(ev)
print(f'{len(E)} su kien churn (co margin), RESCUE {int(E.y.sum())}  DAMAGE {int((1-E.y).sum())}  '
      f'ty le co so {E.y.mean():.3f}')

print('\n=== A. don bien: AUROC cua tung dac trung cho RESCUE-vs-DAMAGE ===')
print(f'{"dac trung":<12} {"AUROC":>7} {"|lech 0.5|":>11} {"Mann-Whitney p":>15}')
rows=[]
for f in FEAT:
    a = roc_auc_score(E.y, E[f]); u,p = stats.mannwhitneyu(E[f][E.y==1], E[f][E.y==0])
    rows.append((f,a,abs(a-.5),p))
for f,a,d,p in sorted(rows,key=lambda r:-r[2]):
    print(f'{f:<12} {a:>7.4f} {d:>11.4f} {p:>15.4f}')

print('\n=== B. da bien, CV 5-fold NHOM THEO MAU (khong ro ri mau qua fold) ===')
X_all = E[FEAT].values; T1H = np.eye(4)[pd.Categorical(E.type,
        categories=['OBJECT','COUNT','COLOR','LOCATION']).codes]
y = E.y.values; g = E.idx.values
def cv(X, seed=0):
    o = np.zeros(len(y))
    for tr,va in GroupKFold(5).split(X,y,g):
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000))
        m.fit(X[tr],y[tr]); o[va] = m.predict_proba(X[va])[:,1]
    return o
ARMS = {'chi type': T1H, 'chi margin': E[['margin']].values,
        'decoder day du': X_all, 'decoder + type': np.hstack([X_all,T1H])}
O={}
for k,X in ARMS.items():
    O[k]=cv(X); print(f'  {k:<18} AUROC {roc_auc_score(y,O[k]):.4f}')
print('\n  bootstrap 95% CI (2000 lan) cho "decoder + type":')
o=O['decoder + type']; d=[]; r2=np.random.RandomState(0)
for _ in range(2000):
    b=r2.randint(0,len(y),len(y))
    if y[b].min()==y[b].max(): continue
    d.append(roc_auc_score(y[b],o[b]))
lo,hi=np.percentile(d,[2.5,97.5])
print(f'    AUROC {np.mean(d):.4f}  [{lo:.4f}, {hi:.4f}]  '
      f'{"VUOT 0.5" if lo>0.5 else "CHUA 0.5 -> khong co tin hieu"}')

print('\n=== C. margin co bao "DE LAT" (ca hai chieu) khong? ===')
allm = M.margin.values
print(f'margin trung vi: toan bo test {np.median(allm):.3f} | '
      f'su kien churn {E.margin.median():.3f} | RESCUE {E.margin[E.y==1].median():.3f} | '
      f'DAMAGE {E.margin[E.y==0].median():.3f}')
u,p = stats.mannwhitneyu(E.margin, np.random.RandomState(0).choice(allm,len(E)))
print(f'churn vs toan bo (margin thap hon?): Mann-Whitney p={p:.3e}  '
      f'AUROC(-margin -> la churn) = {roc_auc_score(np.r_[np.ones(len(E)),np.zeros(len(allm))], np.r_[-E.margin.values,-allm]):.4f}')
print('  -> margin CAO cho biet mau de lat; cau hoi la no co cho biet CHIEU khong (muc A/B)')

print('\n=== D. luat nguong: net dat duoc neu chi ap TCVG khi diem > nguong ===')
print(f'{"phan vi giu":>12} {"n giu":>7} {"RESCUE":>7} {"DAMAGE":>7} {"net":>6} {"net/seed":>9}')
for q in [100,75,50,30,20,10]:
    thr = np.percentile(o,100-q)
    k = o>=thr
    r=int(E.y[k].sum()); dm=int((1-E.y[k]).sum())
    print(f'{q:>11}% {k.sum():>7} {r:>7} {dm:>7} {r-dm:>+6} {(r-dm)/len(SEEDS):>+9.1f}')
print(f'\n(net/seed cua "giu tat ca" phai khop voi net T0->T2 quan sat duoc, ~+8.5 tren toan test;'
      f'\n o day chi tren tap con co gold rank<=2 nen nho hon)')
