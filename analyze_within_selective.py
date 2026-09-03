"""BUOC 3 phan tich — TCVG tren CUNG BO TRONG SO: chat luong can thiep vs khoi luong can thiep."""
import numpy as np, pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import sys
df = pd.read_csv(sys.argv[1] if len(sys.argv)>1 else 'analysis/evread/within_s1.csv')
TN = {0:'OBJECT',1:'COUNT',2:'COLOR',3:'LOCATION'}
df['tname'] = df.type.map(TN)
n = len(df)
R = (df.cell=='RESCUE').sum(); D = (df.cell=='DAMAGE').sum()
print(f'n={n}  EM tat {df.hit_off.mean()*100:.2f} -> bat {df.hit_on.mean()*100:.2f} '
      f'({(df.hit_on.mean()-df.hit_off.mean())*100:+.2f}pp)')
print(f'\n=== A. CHAT LUONG can thiep (cung trong so, khong nhieu seed) ===')
print(f'  RESCUE {R}   DAMAGE {D}   ti le {R/max(D,1):.2f}:1   net {R-D:+d} = {(R-D)/n*100:+.2f}pp')
pm = stats.binomtest(int(R), int(R+D), 0.5).pvalue
print(f'  McNemar (nhi thuc chinh xac) p = {pm:.4f}')
lo,hi = stats.binomtest(int(R), int(R+D), 0.5).proportion_ci(0.95)
print(f'  ti le RESCUE trong so su kien: {R/(R+D):.3f}  CI95 [{lo:.3f}, {hi:.3f}]')
print(f'  -> so voi lien-model (122 vs 114 = 1.07:1): can thiep RIENG cua TCVG ON DINH hon nhieu')

print(f'\n=== B. KHOI LUONG can thiep — TCVG cham vao bao nhieu mau? ===')
act = R+D
print(f'  doi quyet dinh o {act}/{n} = {act/n*100:.2f}% mau')
print(f'  khong doi quyet dinh o {100-act/n*100:.2f}%  (GCA + decoder da chot xong)')
print(f'  |dmargin| phan vi: ' + '  '.join(
    f'p{q}={np.percentile(df.dmargin.abs(),q):.3f}' for q in (50,75,90,99)))
print(f'  alpha_mean trung binh {df.alpha_mean.mean():.3f}')
print(f'\n  theo loai:')
print(f'  {"loai":<10} {"n":>5} {"RESCUE":>7} {"DAMAGE":>7} {"%cham":>7} {"alpha":>7} {"|dmargin|p50":>13}')
for t,g in df.groupby('tname'):
    r=(g.cell=='RESCUE').sum(); d=(g.cell=='DAMAGE').sum()
    print(f'  {t:<10} {len(g):>5} {r:>7} {d:>7} {(r+d)/len(g)*100:>6.2f}% '
          f'{g.alpha_mean.mean():>7.3f} {np.percentile(g.dmargin.abs(),50):>13.3f}')

print(f'\n=== C. co phan biet duoc RESCUE voi DAMAGE truoc khi can thiep khong? ===')
ev = df[df.cell.isin(['RESCUE','DAMAGE'])].copy(); y=(ev.cell=='RESCUE').astype(int).values
F = ['margin_off','ent_off','stop1_off','sgold_off','nvoc','alpha_mean','alpha_std']
F = [f for f in F if f in ev.columns and f!='sgold_off']    # sgold dung gold -> loai
print(f'  n={len(y)} (RESCUE {y.sum()}, DAMAGE {(1-y).sum()})')
for f in F:
    a=roc_auc_score(y,ev[f]); u,p=stats.mannwhitneyu(ev[f][y==1],ev[f][y==0])
    print(f'    {f:<12} AUROC {a:.4f}  p={p:.4f}')
T=np.eye(4)[ev.type.values.astype(int)]; X=np.hstack([ev[F].values,T])
o=np.zeros(len(y))
for tr,va in StratifiedKFold(5,shuffle=True,random_state=0).split(X,y):
    m=make_pipeline(StandardScaler(),LogisticRegression(max_iter=4000))
    m.fit(X[tr],y[tr]); o[va]=m.predict_proba(X[va])[:,1]
d=[];r2=np.random.RandomState(1)
for _ in range(2000):
    b=r2.randint(0,len(y),len(y))
    if y[b].min()==y[b].max(): continue
    d.append(roc_auc_score(y[b],o[b]))
lo2,hi2=np.percentile(d,[2.5,97.5])
print(f'  CV AUROC (decoder truoc-can-thiep + type) {roc_auc_score(y,o):.4f}  [{lo2:.4f}, {hi2:.4f}]')

print(f'\n=== D. LOI THOAT? con bao nhieu cho TCVG CHUA cham toi ===')
hard = df[df.cell=='hard']
print(f'  {len(hard)} mau sai o CA HAI nhanh. gold rank (gate tat): ' +
      '  '.join(f'r{k}={int((hard.rank_off==k).sum())}' for k in (2,3,4,5)) +
      f'  r>5={int((hard.rank_off>5).sum())}')
near = hard[hard.rank_off<=3]
print(f'  trong do gold rank<=3: {len(near)} mau = {len(near)/n*100:.1f}% test '
      f'-> day la du dia TCVG chua voi toi')
print(f'  deficit (stop1-sgold) tren nhom nay: trung vi {np.median(near.stop1_off-near.sgold_off):.3f}')
print(f'  |dmargin| ma TCVG thuc su tao ra: trung vi {np.median(df.dmargin.abs()):.3f}')
print(f'  -> TCVG day duoc ~{np.percentile(df.dmargin.abs(),90):.2f} nat o phan vi 90;'
      f' can {np.median(near.stop1_off-near.sgold_off):.2f} nat de voi toi nhom tren')
