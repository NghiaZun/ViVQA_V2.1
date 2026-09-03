"""BUOC 2b — neu tung su kien churn la ngau nhien, thi tap mau BI CUU LAP LAI qua nhieu seed
co phai la mot tap co cau truc khong? Va co phan biet duoc voi tap BI HONG lap lai khong?

Do la hy vong cuoi cung cho mot luat can thiep chon loc: khong can du doan tung su kien,
chi can nhan dien duoc "loai mau ma TCVG luon giup" vs "loai mau ma TCVG luon hai".
"""
import numpy as np, pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

SEEDS=[0,1,2,3,4,5,6,7,8,42]
ref=pd.read_csv('beam3fixed/seed0_ep40.csv'); n=len(ref); typ=ref.question_type.values
E0=np.stack([pd.read_csv(f'analysis/T0/T0_seed{s}.csv').exact_match.values.astype(int) for s in SEEDS])
E2=np.stack([pd.read_csv(f'beam3fixed/seed{s}_ep40.csv').exact_match.values.astype(int) for s in SEEDS])
Rm=((E0==0)&(E2==1)).astype(int); Dm=((E0==1)&(E2==0)).astype(int)
R=Rm.sum(0); D=Dm.sum(0); S=len(SEEDS)

print('=== A. tap RESCUE/DAMAGE lap lai co vuot muc ngau nhien khong? ===')
rng=np.random.RandomState(0)
def null_counts(M,reps=2000):
    """hoan vi nhan trong TUNG seed -> giu nguyen so su kien moi seed, xoa cau truc theo mau"""
    out=np.zeros((reps,S+1))
    for r in range(reps):
        P=np.stack([rng.permutation(M[i]) for i in range(S)]).sum(0)
        out[r]=np.bincount(P,minlength=S+1)[:S+1]
    return out
for name,M,obs in (('RESCUE',Rm,R),('DAMAGE',Dm,D)):
    nc=null_counts(M); o=np.bincount(obs,minlength=S+1)[:S+1]
    print(f'\n  {name}: so mau xuat hien o >=k seed')
    print(f'  {"k":>3} {"quan sat":>9} {"null tb":>9} {"null 97.5%":>11}')
    for k in (1,2,3,5,7,10):
        ob=(obs>=k).sum(); nu=(nc[:,k:].sum(1)); 
        print(f'  {k:>3} {ob:>9} {nu.mean():>9.1f} {np.percentile(nu,97.5):>11.1f}')

print('\n=== B. tap "TCVG luon giup" vs "TCVG luon hai" co phan biet duoc bang dac trung? ===')
M=pd.read_csv('analysis/evread/s1_run87_meta.csv').set_index('idx')
FEAT=['margin','s1','s2','margin_ln','ent','s_std','s1_z','nvoc','logf1','logf2','logf_d','qc1','qc2','qc_d']
for kthr in (3,4,5):
    sel=np.where(((R>=kthr)&(D==0))|((D>=kthr)&(R==0)))[0]
    sel=[i for i in sel if i in M.index]
    if len(sel)<20: print(f'  k>={kthr}: chi {len(sel)} mau, bo qua'); continue
    y=np.array([1 if R[i]>=kthr else 0 for i in sel])
    X=M.loc[sel,FEAT].values
    T=np.eye(4)[pd.Categorical(typ[sel],categories=['OBJECT','COUNT','COLOR','LOCATION']).codes]
    XT=np.hstack([X,T])
    o=np.zeros(len(y))
    for tr,va in StratifiedKFold(5,shuffle=True,random_state=0).split(XT,y):
        mdl=make_pipeline(StandardScaler(),LogisticRegression(max_iter=4000))
        mdl.fit(XT[tr],y[tr]); o[va]=mdl.predict_proba(XT[va])[:,1]
    auc=roc_auc_score(y,o)
    d=[];r2=np.random.RandomState(1)
    for _ in range(2000):
        b=r2.randint(0,len(y),len(y))
        if y[b].min()==y[b].max(): continue
        d.append(roc_auc_score(y[b],o[b]))
    lo,hi=np.percentile(d,[2.5,97.5])
    print(f'  k>={kthr}: n={len(y)} (giup {y.sum()} / hai {(1-y).sum()})  '
          f'CV AUROC {auc:.4f}  [{lo:.4f}, {hi:.4f}]  '
          f'{"CO tin hieu" if lo>0.5 else "chua 0.5"}')

print('\n=== C. dac trung co tuong quan voi "net rescue" lien tuc (R-D) khong? ===')
ch=np.where((R+D)>0)[0]; ch=[i for i in ch if i in M.index]
net=(R-D)[ch]
print(f'  {len(ch)} mau co it nhat 1 su kien churn; net R-D: '
      f'trung binh {net.mean():+.3f}, sd {net.std():.3f}')
print(f'  {"dac trung":<12} {"Spearman":>10} {"p":>10}')
rows=[]
for f in FEAT:
    rho,p=stats.spearmanr(M.loc[ch,f].values,net); rows.append((f,rho,p))
for f,rho,p in sorted(rows,key=lambda r:-abs(r[1]))[:6]:
    print(f'  {f:<12} {rho:>+10.4f} {p:>10.4f}')
print(f'  (Bonferroni cho {len(FEAT)} dac trung: nguong p = {0.05/len(FEAT):.4f})')
