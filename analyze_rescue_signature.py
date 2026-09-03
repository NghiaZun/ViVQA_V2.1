"""What distinguishes (base WRONG -> oracle CORRECT) from (base WRONG -> oracle STILL WRONG)?

Uses ONLY pre-intervention observables. Six feature families, then a family-wise cross-validated
comparison. The decisive question is not "can we predict rescue" (we already know decoder-side
uncertainty does that) but "does any VISION-side signal add anything the decoder state does not
already contain". If not, TCVG's success is a property of the decoder's tie structure, not of the
visual evidence -- i.e. the operator perturbs a near-tie rather than exposing image information.
"""
import numpy as np, pandas as pd, torch, unicodedata as ud
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from scipy import stats

Z=np.load('analysis/mech/rescue_state_s2T2.npz')
M=pd.read_csv('analysis/mech/rescue_state_s2T2_meta.csv')
POST=Z['post'].astype(np.float32); VPR=Z['vproj'].astype(np.float32); Q=Z['q'].astype(np.float32); ALP=Z['alpha'].astype(np.float32)
N,P,D=POST.shape
y=M.rescued.values.astype(int)
print(f"n={N} errors, rescued={y.sum()} ({100*y.mean():.1f}%), P={P}, D={D}")

# candidate embeddings from the checkpoint's shared embedding matrix (CPU, no GPU)
ck=torch.load('checkpoints_s2_T2/best_model.pt',map_location='cpu',weights_only=False)
sd=ck['model_state_dict']
ek=[k for k in sd if k.endswith('shared.weight') or k.endswith('embed_tokens.weight')]
EMB=sd[ek[0]].float().numpy(); print(f"embedding matrix {ek[0]} {EMB.shape}")
from transformers import AutoTokenizer
tk=AutoTokenizer.from_pretrained('vinai/bartpho-syllable')
norm=lambda s: ud.normalize('NFC',str(s)).strip().lower()
tr=pd.read_csv('archive/train_split_original.csv'); tr['an']=tr.answer.map(norm)
TVOC={t:sorted(set(g.an)) for t,g in tr.groupby('type')}
def emb_of(txt):
    ids=[i for i in tk.encode(txt) if i not in (tk.pad_token_id,tk.bos_token_id,tk.eos_token_id)]
    return EMB[ids].mean(0) if ids else np.zeros(EMB.shape[1],np.float32)

def nrm(a,ax=-1): return a/np.linalg.norm(a,axis=ax,keepdims=True).clip(1e-9)
Pn=nrm(POST); Vn=nrm(VPR); Qn=nrm(Q)
F={}
# ---------- 1. DECODER / candidate-logit family ----------
mar=[];top1=[];ent=[];gap23=[];nclose=[];spread=[];pmax=[]
c1e=[];c2e=[]
for i,r in M.iterrows():
    s=Z[f"{int(r.type)}_{int(r.idx)}"].astype(np.float64)
    o=np.argsort(-s); voc=TVOC[int(r.type)]
    mar.append(s[o[0]]-s[o[1]]); top1.append(s[o[0]])
    gap23.append(s[o[1]]-s[o[2]] if len(s)>2 else 0.0)
    pr=np.exp(s-s.max()); pr/=pr.sum()
    ent.append(float(-(pr*np.log(pr+1e-12)).sum())); pmax.append(pr.max())
    nclose.append(int((s>s[o[0]]-1.0).sum())); spread.append(float(s.std()))
    c1e.append(emb_of(voc[o[0]])); c2e.append(emb_of(voc[o[1]]))
F['dec_margin']=np.array(mar); F['dec_top1']=np.array(top1); F['dec_entropy']=np.array(ent)
F['dec_gap23']=np.array(gap23); F['dec_nclose1nat']=np.array(nclose); F['dec_spread']=np.array(spread); F['dec_pmax']=np.array(pmax)
# ---------- 2. PATCH-REPRESENTATION family ----------
F['vis_norm_mean']=np.linalg.norm(VPR,axis=-1).mean(1); F['vis_norm_std']=np.linalg.norm(VPR,axis=-1).std(1)
C=np.einsum('npd,nqd->npq',Pn,Pn)
off=(C.sum((1,2))-np.trace(C,axis1=1,axis2=2))/(P*(P-1)); F['vis_pair_cos']=off
sv=np.linalg.svd(POST-POST.mean(1,keepdims=True),compute_uv=False)
F['vis_eff_rank']=(sv.sum(1)**2)/(sv**2).sum(1)
F['vis_alpha_mean']=ALP.mean(1); F['vis_alpha_std']=ALP.std(1)
# ---------- 3. QUESTION-PATCH similarity ----------
qa=np.einsum('npd,nd->np',Vn,Qn)
F['qp_cos_mean']=qa.mean(1); F['qp_cos_max']=qa.max(1); F['qp_cos_std']=qa.std(1)
# ---------- 4. CANDIDATE-PATCH similarity (top-1 vs top-2 -- both observable) ----------
c1=nrm(np.array(c1e)); c2=nrm(np.array(c2e))
s1=np.einsum('npd,nd->np',Pn,c1); s2=np.einsum('npd,nd->np',Pn,c2)
F['cp_top1_max']=s1.max(1); F['cp_top1_mean']=s1.mean(1)
F['cp_top2_max']=s2.max(1); F['cp_top2_mean']=s2.mean(1)
F['cp_discrim_max']=np.abs(s1-s2).max(1)          # is there ANY patch that separates the top-2?
F['cp_discrim_mean']=np.abs(s1-s2).mean(1)
F['cp_pref_frac']=(s1>s2).mean(1)                 # fraction of patches favouring the current top-1
# ---------- 5. SPATIAL distribution (14x14 grid; index 0 is the pooler token) ----------
g=int(round(np.sqrt(P-1))); idx=np.arange(1,P)
gy,gx=np.divmod(idx-0,g)[0][:g*g] if False else (np.arange(g*g)//g, np.arange(g*g)%g)
def moran(v):
    """spatial autocorrelation of a per-patch scalar on the grid (rook adjacency)"""
    out=[]
    for k in range(v.shape[0]):
        z=v[k,1:1+g*g].reshape(g,g); z=z-z.mean()
        num=(z[:,:-1]*z[:,1:]).sum()+(z[:-1,:]*z[1:,:]).sum()
        den=(z**2).sum(); W=2*g*(g-1)
        out.append(float((g*g/W)*num/max(den,1e-9)))
    return np.array(out)
F['sp_moran_alpha']=moran(ALP); F['sp_moran_qp']=moran(qa); F['sp_moran_disc']=moran(np.abs(s1-s2))
F['sp_disc_top10share']=np.sort(np.abs(s1-s2),axis=1)[:,-max(1,P//10):].sum(1)/np.abs(s1-s2).sum(1).clip(1e-9)
# ---------- 6. TYPE ----------
for t,nm in [(1,'COUNT'),(2,'COLOR'),(3,'LOCATION')]: F[f'type_{nm}']=(M.type.values==t).astype(float)

FAM={'DECODER (candidate logits/uncertainty)':[k for k in F if k.startswith('dec_')],
     'PATCH REPRESENTATION':[k for k in F if k.startswith('vis_')],
     'QUESTION-PATCH similarity':[k for k in F if k.startswith('qp_')],
     'CANDIDATE-PATCH similarity':[k for k in F if k.startswith('cp_')],
     'SPATIAL distribution':[k for k in F if k.startswith('sp_')],
     'QUESTION TYPE':[k for k in F if k.startswith('type_')]}
print("\n=== UNIVARIATE: AUROC for (oracle rescues) vs (oracle fails), n=%d ==="%N)
rows=[]
for fam,ks in FAM.items():
    for k in ks:
        v=F[k]; auc=roc_auc_score(y,v)
        rows.append((fam,k,auc,max(auc,1-auc),stats.mannwhitneyu(v[y==1],v[y==0]).pvalue))
R=pd.DataFrame(rows,columns=['family','feature','AUROC','|AUROC|','MWU_p']).sort_values('|AUROC|',ascending=False)
print(R.to_string(index=False,float_format=lambda x:f'{x:.3f}'))
print("\n=== FAMILY-WISE cross-validated AUROC (5-fold, logistic) ===")
def cv(ks):
    X=np.column_stack([F[k] for k in ks])
    p=cross_val_predict(make_pipeline(StandardScaler(),LogisticRegression(max_iter=3000,C=0.5)),
                        X,y,cv=StratifiedKFold(5,shuffle=True,random_state=0),method='predict_proba')[:,1]
    return roc_auc_score(y,p)
base=None
for fam,ks in FAM.items():
    if not ks: continue
    print(f"  {fam:<40} {cv(ks):.3f}   ({len(ks)} features)")
dec=FAM['DECODER (candidate logits/uncertainty)']
allf=[k for ks in FAM.values() for k in ks]
vis=[k for k in allf if not k.startswith('dec_')]
print(f"\n  {'DECODER only':<40} {cv(dec):.3f}")
print(f"  {'everything EXCEPT decoder':<40} {cv(vis):.3f}")
print(f"  {'ALL families':<40} {cv(allf):.3f}")
print(f"  {'DECODER + type':<40} {cv(dec+FAM['QUESTION TYPE']):.3f}")

print("\n" + "="*78)
print("INCREMENTAL TEST: does any VISION signal add anything beyond decoder state + type?")
print("="*78)
rng=np.random.default_rng(0)
def cvp(ks,seed=0):
    X=np.column_stack([F[k] for k in ks])
    return cross_val_predict(make_pipeline(StandardScaler(),LogisticRegression(max_iter=3000,C=0.5)),
                             X,y,cv=StratifiedKFold(5,shuffle=True,random_state=seed),method='predict_proba')[:,1]
def boot_diff(p1,p2,B=3000):
    d=[]
    for _ in range(B):
        i=rng.integers(0,len(y),len(y))
        if len(np.unique(y[i]))<2: continue
        d.append(roc_auc_score(y[i],p2[i])-roc_auc_score(y[i],p1[i]))
    d=np.array(d); return d.mean(),np.percentile(d,[2.5,97.5])
TY=FAM['QUESTION TYPE']; DEC=FAM['DECODER (candidate logits/uncertainty)']
CP=FAM['CANDIDATE-PATCH similarity']; VIS=FAM['PATCH REPRESENTATION']
QP=FAM['QUESTION-PATCH similarity']; SP=FAM['SPATIAL distribution']
ALLV=CP+VIS+QP+SP
base=DEC+TY
# average over 5 CV seeds to stabilise
def auc_multi(ks): return np.mean([roc_auc_score(y,cvp(ks,s)) for s in range(5)])
print(f"  {'decoder + type (reference)':<44} {auc_multi(base):.3f}")
for nm,add in [('+ candidate-patch similarity',CP),('+ patch representation',VIS),
               ('+ question-patch similarity',QP),('+ spatial distribution',SP),
               ('+ ALL vision families',ALLV)]:
    m,ci=boot_diff(cvp(base),cvp(base+add))
    print(f"  {nm:<44} {auc_multi(base+add):.3f}   delta={m:+.3f}  95%CI [{ci[0]:+.3f}, {ci[1]:+.3f}]")
print("\n=== is candidate-patch similarity independent of the decoder's tie structure? ===")
for k in ['cp_top2_mean','cp_top2_max','cp_discrim_max','vis_alpha_mean']:
    r_m=stats.spearmanr(F[k],F['dec_margin']).statistic
    print(f"  {k:<18} corr with dec_margin = {r_m:+.3f}")
print("\n  within-type AUROC (vocabulary and prior held constant):")
for k in ['cp_top2_mean','cp_top1_max','cp_discrim_max','dec_margin','vis_alpha_mean']:
    line=f"    {k:<18}"
    for t,nm in [(1,'COUNT'),(2,'COLOR'),(3,'LOCATION'),(0,'OBJECT')]:
        m_=M.type.values==t
        if len(np.unique(y[m_]))<2: line+=f"  {nm}=  n/a"; continue
        line+=f"  {nm}={roc_auc_score(y[m_],F[k][m_]):.2f}"
    print(line)
print("\n=== partial: rescue ~ cp_top2_mean CONTROLLING for margin (residualised) ===")
for k in ['cp_top2_mean','cp_top2_max','cp_discrim_max']:
    b=np.polyfit(F['dec_margin'],F[k],1); res=F[k]-np.polyval(b,F['dec_margin'])
    print(f"  {k:<18} residual AUROC = {roc_auc_score(y,res):.3f}  (raw {roc_auc_score(y,F[k]):.3f})")
