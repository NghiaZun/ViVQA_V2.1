"""Kiem y tuong tu literature token-pruning (E-AdaPrune 2026): do SUY GIAM PHO cua ma tran
dac trung patch lam tin hieu PER-SAMPLE khong can nhan.

Khoang trong da do trong du an nay:
  - alpha VO HUONG rieng tung mau co +3.50 headroom (oracle scalar 75.84 vs base 72.34)
  - scalar - scalarshuffle = +7.10 -> tin hieu THAT SU rieng tung mau
  - nhung 3 ho nhan deu that bai: mat na box (di NGUOC chieu), "thi giac co can thiet khong"
    (+0.0073 = 5% mot sigma), loi cua model (rong vi train da 99.12%)
  => thieu mot NHAN khong can giam sat cho alpha per-sample.

Y tuong tu literature: phi suy giam doc = nang luong don vao vai huong = anh du thua =
  can it thong tin thi giac hon. Do la mot uoc luong "can bao nhieu thi giac" TINH TU DAC TRUNG,
  khong can dap an, khong can annotation.

Kiem: cac chi so pho co tuong quan voi ORACLE alpha (fit bang dap an gold) khong?
  Neu CO -> tim duoc nhan cho alpha per-sample -> huong that su moi.
  Neu KHONG -> dong, va biet ngay khong ton mot lan train nao.
"""
import sys, torch, numpy as np, pandas as pd
from PIL import Image
from transformers import AutoModel, AutoProcessor
from scipy import stats

DEV='cuda'; VM='google/siglip2-base-patch16-224'; N=1200
te=pd.read_csv('archive/test.csv').head(N)
ora=pd.read_csv('analysis/oracle_alpha/perpatch.csv').head(N)   # oracle-fit alpha (run87/SigLIP1)
base=pd.read_csv('analysis/oracle_alpha/base.csv').head(N)      # model alpha cung checkpoint
assert len(ora)==len(te)

m=AutoModel.from_pretrained(VM).vision_model.to(DEV).eval().half()
pr=AutoProcessor.from_pretrained(VM)
feat=[]
with torch.no_grad():
    for i in range(0,len(te),32):
        ims=[Image.open(f'archive/data/images/test/{x}.jpg').convert('RGB') for x in te.img_id[i:i+32]]
        px=pr(images=ims,return_tensors='pt')['pixel_values'].to(DEV).half()
        feat.append(m(pixel_values=px).last_hidden_state.float().cpu())   # [B,196,768]
X=torch.cat(feat); del m; torch.cuda.empty_cache()
print(f'{X.shape[0]} anh, dac trung {tuple(X.shape[1:])}')

# --- cac chi so pho, deu KHONG CAN NHAN ---
met={}
sv=[]
for i in range(len(X)):
    Z=X[i]-X[i].mean(0, keepdim=True)
    s=torch.linalg.svdvals(Z.double()).numpy()
    sv.append(s)
S=np.stack(sv)
p=S**2; p=p/p.sum(1,keepdims=True)
met['participation_ratio']=1.0/ (p**2).sum(1)                       # so huong hieu dung
met['entropy_pho']=-(p*np.log(p+1e-12)).sum(1)                       # entropy pho
met['top1_energy']=p[:,0]                                            # ti le nang luong huong dau
met['top10_energy']=p[:,:10].sum(1)
met['effective_rank']=np.exp(met['entropy_pho'])
met['patch_std']=X.std(1).mean(1).numpy()                            # do bien thien giua patch

oa=ora.alpha_mean.values; ma=base.alpha_mean.values; ok=base.exact_match.values>0.5
print(f'\n{"chi so":22s} {"vs ORACLE alpha":>18s} {"vs alpha MODEL":>16s} {"vs dung/sai":>13s}')
for k,v in met.items():
    r1=stats.spearmanr(v,oa); r2=stats.spearmanr(v,ma); r3=stats.spearmanr(v,ok.astype(float))
    print(f'{k:22s} {r1.correlation:+8.3f} (p={r1.pvalue:.1e}) {r2.correlation:+8.3f} {r3.correlation:+8.3f}')
print('\nchi so nao co |rho| lon voi ORACLE alpha => ung vien lam nhan per-sample')
print('luu y: oracle alpha o day fit tren run87 (SigLIP1) nhung dac trung do tu SigLIP2 —')
print('  neu tuong quan MANH thi van dang theo, se do lai dung cap sau.')
