"""LUOI 4x4 CO DU DOAN DUOC TU DAU VAO CUA GATE KHONG?

Phep do dut diem. Giao thuc y het probe_gateinput_predictability.py (da cho R^2 < 0 tren muc tieu
197 chieu, doi chieu positive control +0.975), nhung muc tieu gio la LUOI 4x4 = 16 chieu.

Ly do: truong alpha toi uu nen duoc dang ke — luoi muot 4x4 thu 59% tran voi 8.6% tham so
(probe_oracle_smooth.py, n=240). Neu that bai truoc day la do SAI DO PHAN GIAI tham so hoa thi
o 16 chieu phai du doan duoc.

Dac trung dau vao: DUNG thu gate_net nhin thay — [v_proj_i ; W_q[t_cls; e_type]] — nhung gop ve
cung luoi 4x4 de khop do phan giai cua muc tieu. Chia fold THEO MAU (khong ro ri).
Doi chieu BAT BUOC:
  - positive control: du doan chinh dau ra pre-sigmoid cua gate_net (da gop 4x4). Phai R^2 cao.
  - null: muc tieu lay tu MAU KHAC.
"""
import sys, argparse, unicodedata as ud
import torch, torch.nn.functional as F, numpy as np, pandas as pd
sys.path.insert(0,'src')
from probe_evidence_readout import build_model
norm=lambda s: ud.normalize('NFC',str(s)).strip().lower()
p=argparse.ArgumentParser()
p.add_argument('--checkpoint',default='checkpoints_run87/best_model.pt')
p.add_argument('--per_type',type=int,default=80); p.add_argument('--steps',type=int,default=25)
p.add_argument('--lr',type=float,default=0.1); p.add_argument('--out',default='analysis/mech/grid_pred')
a=p.parse_args()
from dataset import VQAGenDataset
from torch.utils.data import DataLoader, Subset
from transformers import AutoProcessor
DEV='cuda'
m,sa=build_model(a.checkpoint,DEV); m.vision_gating.gate_mode='multiply'; tok=m.tokenizer
C={}
m.vision_gating.gate_net.register_forward_pre_hook(lambda mo,i: C.__setitem__('gin',i[0].detach()))
tr=pd.read_csv('archive/train_split_original.csv'); tr['an']=tr.answer.map(norm)
TVOC={int(t):sorted(set(g.an)) for t,g in tr.groupby('type')}
te=pd.read_csv('archive/test.csv'); gold=te.answer.map(norm)
vp=AutoProcessor.from_pretrained(sa.get('vision_model'))
ds=VQAGenDataset(csv_path='archive/test.csv',image_folder='archive/data/images/test',
  vision_processor=vp,tokenizer_name='vinai/bartpho-syllable',max_q_len=32,max_a_len=10,
  include_question_type=True,auto_detect_type=False)
LB={}
for t,voc in TVOC.items():
    e=tok(voc,return_tensors='pt',padding='max_length',truncation=True,max_length=10)
    x=e.input_ids.to(DEV).clone(); x[x==tok.pad_token_id]=-100; LB[t]=x
rng=np.random.RandomState(0); idx=[]
for t,g in te.groupby('type'):
    ii=[i for i in g.index if gold.iloc[i] in TVOC[int(t)]]
    idx+=list(rng.choice(ii,min(a.per_type,len(ii)),replace=False))
idx=sorted(int(i) for i in idx)
print(f'{len(idx)} mau | muc tieu = luoi 4x4 (16 chieu) | multiply',flush=True)

X=[]; Y=[]; POS=[]; TY=[]
for n_,j in enumerate(idx):
    t=int(te.type.iloc[j]); voc=TVOC[t]; gi=voc.index(gold.iloc[j]); L=LB[t]
    b=next(iter(DataLoader(Subset(ds,[j]),batch_size=1)))
    kw=dict(pixel_values=b['pixel_values'].to(DEV),input_ids=b['input_ids'].to(DEV),
            attention_mask=b['attention_mask'].to(DEV),
            question_types=b['question_type'].to(DEV).long())
    m.vision_gating.alpha_override=None
    with torch.no_grad():
        _=m(**kw,labels=L[gi:gi+1]); a0=m.vision_gating.last_alpha.detach().float()
    if a0.dim()==3: a0=a0.squeeze(-1)
    P=a0.size(1); npad=P-196
    gin=C['gin'][0]; D=gin.size(-1)//2
    vp_i=gin[npad:,:D].float(); q=gin[0,D:].float()          # [196,D], [D]
    # gop dac trung ve luoi 4x4 -> khop do phan giai voi muc tieu
    vg=F.adaptive_avg_pool2d(vp_i.T.reshape(1,D,14,14),(4,4)).reshape(D,16).T   # [16,D]
    X.append(torch.cat([vg, q.unsqueeze(0).expand(16,-1)],dim=-1).cpu().numpy())
    # positive control: dau ra pre-sigmoid cua gate chinh no, gop 4x4
    pre=torch.logit(a0[0,npad:].clamp(1e-4,1-1e-4))
    POS.append(F.adaptive_avg_pool2d(pre.reshape(1,1,14,14),(4,4)).reshape(16).cpu().numpy())
    # muc tieu: luoi 4x4 oracle
    z=torch.zeros(1,1,4,4,device=DEV,requires_grad=True)
    base_logit=torch.logit(a0[:,npad:].clamp(1e-4,1-1e-4)).view(1,1,14,14)
    opt=torch.optim.Adam([z],lr=a.lr); lab=L[gi:gi+1]
    for _ in range(a.steps):
        opt.zero_grad()
        up=F.interpolate(z,size=(14,14),mode='bilinear',align_corners=False)
        al=torch.sigmoid(base_logit+up).view(1,196)
        if npad>0: al=torch.cat([a0[:,:npad],al],1)
        m.vision_gating.alpha_override=al
        o=m(**kw,labels=lab)
        F.cross_entropy(o.answer_logits.reshape(-1,o.answer_logits.size(-1)).float(),
                        lab.reshape(-1),ignore_index=-100).backward(); opt.step()
    Y.append(z.detach().reshape(16).cpu().numpy()); TY.append(t)
    m.vision_gating.alpha_override=None
    if (n_+1)%50==0: print(f'  {n_+1}/{len(idx)}',flush=True)

X=np.stack(X); Y=np.stack(Y); POS=np.stack(POS); TY=np.array(TY)
np.savez_compressed(a.out+'.npz',X=X,Y=Y,POS=POS,TY=TY,idx=np.array(idx[:len(Y)]))
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
N=len(Y); grp=np.repeat(np.arange(N),16)
Xf=X.reshape(N*16,-1); 
def cvr2(target_flat):
    o=np.zeros_like(target_flat)
    for tr_,va in GroupKFold(5).split(Xf,target_flat,grp):
        r=Ridge(alpha=10.0).fit(Xf[tr_],target_flat[tr_]); o[va]=r.predict(Xf[va])
    return r2_score(target_flat,o)
print(f'\n=== n={N} mau x 16 o luoi | chia fold THEO MAU ===')
print(f'{"muc tieu":<44} {"R^2 (CV 5-fold)":>16}')
print(f'{"positive control: pre-sigmoid cua chinh gate":<44} {cvr2(POS.reshape(-1)):>16.4f}')
print(f'{"LUOI 4x4 oracle (16 chieu)":<44} {cvr2(Y.reshape(-1)):>16.4f}')
Yc=Y-Y.mean(1,keepdims=True)
print(f'{"... rieng HINH DANG trong-mau (da tru trung binh)":<44} {cvr2(Yc.reshape(-1)):>16.4f}')
perm=np.random.RandomState(0).permutation(N)
print(f'{"NULL: luoi lay tu MAU KHAC":<44} {cvr2(Y[perm].reshape(-1)):>16.4f}')
print(f'\nMOC 197 chieu (probe_gateinput_predictability.py): R^2 = -0.042 (all) / -0.101 (rescued)')
print('DOC: R^2 > 0 ro ret o 16 chieu -> that bai truoc day la do SAI DO PHAN GIAI, ban sua co co so.')
print('     R^2 <= 0 -> muc tieu khong du doan duoc o MOI do phan giai; ket luan khong hoc duoc vung hon.')
