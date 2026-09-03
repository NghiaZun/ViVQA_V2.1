"""NGHIEM TOI UU CO NAM TRONG KHONG GIAN IT CHIEU KHONG?

Sua mot loi doc cua chinh toi: toi da coi "truong alpha khuech tan + truc giao" la bang chung
CHONG chon loc. Nhung luoi 14x14 = 196 patch, mot vat the trai tren HANG CHUC patch. Nen
"chon vung chua vat the" — khi ep thanh 197 so vo huong DOC LAP — se trong DUNG NHU vay:
khuech tan (nhieu patch tham gia) va truc giao giua cac mau (vat the khac cho). Do la
chu ky cua chon loc muc VUNG, khong phai bang chung chong chon loc.

Ho tro: sp_moran_disc (Moran's I — tu tuong quan KHONG GIAN) co AUROC 0.631 cho viec cuu duoc,
tuc truong toi uu CO cau truc khong gian that.

PHEP THU: khop lai oracle alpha nhung RANG BUOC muot — tham so hoa bang luoi tho kxk roi noi suy
len 14x14. Neu tran giu duoc voi it dof thi nghiem nam trong khong gian it chieu -> HOC DUOC,
va moi probe truoc day that bai vi tham so hoa o SAI DO PHAN GIAI (197 dof cho mot nghiem ~16 dof).

Toan tu: multiply (toan tu chon loc that; tran tu do da do = +7.08 tren cung 240 mau).
"""
import sys, argparse, unicodedata as ud
import torch, torch.nn.functional as F, numpy as np, pandas as pd
sys.path.insert(0,'src')
from probe_evidence_readout import build_model
norm=lambda s: ud.normalize('NFC',str(s)).strip().lower()
p=argparse.ArgumentParser()
p.add_argument('--checkpoint',default='checkpoints_run87/best_model.pt')
p.add_argument('--per_type',type=int,default=60); p.add_argument('--steps',type=int,default=25)
p.add_argument('--lr',type=float,default=0.1); p.add_argument('--grids',default='2,4,7,14')
p.add_argument('--out',default='analysis/mech/oracle_smooth.csv')
a=p.parse_args()
GRIDS=[int(x) for x in a.grids.split(',')]
from dataset import VQAGenDataset
from torch.utils.data import DataLoader, Subset
from transformers import AutoProcessor
DEV='cuda'
m,sa=build_model(a.checkpoint,DEV); m.vision_gating.gate_mode='multiply'; tok=m.tokenizer
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
print(f'{len(idx)} mau | luoi {GRIDS} (14 = KHONG rang buoc, doi chieu) | multiply',flush=True)

rows=[]
for n_,j in enumerate(idx):
    t=int(te.type.iloc[j]); voc=TVOC[t]; gi=voc.index(gold.iloc[j]); L=LB[t]
    b=next(iter(DataLoader(Subset(ds,[j]),batch_size=1)))
    kw=dict(pixel_values=b['pixel_values'].to(DEV),input_ids=b['input_ids'].to(DEV),
            attention_mask=b['attention_mask'].to(DEV),
            question_types=b['question_type'].to(DEV).long())
    def score(ov):
        s=torch.empty(len(voc),device=DEV)
        with torch.no_grad():
            for st in range(0,len(voc),48):
                x=L[st:st+48]; k=x.size(0)
                m.vision_gating.alpha_override=(None if ov is None else ov.detach().expand(k,-1))
                o=m(**{kk:(vv.expand(k,*vv.shape[1:]) if vv.dim()>1 else vv.expand(k))
                       for kk,vv in kw.items()},labels=x)
                s[st:st+k]=-F.cross_entropy(o.answer_logits.reshape(-1,o.answer_logits.size(-1)).float(),
                    x.reshape(-1),ignore_index=-100,reduction='none').view(x.shape).sum(1)
        return s
    base=score(None); hit_base=int(int(base.argmax())==gi)
    m.vision_gating.alpha_override=None
    with torch.no_grad():
        _=m(**kw,labels=L[gi:gi+1]); a0=m.vision_gating.last_alpha.detach().float()
    if a0.dim()==3: a0=a0.squeeze(-1)
    P=a0.size(1); npad=P-196                      # token pooler o dau (neu co)
    r=dict(idx=j,type=t,hit_base=hit_base)
    for G in GRIDS:
        # tham so hoa alpha bang luoi GxG roi noi suy len 14x14 -> dof = G*G (+1 cho pooler)
        z=torch.zeros(1,1,G,G,device=DEV,requires_grad=True)
        zp=torch.zeros(1,npad,device=DEV,requires_grad=True) if npad>0 else None
        base_logit=torch.logit(a0[:,npad:].clamp(1e-4,1-1e-4)).view(1,1,14,14)
        prm=[z]+([zp] if zp is not None else [])
        opt=torch.optim.Adam(prm,lr=a.lr); lab=L[gi:gi+1]
        for _ in range(a.steps):
            opt.zero_grad()
            up=F.interpolate(z,size=(14,14),mode='bilinear',align_corners=False)
            al=torch.sigmoid(base_logit+up).view(1,196)
            if npad>0: al=torch.cat([torch.sigmoid(torch.logit(a0[:,:npad].clamp(1e-4,1-1e-4))+zp),al],1)
            m.vision_gating.alpha_override=al
            o=m(**kw,labels=lab)
            loss=F.cross_entropy(o.answer_logits.reshape(-1,o.answer_logits.size(-1)).float(),
                                 lab.reshape(-1),ignore_index=-100)
            loss.backward(); opt.step()
        with torch.no_grad():
            up=F.interpolate(z,size=(14,14),mode='bilinear',align_corners=False)
            al=torch.sigmoid(base_logit+up).view(1,196)
            if npad>0: al=torch.cat([torch.sigmoid(torch.logit(a0[:,:npad].clamp(1e-4,1-1e-4))+zp),al],1)
        r[f'hit_g{G}']=int(int(score(al).argmax())==gi); r[f'dof_g{G}']=G*G+npad
        m.vision_gating.alpha_override=None
    rows.append(r)
    if (n_+1)%40==0:
        D=pd.DataFrame(rows)
        print(f'  {n_+1}/{len(idx)} base {100*D.hit_base.mean():.2f} | ' +
              ' '.join(f'g{G} {100*D[f"hit_g{G}"].mean():.2f}' for G in GRIDS),flush=True)

D=pd.DataFrame(rows); D.to_csv(a.out,index=False)
print(f'\n=== n={len(D)} | toan tu multiply | run87 (train bang blend — lech phan bo) ===')
print(f'{"tham so hoa":<24} {"dof/mau":>8} {"EM":>7} {"tran":>7} {"% giu duoc cua tu do":>21}')
print(f'{"base (alpha model)":<24} {"—":>8} {100*D.hit_base.mean():>7.2f} {"—":>7} {"—":>21}')
full=100*(D[f'hit_g{max(GRIDS)}'].mean()-D.hit_base.mean())
for G in GRIDS:
    c=100*(D[f'hit_g{G}'].mean()-D.hit_base.mean())
    lab='KHONG rang buoc' if G==max(GRIDS) else f'luoi {G}x{G} noi suy'
    print(f'{lab:<24} {int(D[f"dof_g{G}"].iloc[0]):>8} {100*D[f"hit_g{G}"].mean():>7.2f} '
          f'{c:>+7.2f} {(100*c/full if full else float("nan")):>20.0f}%')
print('\nDOC: neu luoi THO giu duoc phan lon tran cua 197-dof -> nghiem nam trong khong gian')
print('     IT CHIEU -> HOC DUOC, va cac probe truoc that bai vi SAI DO PHAN GIAI tham so hoa.')
print('     Neu tran sup khi rang buoc -> nghiem that su can 197 dof doc lap.')
