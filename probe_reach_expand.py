"""BUOC 4 — TCVG tu TAT chinh no tren 64% test (alpha=1.000 cho OBJECT va LOCATION).
Do la LUA CHON HOC DUOC hay la RANG BUOC BAT BUOC?

Do duoc o buoc 3 (run87, cung trong so):
  - chat luong can thiep TOT: RESCUE 32 / DAMAGE 17 = 1.88:1, McNemar p=0.044
  - nhung chi cham 1.65% mau; OBJECT va LOCATION alpha = 1.000 tuc TOAN TU LA DONG NHAT
  - du dia chua voi toi: 468 mau (15.8%) sai ca hai nhanh voi gold rank<=3

Cau hoi: NEU ep gate hanh dong tren dung hai loai no da tu tat, ti le RESCUE:DAMAGE co giu
duoc >= 1 khong?
  ti le >= 1  -> alpha=1 la BAO THU DO HOC, tam voi mo rong duoc -> co loi thoat
  ti le <  1  -> alpha=1 la NGHIEM DUNG, TCVG dang o tran cua no -> dong that su

Khong train gi, khong khop oracle. Chi alpha_override hang so tren tap con OBJECT+LOCATION.
"""
import sys, argparse, unicodedata as ud
import torch, torch.nn.functional as F, numpy as np, pandas as pd
sys.path.insert(0,'src')
from probe_evidence_readout import build_model
norm=lambda s: ud.normalize('NFC',str(s)).strip().lower()
p=argparse.ArgumentParser()
p.add_argument('--checkpoint',required=True); p.add_argument('--out',required=True)
p.add_argument('--train_csv',default='archive/train_split_original.csv')
p.add_argument('--test_csv',default='archive/test.csv')
p.add_argument('--image_folder',default='archive/data/images/test')
p.add_argument('--types',default='0,3')          # OBJECT, LOCATION
p.add_argument('--alphas',default='0.95,0.9,0.85,0.8')
p.add_argument('--chunk',type=int,default=48)
a=p.parse_args()
TYPES=[int(x) for x in a.types.split(',')]; ALPHAS=[float(x) for x in a.alphas.split(',')]
from dataset import VQAGenDataset
from torch.utils.data import DataLoader, Subset
from transformers import AutoProcessor
DEV='cuda'; m,sa=build_model(a.checkpoint,DEV); tok=m.tokenizer
tr=pd.read_csv(a.train_csv); tr['an']=tr.answer.map(norm)
TVOC={int(t):sorted(set(g.an)) for t,g in tr.groupby('type')}
LB={}
for t,voc in TVOC.items():
    e=tok(voc,return_tensors='pt',padding='max_length',truncation=True,max_length=10)
    x=e.input_ids.to(DEV).clone(); x[x==tok.pad_token_id]=-100; LB[t]=x
te=pd.read_csv(a.test_csv); gold=te.answer.map(norm)
vp=AutoProcessor.from_pretrained(sa.get('vision_model'))
ds=VQAGenDataset(csv_path=a.test_csv,image_folder=a.image_folder,vision_processor=vp,
  tokenizer_name='vinai/bartpho-syllable',max_q_len=32,max_a_len=10,
  include_question_type=True,auto_detect_type=False)
def logp(lg,lb): return -F.cross_entropy(lg.reshape(-1,lg.size(-1)).float(),lb.reshape(-1),
                       ignore_index=-100,reduction='none').view(lb.shape).sum(1)
R=[]
idxs=[j for j in range(len(te)) if int(te.type.iloc[j]) in TYPES
      and gold.iloc[j] in TVOC[int(te.type.iloc[j])]]
print(f'{len(idxs)} mau thuoc loai {TYPES}; alpha {ALPHAS}',flush=True)
for c_,j in enumerate(idxs):
    t=int(te.type.iloc[j]); L=LB[t]; voc=TVOC[t]; gi=voc.index(gold.iloc[j])
    b=next(iter(DataLoader(Subset(ds,[j]),batch_size=1)))
    pv=b['pixel_values'].to(DEV); ii=b['input_ids'].to(DEV)
    am=b['attention_mask'].to(DEV); qt=b['question_type'].to(DEV).long()
    d={}
    for av in [None]+ALPHAS:
        s=torch.empty(L.size(0),device=DEV)
        with torch.no_grad():
            for st in range(0,L.size(0),a.chunk):
                x=L[st:st+a.chunk]; k=x.size(0)
                m.vision_gating.alpha_override=(None if av is None else
                                                torch.full((k,1),av,device=DEV))
                o=m(pixel_values=pv.expand(k,-1,-1,-1),input_ids=ii.expand(k,-1),
                    attention_mask=am.expand(k,-1),labels=x,question_types=qt.expand(k))
                s[st:st+k]=logp(o.answer_logits,x)
        s=s.float(); o_=torch.argsort(s,descending=True); c1=int(o_[0])
        tag='base' if av is None else f'{av}'
        d[f'hit_{tag}']=int(c1==gi)
        d[f'margin_{tag}']=float(s[c1]-s[int(o_[1])])
        d[f'rank_{tag}']=int((o_==gi).nonzero()[0,0])+1
    m.vision_gating.alpha_override=None
    R.append(dict(idx=j,type=t,nvoc=len(voc),**d))
    if (c_+1)%200==0:
        x=pd.DataFrame(R); msg=' '.join(f'a{v}:{x[f"hit_{v}"].mean():.4f}' for v in ALPHAS)
        print(f'  {c_+1}/{len(idxs)} base:{x.hit_base.mean():.4f} {msg}',flush=True)
df=pd.DataFrame(R); df.to_csv(a.out,index=False)
print(f'\nluu {len(df)} -> {a.out}')
print(f'\n{"alpha":>7} {"EM":>7} {"RESCUE":>7} {"DAMAGE":>7} {"ti le":>7} {"net":>6} {"McNemar p":>10}')
from scipy import stats
print(f'{"base":>7} {df.hit_base.mean()*100:>7.2f} {"-":>7} {"-":>7} {"-":>7} {"-":>6} {"-":>10}')
for v in ALPHAS:
    r=int(((df.hit_base==0)&(df[f'hit_{v}']==1)).sum()); dd=int(((df.hit_base==1)&(df[f'hit_{v}']==0)).sum())
    pp=stats.binomtest(r,r+dd,0.5).pvalue if r+dd>0 else float('nan')
    print(f'{v:>7} {df[f"hit_{v}"].mean()*100:>7.2f} {r:>7} {dd:>7} {r/max(dd,1):>7.2f} {r-dd:>+6} {pp:>10.4f}')
