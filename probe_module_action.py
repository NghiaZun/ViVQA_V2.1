"""GCA va TCVG co THUC SU day dap an len TOP-1 khong?

Do da co: TCVG doi quyet dinh o 1.65% mau (49/2970), RESCUE 32 / DAMAGE 17.
Chua bao gio do: GCA. Day la phan ra nhan qua 2x2 NGAY TREN MODEL DA TRAIN
(cung trong so, chi tat/bat toan tu luc suy luan) — nen moi thay doi deu quy duoc cho module.

  full      : GCA bat, TCVG bat
  tcvg_off  : alpha_override = 1.0  -> v_hat = LN(v), khong tiem text
  gca_off   : gca_strength = 0.0    -> bo residual cross-attn cua Flamingo
  both_off  : ca hai

Do: (a) bao nhieu % mau DOI quyet dinh, (b) HANG cua gold dich bao nhieu.
Neu ca hai deu khong dich hang gold -> khong module nao day dap an len top-1.

CANH BAO: tat luc suy luan la LECH PHAN BO (model train voi ca hai bat). Doc KHOI LUONG
va HUONG, khong doc EM tuyet doi.
"""
import sys, argparse, unicodedata as ud
import torch, torch.nn.functional as F, numpy as np, pandas as pd
sys.path.insert(0,'src')
from probe_evidence_readout import build_model
norm=lambda s: ud.normalize('NFC',str(s)).strip().lower()
p=argparse.ArgumentParser(); p.add_argument('--per_type',type=int,default=100)
p.add_argument('--out',default='analysis/mech/module_action.csv'); a=p.parse_args()
from dataset import VQAGenDataset
from torch.utils.data import DataLoader, Subset
from transformers import AutoProcessor
DEV='cuda'; m,sa=build_model('checkpoints_run87/best_model.pt',DEV); tok=m.tokenizer
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
BASE_GCA=float(getattr(m,'gca_strength',1.0))
print(f'{len(idx)} mau | gca_strength goc = {BASE_GCA}',flush=True)
R=[]
for n_,j in enumerate(idx):
    t=int(te.type.iloc[j]); voc=TVOC[t]; gi=voc.index(gold.iloc[j]); L=LB[t]
    b=next(iter(DataLoader(Subset(ds,[j]),batch_size=1)))
    kw=dict(pixel_values=b['pixel_values'].to(DEV),input_ids=b['input_ids'].to(DEV),
            attention_mask=b['attention_mask'].to(DEV),question_types=b['question_type'].to(DEV).long())
    def sc():
        s=torch.empty(len(voc),device=DEV)
        with torch.no_grad():
            for st in range(0,len(voc),48):
                x=L[st:st+48]; k=x.size(0)
                if m.vision_gating.alpha_override is not None:
                    m.vision_gating.alpha_override=torch.ones(k,1,device=DEV)
                o=m(**{kk:(vv.expand(k,*vv.shape[1:]) if vv.dim()>1 else vv.expand(k))
                       for kk,vv in kw.items()},labels=x)
                s[st:st+k]=-F.cross_entropy(o.answer_logits.reshape(-1,o.answer_logits.size(-1)).float(),
                    x.reshape(-1),ignore_index=-100,reduction='none').view(x.shape).sum(1)
        return s
    out={}
    for name,tcvg_off,gca_off in (('full',0,0),('tcvg_off',1,0),('gca_off',0,1),('both_off',1,1)):
        m.vision_gating.alpha_override=(torch.ones(1,1,device=DEV) if tcvg_off else None)
        m.gca_strength=0.0 if gca_off else BASE_GCA
        s=sc(); o=torch.argsort(s,descending=True)
        out[name]=(int(o[0]), int((o==gi).nonzero()[0,0])+1)
    m.vision_gating.alpha_override=None; m.gca_strength=BASE_GCA
    R.append(dict(idx=j,type=t,gold=gi,
                  **{f'{k}_top1':v[0] for k,v in out.items()},
                  **{f'{k}_rank':v[1] for k,v in out.items()}))
    if (n_+1)%50==0: print(f'  {n_+1}/{len(idx)}',flush=True)
D=pd.DataFrame(R); D.to_csv(a.out,index=False)
print(f'\n=== n={len(D)} | phan ra nhan qua tren CUNG trong so ===')
print(f'{"tat module":<12} {"EM":>7} {"% doi quyet dinh":>18} {"hang gold trung vi":>20} {"hang xau di":>13}')
base_rank=D.full_rank.values
print(f'{"(full)":<12} {100*(D.full_top1==D.gold).mean():>7.2f} {"—":>18} {np.median(base_rank):>20.1f} {"—":>13}')
for k in ('tcvg_off','gca_off','both_off'):
    ch=100*(D[f'{k}_top1']!=D.full_top1).mean()
    worse=100*(D[f'{k}_rank']>base_rank).mean()
    print(f'{k:<12} {100*(D[f"{k}_top1"]==D.gold).mean():>7.2f} {ch:>17.2f}% '
          f'{np.median(D[f"{k}_rank"]):>20.1f} {worse:>12.1f}%')
print('\nDOC: "% doi quyet dinh" = khoi luong tac dong nhan qua cua module.')
print('     "hang xau di" = ty le mau ma tat module lam gold TUT hang -> module DANG day gold len.')
