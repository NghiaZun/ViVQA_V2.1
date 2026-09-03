"""PHEP THU: thap van ban SigLIP co xep hang duoc dap an TIENG VIET tu anh khong?

Cau hoi quyet dinh cho nhanh TRUY HOI open-vocab. Decoder hien tai de gold cua lop CHUA TRAIN o
rank trung vi 75/331 (32 sau khi sua prior PMI). Neu truy hoi SigLIP dat rank <= 5 thi no giai
duoc rao can TRI GIAC ma moi can thiep phia decoder deu khong dong toi.

KHONG train gi. Nhanh truy hoi la BO CHAM RIENG nen dung SigLIP multilingual o day khong phai
"doi encoder cua pipeline" — vision encoder chinh van la siglip-base-patch16-224.

Doi chung bat buoc: mot tap dap an XAO (gan ngau nhien) de chac rank thap khong den tu artefact.
"""
import sys, argparse, unicodedata as ud
import torch, pandas as pd, numpy as np
from PIL import Image
norm=lambda s: ud.normalize('NFC',str(s)).strip().lower()

p=argparse.ArgumentParser()
p.add_argument('--model',default='google/siglip-base-patch16-256-multilingual')
p.add_argument('--n',type=int,default=0)
p.add_argument('--out',default='analysis/openvocab/siglip_retr.csv')
a=p.parse_args()

from transformers import AutoProcessor, AutoModel
DEV='cuda'
print(f'nap {a.model}',flush=True)
proc=AutoProcessor.from_pretrained(a.model)
m=AutoModel.from_pretrained(a.model).to(DEV).eval()


def _t(o):
    """get_*_features co the tra ve tensor hoac BaseModelOutputWithPooling tuy phien ban."""
    if hasattr(o,'pooler_output'): return o.pooler_output
    if hasattr(o,'last_hidden_state'): return o.last_hidden_state[:,0]
    return o

tr=pd.read_csv('archive/train_split.csv'); te=pd.read_csv('archive/test.csv')
V=sorted(set(tr.answer.map(norm)))
gold=te.answer.map(norm)
oov=~gold.isin(set(V))
# tap ung vien = 328 dap an train + cac gold OOV (de OOV co the duoc chon)
CAND=V+sorted(set(gold[oov]))
print(f'{len(V)} dap an train + {oov.sum()} gold OOV = {len(CAND)} ung vien')

with torch.no_grad():
    ti=proc(text=CAND,padding='max_length',truncation=True,max_length=64,return_tensors='pt')
    TE=[]
    for s in range(0,len(CAND),128):
        TE.append(_t(m.get_text_features(**{k:v[s:s+128].to(DEV) for k,v in ti.items()})).float())
    TE=torch.nn.functional.normalize(torch.cat(TE),dim=-1)
print(f'embedding van ban: {tuple(TE.shape)}',flush=True)

idx=list(range(len(te)))[:a.n] if a.n else list(range(len(te)))
R=[]; B=32
for s in range(0,len(idx),B):
    chunk=idx[s:s+B]; ims=[]
    for j in chunk:
        ims.append(Image.open(f"archive/data/images/test/{te.img_id.iloc[j]}.jpg").convert('RGB'))
    with torch.no_grad():
        vi=proc(images=ims,return_tensors='pt')
        IE=torch.nn.functional.normalize(
            _t(m.get_image_features(pixel_values=vi['pixel_values'].to(DEV))).float(),dim=-1)
        S=IE@TE.T                                        # [b, C]
    for bi,j in enumerate(chunk):
        gi=CAND.index(gold.iloc[j])
        order=torch.argsort(S[bi],descending=True)
        rank=int((order==gi).nonzero()[0,0])+1
        R.append(dict(idx=j,oov=bool(oov.iloc[j]),rank=rank,
                      top1=CAND[int(order[0])],gold=gold.iloc[j],
                      hit=int(rank==1),top5=int(rank<=5)))
    if (s+B)%640==0: print(f'  {s+B}/{len(idx)}',flush=True)

D=pd.DataFrame(R); D.to_csv(a.out,index=False)
print(f'\n=== TRUY HOI SigLIP thuan, {len(D)} mau, {len(CAND)} ung vien ===')
for name,sub in (('TAT CA',D),('trong tu vung',D[~D.oov]),('NGOAI tu vung (OOV)',D[D.oov])):
    if not len(sub): continue
    print(f'{name:<22} n={len(sub):<5} top1 {100*sub.hit.mean():5.2f}%  '
          f'top5 {100*sub.top5.mean():5.2f}%  rank trung vi {sub["rank"].median():6.1f}')
print(f'\nMOC de so (rank trung vi cua gold LOP CHUA TRAIN duoi decoder):')
print(f'  decoder khong sua prior : 75.0')
print(f'  decoder + PMI lambda=1.0: 32.0')
print(f'  -> truy hoi SigLIP tren OOV: {D[D.oov]["rank"].median() if D.oov.any() else float("nan"):.1f}')
print('\nDOC: rank <=5 -> nhanh truy hoi giai duoc rao can tri giac, dang xay.')
print('     rank ~30-75 -> SigLIP cung khong nhan ra, huong dong (ket qua sach).')
