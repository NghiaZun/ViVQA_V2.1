"""Closed-set candidate margin for every TEST row under a fixed reference model.

Used to stratify the test set by recoverability (low margin = the subset where visual
intervention can work, AUROC 0.70 for oracle rescue). The reference model is held FIXED and is
not one of the arms being compared, so the stratification is independent of the comparison.
"""
import sys, unicodedata as ud, argparse
import torch, torch.nn.functional as F, numpy as np, pandas as pd
sys.path.insert(0,'src')
from model import DeterministicVQA
from dataset import VQAGenDataset
from torch.utils.data import DataLoader
from transformers import AutoProcessor
p=argparse.ArgumentParser()
p.add_argument('--checkpoint',required=True); p.add_argument('--train_csv',default=None)
p.add_argument('--out',required=True); p.add_argument('--chunk',type=int,default=64)
a=p.parse_args()
DEV='cuda'; norm=lambda s: ud.normalize('NFC',str(s)).strip().lower()
ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
sa=ck['args']; sa=sa if isinstance(sa,dict) else vars(sa); sd=ck['model_state_dict']; K=list(sd.keys())
tlr=next((sd[k].shape[0] for k in K if k.startswith('encoder.base_model.model.layers.0.self_attn.q_proj.lora_A')),16)
m=DeterministicVQA(vision_model_name=sa.get('vision_model'),bartpho_model_name='vinai/bartpho-syllable',
  num_fusion_layers=sa.get('num_fusion_layers',2),fusion_type=sa.get('fusion_type','text2vision'),
  use_text_lora=True,text_lora_r=tlr,text_lora_alpha=sa.get('text_lora_alpha',32),
  use_vision_gate=True,vision_gate_init=sa.get('vision_gate_init',1.5),
  vision_gate_min_alpha=sa.get('vision_gate_min_alpha',0.0),vision_gate_max_alpha=sa.get('vision_gate_max_alpha',1.0),
  use_type_task=any(k.startswith('type_head.') or k.startswith('type_classifier.') for k in K),
  use_siglip_pooler=sa.get('use_siglip_pooler',True)).to(DEV).eval()
r=m.load_state_dict(sd,strict=False); assert not [k for k in r.missing_keys if 'teacher' not in k]
for q in m.parameters(): q.requires_grad_(False)
tok=m.tokenizer
tr=pd.read_csv(a.train_csv or sa.get('train_csv','archive/train_split_original.csv')); tr['an']=tr.answer.map(norm)
TVOC={t:sorted(set(g.an)) for t,g in tr.groupby('type')}
LB={}
for t,voc in TVOC.items():
    e=tok(voc,return_tensors='pt',padding='max_length',truncation=True,max_length=10)
    x=e.input_ids.to(DEV).clone(); x[x==tok.pad_token_id]=-100; LB[t]=x
te=pd.read_csv('archive/test.csv'); gold=te.answer.map(norm)
vp=AutoProcessor.from_pretrained(sa.get('vision_model'))
ds=VQAGenDataset(csv_path='archive/test.csv',image_folder='archive/data/images/test',vision_processor=vp,
  tokenizer_name='vinai/bartpho-syllable',max_q_len=32,max_a_len=10,include_question_type=True,auto_detect_type=False)
def logp(lg,lb): return -F.cross_entropy(lg.reshape(-1,lg.size(-1)).float(),lb.reshape(-1),ignore_index=-100,reduction='none').view(lb.shape).sum(1)
rows=[]
dl=DataLoader(ds,batch_size=1,shuffle=False,num_workers=2)
import time; t0=time.time()
for j,b in enumerate(dl):
    pv=b['pixel_values'].to(DEV); ii=b['input_ids'].to(DEV); am=b['attention_mask'].to(DEV); qt=b['question_type'].to(DEV).long()
    t=int(te.type.iloc[j]); L=LB[t]
    s=torch.empty(L.size(0),device=DEV)
    with torch.no_grad():
        for st in range(0,L.size(0),a.chunk):
            x=L[st:st+a.chunk]; k=x.size(0)
            o=m(pixel_values=pv.expand(k,-1,-1,-1),input_ids=ii.expand(k,-1),attention_mask=am.expand(k,-1),labels=x,question_types=qt.expand(k))
            s[st:st+k]=logp(o.answer_logits,x)
    sn=s.cpu().numpy(); o2=np.argsort(-sn)
    g=TVOC[t].index(gold.iloc[j]) if gold.iloc[j] in TVOC[t] else -1
    rows.append(dict(row=j,type=t,margin=float(sn[o2[0]]-sn[o2[1]]),
                     rank_gold=int((sn>sn[g]).sum())+1 if g>=0 else -1))
    if (j+1)%500==0:
        pd.DataFrame(rows).to_csv(a.out,index=False); print(f'  {j+1}/{len(ds)} ({(time.time()-t0)/60:.1f}m)',flush=True)
pd.DataFrame(rows).to_csv(a.out,index=False); print(f'saved {a.out}')
