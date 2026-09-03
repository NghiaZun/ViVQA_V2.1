"""Dump the PRE-INTERVENTION state of every error sample, in the space the decoder actually reads.

For each sample where the base model is WRONG, capture (with the model's OWN alpha, no override):
  post   : gated_vision after TCVG + LayerNorm  [P, D]   <- exactly what decoder cross-attention sees
  vproj  : vision_gating.vision_proj(v_fused)   [P, D]   <- what the gate scores
  q      : the gate query W_q[t_cls ; e_type]   [D]
  tproj  : projected text tokens                [L, D]
  alpha  : the model's own per-patch alpha      [P]
  scores : full closed-set candidate score vector over the type vocabulary
Plus the label of interest: whether the ORACLE per-patch alpha rescues it (from mech csv).

Everything captured here is available at inference time. The oracle outcome is the target.
"""
import sys, argparse, unicodedata as ud
import torch, torch.nn.functional as F, numpy as np, pandas as pd
sys.path.insert(0, 'src')
from model import DeterministicVQA
from dataset import VQAGenDataset
from torch.utils.data import DataLoader, Subset
from transformers import AutoProcessor

p = argparse.ArgumentParser()
p.add_argument('--checkpoint', required=True); p.add_argument('--mech_csv', required=True)
p.add_argument('--train_csv', default=None); p.add_argument('--out', required=True)
p.add_argument('--chunk', type=int, default=32)
a = p.parse_args()
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
C={}
m.vision_gating.gate_net.register_forward_pre_hook(lambda mo,i: C.__setitem__('gin',i[0].detach()))
m.vision_gating.text_proj.register_forward_hook(lambda mo,i,o: C.__setitem__('tproj',o.detach()))
m.vision_gating.layer_norm.register_forward_hook(lambda mo,i,o: C.__setitem__('post',o.detach()))

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
mech=pd.read_csv(a.mech_csv); err=mech[mech.hit_base==0]
POST=[];VPR=[];Q=[];AL=[];SC={};META=[]
for n,row in enumerate(err.itertuples()):
    j=int(row.idx); t=int(row.type); L=LB[t]; voc=TVOC[t]
    b=next(iter(DataLoader(Subset(ds,[j]),batch_size=1)))
    pv=b['pixel_values'].to(DEV); ii=b['input_ids'].to(DEV); am=b['attention_mask'].to(DEV); qt=b['question_type'].to(DEV).long()
    m.vision_gating.alpha_override=None
    s=torch.empty(L.size(0),device=DEV)
    with torch.no_grad():
        for st in range(0,L.size(0),a.chunk):
            x=L[st:st+a.chunk]; k=x.size(0)
            o=m(pixel_values=pv.expand(k,-1,-1,-1),input_ids=ii.expand(k,-1),attention_mask=am.expand(k,-1),labels=x,question_types=qt.expand(k))
            s[st:st+k]=logp(o.answer_logits,x)
    gin=C['gin']; D=gin.size(-1)//2
    POST.append(C['post'][0].half().cpu().numpy()); VPR.append(gin[0,:,:D].half().cpu().numpy())
    Q.append(gin[0,0,D:].half().cpu().numpy())
    al=m.vision_gating.last_alpha.detach().float()
    if al.dim()==3: al=al.squeeze(-1)
    AL.append(al[0].cpu().numpy())
    SC[f'{t}_{j}']=s.cpu().numpy()
    gi=voc.index(gold.iloc[j])
    META.append(dict(idx=j,type=t,rescued=int(row.hit_fit),rank_base=int(row.rank_base),gold_i=gi,nvoc=len(voc),
                     Lreal=int(am[0].sum().item())))
    np.save('/dev/null',np.array([0])) if False else None
    if (n+1)%50==0: print(f'  {n+1}/{len(err)}',flush=True)
np.savez_compressed(a.out, post=np.stack(POST), vproj=np.stack(VPR), q=np.stack(Q), alpha=np.stack(AL), **SC)
pd.DataFrame(META).to_csv(a.out.replace('.npz','_meta.csv'),index=False)
print(f'saved {a.out}  post={np.stack(POST).shape}')
