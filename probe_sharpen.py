"""MINIMAL COUNTERFACTUAL: can TCVG control CONTRAST instead of SUPPRESSION?

Three independent measurements say the same thing: the blend is a CONTRACTION, and rescue happens
where it contracts LEAST (patch cloud stays spread; candidate scores stay separated). The current
formulation is structurally one-directional --

    v_hat_i = alpha_i * v_i + (1 - alpha_i) * t_bar,  alpha in [0, 1]

is a convex combination, so it can only move patches TOWARD one shared point t_bar. It cannot
increase the diversity of the patch cloud, which is exactly the quantity that tracks rescue.

Rewriting the same expression:

    v_hat_i = v_i + (alpha_i - 1) * (v_i - t_bar)

alpha > 1 is EXTRAPOLATION away from t_bar -- evidence sharpening rather than text substitution.
It is the identical formula and the identical parameter; only the RANGE changes. `alpha_override`
applies its value with no clamp when the checkpoint uses min=0/max=1, so alpha > 1 is testable at
eval time with NO training and NO code change to the model.

This script sweeps a constant alpha across the contraction side and the extrapolation side on the
same 600 samples used by probe_mechanism.py, reporting rescue and breakage separately.
"""
import sys, argparse, unicodedata as ud
import torch, torch.nn.functional as F, numpy as np, pandas as pd
sys.path.insert(0, 'src')
from model import DeterministicVQA
from dataset import VQAGenDataset
from torch.utils.data import DataLoader, Subset
from transformers import AutoProcessor

p = argparse.ArgumentParser()
p.add_argument('--checkpoint', required=True)
p.add_argument('--train_csv', default=None)
p.add_argument('--per_type', type=int, default=150)
p.add_argument('--alphas', default='0.75,1.0,1.1,1.25,1.5,2.0,3.0')
p.add_argument('--tag', required=True)
a = p.parse_args()
DEV='cuda'; norm=lambda s: ud.normalize('NFC',str(s)).strip().lower()
T={0:'OBJECT',1:'COUNT',2:'COLOR',3:'LOCATION'}
AL=[float(x) for x in a.alphas.split(',')]

ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
sa=ck['args']; sa=sa if isinstance(sa,dict) else vars(sa); sd=ck['model_state_dict']; K=list(sd.keys())
assert float(sa.get('vision_gate_min_alpha',0.0))==0.0 and float(sa.get('vision_gate_max_alpha',1.0))==1.0
tlr=next((sd[k].shape[0] for k in K if k.startswith('encoder.base_model.model.layers.0.self_attn.q_proj.lora_A')),16)
m=DeterministicVQA(vision_model_name=sa.get('vision_model'),bartpho_model_name='vinai/bartpho-syllable',
    num_fusion_layers=sa.get('num_fusion_layers',2),fusion_type=sa.get('fusion_type','text2vision'),
    use_text_lora=True,text_lora_r=tlr,text_lora_alpha=sa.get('text_lora_alpha',32),
    use_vision_gate=True,vision_gate_init=sa.get('vision_gate_init',1.5),
    vision_gate_min_alpha=0.0,vision_gate_max_alpha=1.0,
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
rng=np.random.default_rng(0); sel=[]
for t in [0,1,2,3]:
    pool=[i for i in range(len(te)) if int(te.type.iloc[i])==t and gold.iloc[i] in TVOC[t]]
    sel+=rng.choice(pool,min(a.per_type,len(pool)),replace=False).tolist()
sel=sorted(sel)
def logp(lg,lb): return -F.cross_entropy(lg.reshape(-1,lg.size(-1)).float(),lb.reshape(-1),ignore_index=-100,reduction='none').view(lb.shape).sum(1)
def score(pv,ii,am,qt,lb,chunk=24):
    s=torch.empty(lb.size(0),device=DEV)
    with torch.no_grad():
        for st in range(0,lb.size(0),chunk):
            x=lb[st:st+chunk]; k=x.size(0)
            o=m(pixel_values=pv.expand(k,-1,-1,-1),input_ids=ii.expand(k,-1),attention_mask=am.expand(k,-1),labels=x,question_types=qt.expand(k))
            s[st:st+k]=logp(o.answer_logits,x)
    return s.cpu().numpy()
rows=[]
for c,j in enumerate(sel):
    b=next(iter(DataLoader(Subset(ds,[j]),batch_size=1)))
    pv=b['pixel_values'].to(DEV); ii=b['input_ids'].to(DEV); am=b['attention_mask'].to(DEV); qt=b['question_type'].to(DEV).long()
    t=int(te.type.iloc[j]); voc=TVOC[t]; gi=voc.index(gold.iloc[j]); L=LB[t]
    m.vision_gating.alpha_override=None
    s=score(pv,ii,am,qt,L); P=m.vision_gating.last_alpha.shape[1]
    rec=dict(idx=j,type=t,hit_base=int(s.argmax()==gi),alpha_base=float(m.vision_gating.last_alpha.float().mean()))
    for av in AL:
        m.vision_gating.alpha_override=torch.full((1,P),av,device=DEV)
        sx=score(pv,ii,am,qt,L); rec[f'hit_{av}']=int(sx.argmax()==gi)
    m.vision_gating.alpha_override=None
    rows.append(rec)
    if (c+1)%50==0:
        pd.DataFrame(rows).to_csv(f'analysis/mech/sharpen_{a.tag}.csv',index=False)
        print(f'  {c+1}/{len(sel)}',flush=True)
d=pd.DataFrame(rows); d.to_csv(f'analysis/mech/sharpen_{a.tag}.csv',index=False)
print(f'\nsaved analysis/mech/sharpen_{a.tag}.csv\n')
bb=d.hit_base.values
print(f'  base (model own alpha, mean {d.alpha_base.mean():.3f})   EM = {100*bb.mean():.2f}\n')
print(f'{"alpha":>7}{"regime":>16}{"EM":>9}{"rescued":>9}{"broken":>8}{"net":>7}')
for av in AL:
    h=d[f'hit_{av}'].values
    reg='contract' if av<1 else ('identity-ish' if av==1.0 else 'SHARPEN')
    print(f'{av:>7}{reg:>16}{100*h.mean():>9.2f}{int(((bb==0)&(h==1)).sum()):>9}{int(((bb==1)&(h==0)).sum()):>8}{int(((bb==0)&(h==1)).sum())-int(((bb==1)&(h==0)).sum()):>7}')
print('\nper type, best sharpening level vs base:')
for t in [0,1,2,3]:
    s=d[d.type==t]
    best=max(AL,key=lambda av:s[f'hit_{av}'].mean())
    print(f'  {T[t]:<9} base={100*s.hit_base.mean():6.2f}  best alpha={best:<4} EM={100*s[f"hit_{best}"].mean():6.2f}  ({100*(s[f"hit_{best}"].mean()-s.hit_base.mean()):+.2f})')
