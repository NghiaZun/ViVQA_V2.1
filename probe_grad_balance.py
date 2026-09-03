"""QUICK EVAL: gradient cua decoder co ap dao duong thi giac khong?

Gia dinh dang kiem: decoder full fine-tune (~285M, 99.6% tham so duoc train) hut het gradient,
nen cau noi thi giac (Flamingo + TCVG, ~5-30M) hoc rat it -> TCVG bi bo doi tin hieu hoc.

Dai luong doc duoc la CAP NHAT TUONG DOI ||g|| / ||theta||, khong phai ||g|| tho:
Adam chuan hoa theo tung tham so nen ||g|| tho khong so sanh duoc giua module khac kich thuoc.
Neu ti le nay xap xi nhau giua cac module -> gia dinh SAI, khong chay sweep LR.
"""
import sys, torch, pandas as pd, torch.nn.functional as F
sys.path.insert(0,'src')
from probe_evidence_readout import build_model
from dataset import VQAGenDataset
from torch.utils.data import DataLoader
from transformers import AutoProcessor

DEV='cuda'
import os
if os.environ.get('FRESH'):
    from model import DeterministicVQA
    sa={'vision_model':'google/siglip-base-patch16-224'}
    m=DeterministicVQA(vision_model_name=sa['vision_model'],bartpho_model_name='vinai/bartpho-syllable',
      num_fusion_layers=2,fusion_type='text2vision',use_text_lora=True,text_lora_r=16,text_lora_alpha=32,
      use_vision_gate=True,vision_gate_init=1.0,vision_gate_min_alpha=0.0,use_type_task=True,
      use_siglip_pooler=True).to(DEV)
else:
    m,sa=build_model('checkpoints_run87/best_model.pt',DEV)
for p in m.parameters(): p.requires_grad_(True)
m.train()
vp=AutoProcessor.from_pretrained(sa.get('vision_model'))
ds=VQAGenDataset(csv_path='archive/train_split.csv',image_folder='archive/data/images/train',
  vision_processor=vp,tokenizer_name='vinai/bartpho-syllable',max_q_len=32,max_a_len=10,
  include_question_type=True,auto_detect_type=False)
dl=DataLoader(ds,batch_size=12,shuffle=True,num_workers=2)

GROUPS={'decoder':'decoder.','lm_head':'lm_head.','flamingo(GCA)':'flamingo_fusion.',
        'TCVG(gate)':'vision_gating.','type_head':'type_head.','text LoRA':'lora_',
        'vision_proj':'vision_projection.'}
acc={k:0.0 for k in GROUPS}; pn={k:0.0 for k in GROUPS}; cnt={k:0 for k in GROUPS}
NB=8
for i,b in enumerate(dl):
    if i>=NB: break
    m.zero_grad(set_to_none=True)
    o=m(pixel_values=b['pixel_values'].to(DEV),input_ids=b['input_ids'].to(DEV),
        attention_mask=b['attention_mask'].to(DEV),labels=b['labels'].to(DEV),
        question_types=b['question_type'].to(DEV).long())
    loss=F.cross_entropy(o.answer_logits.reshape(-1,o.answer_logits.size(-1)).float(),
                         b['labels'].to(DEV).reshape(-1),ignore_index=-100)
    loss.backward()
    for n,p in m.named_parameters():
        if p.grad is None: continue
        for k,pat in GROUPS.items():
            if pat in n:
                acc[k]+=float(p.grad.detach().float().norm()**2)
                if i==0:
                    pn[k]+=float(p.detach().float().norm()**2); cnt[k]+=p.numel()
                break
print(f'\nloss cuoi = {float(loss):.4f}   ({NB} batch x 12)\n')
print(f'{"module":<16} {"#params":>12} {"||g||":>10} {"||theta||":>11} {"||g||/||theta||":>16}')
rows=[]
for k in GROUPS:
    if cnt[k]==0: continue
    g=(acc[k]/NB)**.5; t=pn[k]**.5; rel=g/max(t,1e-9)
    rows.append((k,cnt[k],g,t,rel))
    print(f'{k:<16} {cnt[k]:>12,} {g:>10.4f} {t:>11.2f} {rel:>16.6f}')
base=[r for r in rows if r[0]=='TCVG(gate)']
if base and rows:
    b=base[0][4]
    print(f'\nchuan hoa theo TCVG (=1.0):')
    for k,c,g,t,rel in sorted(rows,key=lambda r:-r[4]):
        print(f'  {k:<16} {rel/b:>8.2f}x')
    print('\nDOC: ti le ~1x nghia la cac module hoc voi toc do tuong doi NHU NHAU')
    print('     -> gia dinh "decoder hut het gradient" SAI, khong nen sweep decoder_lr.')
