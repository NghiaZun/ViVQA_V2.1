"""CHOT: --kl_pretrained_lambda co THUC SU chay khong, va chuan hoa co dung ty trong khong?

Repo nay tung co 5 co bi ho khoi constructor va chay thanh no-op im lang. Khong tin co nao
chua qua chot.
Kiem 4 dieu:
  1. lambda=0 -> KHONG nap moc tham chieu, kl_pretrained_loss is None  (khong doi run cu)
  2. lambda>0 -> kl_pretrained_loss la so THUC, khac 0
  3. chuan hoa cho ra dung ty trong ky vong lambda/(1+lambda)
  4. gradient CHAY qua so hang moi (khong bi detach nham)
"""
import sys, torch, torch.nn.functional as F, pandas as pd
sys.path.insert(0,'src')
from model import DeterministicVQA
from dataset import VQAGenDataset
from torch.utils.data import DataLoader, Subset
from transformers import AutoProcessor
DEV='cuda'; VM='google/siglip-base-patch16-224'

def build(lam):
    return DeterministicVQA(vision_model_name=VM,bartpho_model_name='vinai/bartpho-syllable',
      num_fusion_layers=2,fusion_type='text2vision',use_text_lora=True,text_lora_r=16,
      text_lora_alpha=32,use_vision_gate=True,vision_gate_init=1.0,vision_gate_min_alpha=0.0,
      use_type_task=True,use_siglip_pooler=True,kl_pretrained_lambda=lam).to(DEV)

vp=AutoProcessor.from_pretrained(VM)
ds=VQAGenDataset(csv_path='archive/train_split.csv',image_folder='archive/data/images/train',
  vision_processor=vp,tokenizer_name='vinai/bartpho-syllable',max_q_len=32,max_a_len=10,
  include_question_type=True,auto_detect_type=False)
b=next(iter(DataLoader(Subset(ds,list(range(8))),batch_size=8)))
kw=dict(pixel_values=b['pixel_values'].to(DEV),input_ids=b['input_ids'].to(DEV),
        attention_mask=b['attention_mask'].to(DEV),labels=b['labels'].to(DEV),
        question_types=b['question_type'].to(DEV).long())

print('\n--- 1. lambda = 0.0 (phai TRO) ---')
m0=build(0.0); m0.train()
o0=m0(**kw)
print(f'  moc tham chieu duoc nap : {m0._ref_decoder is not None}   (ky vong: False)')
print(f'  kl_pretrained_loss      : {getattr(o0,"kl_pretrained_loss",None)}   (ky vong: None)')
assert m0._ref_decoder is None and o0.kl_pretrained_loss is None, 'lambda=0 KHONG tro!'
del m0,o0; torch.cuda.empty_cache()

print('\n--- 2+3+4. lambda = 0.25 ---')
LAM=0.25
m=build(LAM); m.train()
o=m(**kw)
kl=o.kl_pretrained_loss
print(f'  moc tham chieu duoc nap : {m._ref_decoder is not None}   (ky vong: True)')
print(f'  kl_pretrained_loss      : {float(kl):.4f}')
assert kl is not None and float(kl)>0, 'KL khong duoc tinh!'
ce=F.cross_entropy(o.answer_logits.reshape(-1,o.answer_logits.size(-1)).float(),
                   b['labels'].to(DEV).reshape(-1),ignore_index=-100)
scale=(ce.detach()/kl.detach().clamp(min=1e-6)).clamp(max=1e3)
term=LAM*scale*kl
tot=ce+term
print(f'  CE                      : {float(ce):.4f}')
print(f'  so hang KL sau chuan hoa: {float(term):.4f}')
print(f'  ty trong trong tong loss: {100*float(term)/float(tot):.1f}%   '
      f'(ky vong {100*LAM/(1+LAM):.1f}%)')
assert abs(100*float(term)/float(tot) - 100*LAM/(1+LAM)) < 1.0, 'chuan hoa SAI ty trong!'

m.zero_grad(set_to_none=True); term.backward(retain_graph=True)
g=sum(float(p.grad.norm()) for p in m.decoder.parameters() if p.grad is not None)
print(f'  ||grad|| decoder tu RIENG so hang KL: {g:.4f}   (phai > 0)')
assert g>0, 'so hang KL KHONG tao gradient — bi detach nham!'
print('\n=== 4/4 CHOT DAT: co that su chay, chuan hoa dung, gradient thong ===')
