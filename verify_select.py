"""CHOT: gate_mode=multiply co THUC SU doi hanh vi, va co tao ra chon loc THEO PATCH khong?
Repo nay tung co 5 co bi ho thanh no-op im lang. Khong tin co nao chua qua chot."""
import sys, torch, numpy as np
sys.path.insert(0,'src')
from model import DeterministicVQA
from dataset import VQAGenDataset
from torch.utils.data import DataLoader, Subset
from transformers import AutoProcessor
DEV='cuda'; VM='google/siglip-base-patch16-224'
def build(mode, ag):
    torch.manual_seed(0)
    return DeterministicVQA(vision_model_name=VM,bartpho_model_name='vinai/bartpho-syllable',
      num_fusion_layers=2,fusion_type='text2vision',use_text_lora=True,text_lora_r=16,
      text_lora_alpha=32,use_vision_gate=True,vision_gate_init=1.0,vision_gate_min_alpha=0.0,
      use_type_task=True,use_siglip_pooler=True,
      tcvg_gate_mode=mode, tcvg_attn_gate=ag).to(DEV).eval()
vp=AutoProcessor.from_pretrained(VM)
ds=VQAGenDataset(csv_path='archive/train_split.csv',image_folder='archive/data/images/train',
  vision_processor=vp,tokenizer_name='vinai/bartpho-syllable',max_q_len=32,max_a_len=10,
  include_question_type=True,auto_detect_type=False)
b=next(iter(DataLoader(Subset(ds,list(range(8))),batch_size=8)))
kw=dict(pixel_values=b['pixel_values'].to(DEV),input_ids=b['input_ids'].to(DEV),
        attention_mask=b['attention_mask'].to(DEV),labels=b['labels'].to(DEV))
qt=b['question_type'].to(DEV).long()

print(f'{"cau hinh":<26} {"||gated||tb":>12} {"alpha tb":>10} {"alpha std TRONG-MAU":>21}')
out={}
for name,mode,ag in [('blend (hien tai)','blend',False),
                     ('multiply','multiply',False),
                     ('multiply + attn_gate','multiply',True)]:
    m=build(mode,ag); cap={}
    m.vision_gating.register_forward_hook(lambda mo,i,o: cap.__setitem__('g',o[0].detach()))
    with torch.no_grad(): o=m(**kw,question_types=qt)
    g=cap['g']; al=m.vision_gating.last_alpha.detach().float()
    if al.dim()==3: al=al.squeeze(-1)
    out[name]=(g.float().cpu(), o.answer_logits.float().cpu())
    print(f'{name:<26} {g.float().norm(dim=-1).mean():>12.4f} {al.mean():>10.4f} '
          f'{al.std(dim=1).mean():>21.6f}')
    del m; torch.cuda.empty_cache()

print('\n--- 1. multiply co doi hanh vi so voi blend khong? ---')
d=(out['multiply'][1]-out['blend (hien tai)'][1]).abs().max()
print(f'  |logits multiply - blend| toi da = {d:.4f}  -> ' +
      ('KHAC (co chay)' if d>1e-3 else '!! GIONG HET — co la NO-OP'))
print('\n--- 2. multiply co giu duoc KENH BIEN DO khong? (blend bi LN xoa: 0.0004) ---')
for k in ('blend (hien tai)','multiply','multiply + attn_gate'):
    n=out[k][0].norm(dim=-1)
    print(f'  {k:<26} do tan norm giua cac patch = {(n.std(dim=1)/n.mean(dim=1)).mean():.6f}')
print('\n--- 3. attn_gate co tao bien thien alpha THEO PATCH khong? ---')
print('  (xem cot "alpha std TRONG-MAU" o tren: blend/multiply ~0 = khong chon loc;')
print('   attn_gate > 0 ro ret = diem so phu thuoc tung patch)')
