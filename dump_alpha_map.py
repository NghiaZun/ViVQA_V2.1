"""Dump ban do alpha 14x14 cua MODEL cho tung anh cu the.

Cau hoi cua tac gia: y tuong ban dau la voi anh con chim, alpha ~1 o vung con chim va ~0 o nen,
tao mot heat map lam con chim NOI BAT cho decoder. Kiem xem thuc te co phai vay khong.

In ra: alpha trung binh, do lech chuan giua cac patch, va luoi 14x14 de nhin truc tiep.
"""
import sys, torch, numpy as np, pandas as pd
sys.path.insert(0, 'src')
from model import DeterministicVQA
from dataset import VQAGenDataset
from torch.utils.data import DataLoader, Subset

CKPT = 'checkpoints_s2_T2/last_model.pt'; VM = 'google/siglip2-base-patch16-224'; DEV = 'cuda'
IMGS = [149577, 361778]     # con chim do (COLOR), gau bong (LOCATION)

ck = torch.load(CKPT, map_location='cpu', weights_only=False); sa = ck.get('args', {})
model = DeterministicVQA(
    vision_model_name=VM, bartpho_model_name='vinai/bartpho-syllable',
    num_fusion_layers=sa.get('num_fusion_layers', 2), fusion_type=sa.get('fusion_type', 'text2vision'),
    num_heads=8, dropout=0.1, gradient_checkpointing=False,
    use_text_lora=sa.get('use_text_lora', True), text_lora_r=sa.get('text_lora_r', 16),
    text_lora_alpha=sa.get('text_lora_alpha', 32),
    use_vision_gate=True, vision_gate_init=sa.get('vision_gate_init', 1.0),
    vision_gate_min_alpha=sa.get('vision_gate_min_alpha', 0.0),
    use_type_task=sa.get('use_type_loss', True),
    use_siglip_pooler=sa.get('use_siglip_pooler', True),
).to(DEV).eval()
m, u = model.load_state_dict(ck['model_state_dict'], strict=False)
print(f'load: thieu {len(m)}, thua {len(u)}')

te = pd.read_csv('archive/test.csv')
from transformers import AutoProcessor
vp = AutoProcessor.from_pretrained(VM)
ds = VQAGenDataset(csv_path='archive/test.csv', image_folder='archive/data/images/test',
                   vision_processor=vp, tokenizer_name='vinai/bartpho-syllable',
                   max_q_len=32, max_a_len=10, include_question_type=True, auto_detect_type=False)
for img in IMGS:
    idx = te.index[te.img_id == img].tolist()
    for j in idx:
        b = next(iter(DataLoader(Subset(ds, [j]), batch_size=1)))
        with torch.no_grad():
            model(pixel_values=b['pixel_values'].to(DEV), input_ids=b['input_ids'].to(DEV),
                  attention_mask=b['attention_mask'].to(DEV), labels=b['labels'].to(DEV))
        a = model.vision_gating.last_alpha[0].float().cpu().numpy()
        a = a[-196:]                      # bo pooler token o dau neu co
        g = a.reshape(14, 14)
        print(f'\n=== img {img} | {te.question[j]} | gold={te.answer[j]}')
        print(f'    alpha: mean={a.mean():.3f}  std={a.std():.4f}  min={a.min():.3f}  max={a.max():.3f}')
        print(f'    khoang [max-min] = {a.max()-a.min():.4f}')
        print('    ban do 14x14 (x100):')
        for r in g:
            print('      ' + ' '.join(f'{v*100:3.0f}' for v in r))
        np.save(f'analysis/alphamap_{img}.npy', g)
print('\n=> y tuong ban dau: vung vat the ~100, nen ~0 (khoang gan 1.0)')
print('   thuc te: doc dong "khoang [max-min]" o tren')
