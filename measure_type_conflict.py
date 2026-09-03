"""Do XUNG DOT GRADIENT giua cac loai cau hoi — kiem chung "cac type keo co nhau".

Bang chung gian tiep da co (moi can thiep deu la phep CHIA LAI, khong phai phep cong):
    TCVG 10 seed : COLOR +0.77  LOCATION +0.67 | COUNT -0.61  OBJECT -0.10
    bgc          : COUNT +1.35                 | COLOR  -3.52
    harm2.0      : OBJECT +0.80                | LOCATION -0.73  COLOR -0.48
    boxv2 s42    : OBJECT +0.96                | COLOR  -0.96
Nhung do la o muc KET QUA. Goc re phai o muc GRADIENT.

Cach do: batch THUAN MOT LOAI -> gradient cua answer loss theo tham so dung chung -> cosine
giua cac loai. cosine < 0 nghia la buoc cap nhat giup loai nay se HAI loai kia. Do la dinh nghia
co hoc cua "keo co", va no giai thich vi sao 40+ co che deu net zero: chung khong the vuot qua
mot xung dot nam o muc ham muc tieu.

Neu xung dot la that thi phuong thuoc khong phai them co che, ma la doi CACH GOP GRADIENT
(gradient surgery kieu PCGrad: chieu bo thanh phan xung dot truoc khi cong).
"""
import sys, os, torch, numpy as np, pandas as pd
sys.path.insert(0, 'src')
from torch.utils.data import DataLoader, Subset
from model import DeterministicVQA
from dataset import VQAGenDataset

CKPT = 'checkpoints_s2_T2/last_model.pt'
VM = 'google/siglip2-base-patch16-224'
DEV = 'cuda'
T = {0: 'OBJECT', 1: 'COUNT', 2: 'COLOR', 3: 'LOCATION'}
N_PER_TYPE = 120

ck = torch.load(CKPT, map_location='cpu', weights_only=False)
sa = ck.get('args', {})
print('dung model tu', CKPT)

model = DeterministicVQA(
    vision_model_name=VM, bartpho_model_name='vinai/bartpho-syllable',
    num_fusion_layers=sa.get('num_fusion_layers', 2), fusion_type=sa.get('fusion_type', 'text2vision'),
    num_heads=8, dropout=0.1, gradient_checkpointing=False,
    use_text_lora=sa.get('use_text_lora', True), text_lora_r=sa.get('text_lora_r', 16),
    text_lora_alpha=sa.get('text_lora_alpha', 32), text_lora_dropout=0.1,
    use_vision_gate=sa.get('use_vision_gate', True),
    vision_gate_init=sa.get('vision_gate_init', 1.0),
    vision_gate_min_alpha=sa.get('vision_gate_min_alpha', 0.0),
    use_type_task=sa.get('use_type_loss', True),
    use_siglip_pooler=sa.get('use_siglip_pooler', True),
).to(DEV)
missing, unexpected = model.load_state_dict(ck['model_state_dict'], strict=False)
print(f'  thieu {len(missing)} key, thua {len(unexpected)} key')
model.train()

from transformers import AutoProcessor
vp = AutoProcessor.from_pretrained(VM)
ds = VQAGenDataset(csv_path='archive/train_split_original.csv',
                   image_folder='archive/data/images/train',
                   vision_processor=vp, tokenizer_name='vinai/bartpho-syllable',
                   max_q_len=32, max_a_len=10, include_question_type=True,
                   auto_detect_type=False)
df = pd.read_csv('archive/train_split_original.csv')

shared = [(n, p) for n, p in model.named_parameters()
          if p.requires_grad and 'vision_encoder' not in n]
print(f'  {len(shared)} nhom tham so dung chung')

def grad_of_type(t):
    """gradient trung binh cua answer loss tren batch THUAN loai t"""
    idx = df.index[df.type == t].tolist()[:N_PER_TYPE]
    dl = DataLoader(Subset(ds, idx), batch_size=12, shuffle=False, num_workers=2)
    model.zero_grad(set_to_none=True)
    nb = 0
    for b in dl:
        out = model(pixel_values=b['pixel_values'].to(DEV),
                    input_ids=b['input_ids'].to(DEV),
                    attention_mask=b['attention_mask'].to(DEV),
                    labels=b['labels'].to(DEV))
        (out.answer_loss / (N_PER_TYPE // 12)).backward()
        nb += 1
    g = torch.cat([(p.grad.detach().flatten() if p.grad is not None
                    else torch.zeros(p.numel(), device=DEV)) for _, p in shared])
    model.zero_grad(set_to_none=True)
    return g

G = {}
for t, name in T.items():
    G[name] = grad_of_type(t)
    print(f'  {name:9s} ||grad|| = {G[name].norm().item():.4e}')

print('\n=== COSINE GIUA GRADIENT CUA CAC LOAI ===')
names = list(T.values())
print(f'{"":11s}' + ''.join(f'{n:>11s}' for n in names))
for a in names:
    row = f'{a:11s}'
    for b in names:
        c = torch.nn.functional.cosine_similarity(G[a], G[b], dim=0).item()
        row += f'{c:>11.4f}'
    print(row)

print('\ncosine < 0  = hai loai KEO NGUOC nhau (buoc giup loai nay se hai loai kia)')
print('cosine ~ 0  = truc giao, khong xung dot ma cung khong ho tro')
print('cosine > 0  = cung huong, cai thien mot loai keo theo loai kia')

# doi chung: chia NGAU NHIEN thay vi theo loai -> cosine cua "khong co xung dot"
rng = np.random.default_rng(0)
allidx = rng.permutation(df.index.values)[:N_PER_TYPE * 2]
def grad_of_idx(idx):
    dl = DataLoader(Subset(ds, list(idx)), batch_size=12, shuffle=False, num_workers=2)
    model.zero_grad(set_to_none=True)
    for b in dl:
        out = model(pixel_values=b['pixel_values'].to(DEV), input_ids=b['input_ids'].to(DEV),
                    attention_mask=b['attention_mask'].to(DEV), labels=b['labels'].to(DEV))
        (out.answer_loss / (len(idx) // 12)).backward()
    g = torch.cat([(p.grad.detach().flatten() if p.grad is not None
                    else torch.zeros(p.numel(), device=DEV)) for _, p in shared])
    model.zero_grad(set_to_none=True)
    return g
gA = grad_of_idx(allidx[:N_PER_TYPE]); gB = grad_of_idx(allidx[N_PER_TYPE:])
print(f'\nDOI CHUNG hai batch NGAU NHIEN (khong theo loai): cosine = '
      f'{torch.nn.functional.cosine_similarity(gA, gB, dim=0).item():.4f}')
print('  -> day la muc cosine khi KHONG co xung dot theo loai, chi co nhieu lay mau.')
print('  Neu cosine giua cac LOAI thap hon han con so nay thi xung dot theo loai la THAT.')
