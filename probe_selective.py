"""HUONG 1 — DU DOAN CO CHON LOC: model co BIET luc nao no khong chac khong?

Su that da do: 98.8% loi la mot dap an HOP LE khac, gold nam o hang trung vi 3, 74.5% loi COUNT
lech dung 1. Model that su khong chac, va cai khong chac do la CO THAT.
Chua ai hoi: do khong-chac do co DO DUOC khong.

Neu do duoc thi dung duoc duong risk-coverage: "dat X% chinh xac tren Y% so cau, tu choi phan con
lai". Do la ket qua DUONG ma khong can EM nhuc nhich, khong doi kien truc, khong train lai.

Do BON thang do do tin cay, tren TOAN BO 3001 mau:
  margin  = logP(ung vien 1) - logP(ung vien 2)      <- ky vong manh nhat
  top1    = logP(ung vien 1)
  marginN / top1N = ban chia do dai
So sanh bang AUROC tach ca DUNG khoi ca SAI, roi ve duong risk-coverage.

NHUONG GPU: phaseB3 dang train. Cho den khi con cho moi chay, va dung batch nho.
"""
import sys, os, time, torch, pandas as pd, numpy as np, unicodedata as ud
import torch.nn.functional as F
sys.path.insert(0, 'src')
from model import DeterministicVQA
from dataset import VQAGenDataset
from torch.utils.data import DataLoader, Subset
from transformers import AutoProcessor

CKPT = 'checkpoints_run87/best_model.pt'          # SigLIP1, 72.34
DEV = 'cuda'; CHUNK = 32; NEED_MIB = 11000
norm = lambda s: ud.normalize('NFC', str(s)).strip().lower()


def free_mib():
    import subprocess
    o = subprocess.run(['nvidia-smi', '--query-gpu=memory.total,memory.used',
                        '--format=csv,noheader,nounits'], capture_output=True, text=True).stdout
    t, u = [int(x) for x in o.strip().split('\n')[0].split(', ')]
    return t - u


while free_mib() < NEED_MIB:
    print(f'GPU con {free_mib()}MiB, doi 5 phut', flush=True)
    time.sleep(300)

tr = pd.read_csv('archive/train_split_original.csv')
CAND = sorted({norm(a) for a in tr.answer})
print(f'{len(CAND)} ung vien', flush=True)

ck = torch.load(CKPT, map_location='cpu', weights_only=False); sa = ck.get('args', {})
model = DeterministicVQA(
    vision_model_name=sa.get('vision_model', 'google/siglip-base-patch16-224'),
    bartpho_model_name='vinai/bartpho-syllable',
    num_fusion_layers=sa.get('num_fusion_layers', 2), fusion_type='text2vision',
    use_text_lora=True, text_lora_r=16, text_lora_alpha=32,
    use_vision_gate=True, vision_gate_init=sa.get('vision_gate_init', 1.0),
    vision_gate_min_alpha=0.0, use_type_task=sa.get('use_type_loss', True),
    use_siglip_pooler=sa.get('use_siglip_pooler', True)).to(DEV).eval()
model.load_state_dict(ck['model_state_dict'], strict=False)
for p in model.parameters(): p.requires_grad_(False)
tok = model.tokenizer

enc = tok(CAND, return_tensors='pt', padding='max_length', truncation=True, max_length=10)
LB = enc.input_ids.to(DEV).clone(); LB[LB == tok.pad_token_id] = -100
NTOK = (LB != -100).sum(1).float()

te = pd.read_csv('archive/test.csv')
vp = AutoProcessor.from_pretrained(sa.get('vision_model', 'google/siglip-base-patch16-224'))
ds = VQAGenDataset(csv_path='archive/test.csv', image_folder='archive/data/images/test',
                   vision_processor=vp, tokenizer_name='vinai/bartpho-syllable',
                   max_q_len=32, max_a_len=10, include_question_type=True, auto_detect_type=False)
base = pd.read_csv('analysis/advisor_s1/base.csv')      # de lay nhan dung/sai cua he da trien khai

rows = []
t0 = time.time()
for j in range(len(ds)):
    b = next(iter(DataLoader(Subset(ds, [j]), batch_size=1)))
    pv = b['pixel_values'].to(DEV); ii = b['input_ids'].to(DEV); am = b['attention_mask'].to(DEV)
    qt = b['question_type'].to(DEV).long() if b.get('question_type') is not None else None
    sums = torch.empty(len(CAND), device=DEV)
    with torch.no_grad():
        for s in range(0, len(CAND), CHUNK):
            lb = LB[s:s + CHUNK]; k = lb.size(0)
            out = model(pixel_values=pv.expand(k, -1, -1, -1),
                        input_ids=ii.expand(k, -1), attention_mask=am.expand(k, -1),
                        labels=lb, question_types=None if qt is None else qt.expand(k))
            lg = out.answer_logits.float()
            lp = -F.cross_entropy(lg.reshape(-1, lg.size(-1)), lb.reshape(-1),
                                  ignore_index=-100, reduction='none').view(lb.shape)
            sums[s:s + k] = lp.sum(1)
    sN = sums / NTOK
    o = torch.argsort(sums, descending=True); oN = torch.argsort(sN, descending=True)
    g = norm(te.answer.iloc[j])
    rows.append(dict(
        idx=j, type=te.type.iloc[j],
        pick=CAND[int(o[0])], pickN=CAND[int(oN[0])], gold=g,
        hit=int(CAND[int(o[0])] == g), hitN=int(CAND[int(oN[0])] == g),
        deployed_hit=int(base.exact_match.iloc[j] > .5),
        margin=float(sums[o[0]] - sums[o[1]]), top1=float(sums[o[0]]),
        marginN=float(sN[oN[0]] - sN[oN[1]]), top1N=float(sN[oN[0]])))
    if (j + 1) % 250 == 0:
        el = time.time() - t0
        print(f'  {j+1}/{len(ds)}  ({el/60:.1f} phut, con ~{el/(j+1)*(len(ds)-j-1)/60:.0f} phut)', flush=True)

d = pd.DataFrame(rows); d.to_csv('analysis/selective.csv', index=False)
print(f'\nda luu analysis/selective.csv ({len(d)} dong)')
print(f'  do chinh xac khi chon bang tong log-lik      : {100*d.hit.mean():.2f}')
print(f'  do chinh xac khi chia do dai                 : {100*d.hitN.mean():.2f}')
print(f'  do chinh xac cua he DA TRIEN KHAI (beam+trie): {100*d.deployed_hit.mean():.2f}')
