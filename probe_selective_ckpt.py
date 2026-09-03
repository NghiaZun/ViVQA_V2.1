"""AURC/AUROC CO CHON LOC — SO GATE (T2) VOI KHONG GATE (T0) TREN CUNG SEED.

Vi sao huong nay: do duoc hom nay (analysis/ttsel) rang gate DICH CHUYEN do tin cay rat manh
(log-prob doi 0.5-1.1 nat giua cac muc alpha) nhung KHONG dich duoc argmax -> EM net zero.
Mot co che chi dinh hinh lai confidence thi phai do bang metric confidence, khong phai EM.

Gia thuyet: T2 xep hang DUNG/SAI tot hon T0 o cung EM -> duong risk-coverage thap hon.
Do bang E-AURC (AURC tru AURC toi uu cua chinh do chinh xac do) de TACH chat luong xep hang
khoi chenh lech do chinh xac — hai nhanh khong bat buoc cung EM.

Chay tren VAL (khong phai test) — test chi dung sau khi val da chot.
"""
import sys, os, time, argparse, unicodedata as ud
import torch, torch.nn.functional as F, pandas as pd, numpy as np
sys.path.insert(0, 'src')
from model import DeterministicVQA
from dataset import VQAGenDataset
from torch.utils.data import DataLoader
from transformers import AutoProcessor

p = argparse.ArgumentParser()
p.add_argument('--checkpoint', required=True)
p.add_argument('--out', required=True)
p.add_argument('--csv', default='archive/val_split.csv')
p.add_argument('--image_folder', default='archive/data/images/train')
p.add_argument('--chunk', type=int, default=256)
a = p.parse_args()

DEV = 'cuda'
norm = lambda s: ud.normalize('NFC', str(s)).strip().lower()

ck = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
sa = ck.get('args', {})
sa = sa if isinstance(sa, dict) else vars(sa)
sd = ck['model_state_dict']
K = list(sd.keys())

has_vision_gate = any(k.startswith('vision_gating.') for k in K)
has_text_lora = any(k.startswith('encoder.base_model.model.layers.0.self_attn.q_proj.lora_A') for k in K)
has_vision_lora = any('vision_encoder' in k and 'lora_A' in k for k in K)
has_type_task = any(k.startswith('type_head.') or k.startswith('type_classifier.') for k in K)
text_lora_r = next((sd[k].shape[0] for k in K
                    if k.startswith('encoder.base_model.model.layers.0.self_attn.q_proj.lora_A')), 16)

model = DeterministicVQA(
    vision_model_name=sa.get('vision_model', 'google/siglip-base-patch16-224'),
    bartpho_model_name='vinai/bartpho-syllable',
    num_fusion_layers=sa.get('num_fusion_layers', 2),
    fusion_type=sa.get('fusion_type', 'text2vision'),
    use_text_lora=has_text_lora, text_lora_r=text_lora_r,
    text_lora_alpha=sa.get('text_lora_alpha', 32),
    use_vision_lora=has_vision_lora, vision_lora_r=sa.get('vision_lora_r', 8),
    use_decoder_lora=sa.get('use_decoder_lora', False),
    decoder_lora_r=sa.get('decoder_lora_r', 16),
    decoder_lora_alpha=sa.get('decoder_lora_alpha', 32),
    use_vision_gate=has_vision_gate,
    vision_gate_init=sa.get('vision_gate_init', 1.5),
    vision_gate_min_alpha=sa.get('vision_gate_min_alpha', 0.0),
    vision_gate_max_alpha=sa.get('vision_gate_max_alpha', 1.0),
    use_type_task=has_type_task,
    use_siglip_pooler=sa.get('use_siglip_pooler', True),
    use_mean_pool_cls=sa.get('use_mean_pool_cls', False),
    use_attn_pool_cls=sa.get('use_attn_pool_cls', False),
).to(DEV).eval()

res = model.load_state_dict(sd, strict=False)
# Head CHI dung luc train, khong nam trong duong tinh diem dap an -> vut la dung.
# Moi thu khac bi vut, va BAT KY key thieu nao (= khoi tao ngau nhien), deu lam ket qua sai.
AUX = ('teacher', 'contrastive_head', 'qgnd_ids')
unexp = [k for k in res.unexpected_keys if not any(t in k for t in AUX)]
miss = [k for k in res.missing_keys if 'teacher' not in k]
if unexp or miss:
    print(f'!! KIEN TRUC KHONG KHOP: {len(unexp)} vut {unexp[:6]} | {len(miss)} ngau nhien {miss[:6]}')
    sys.exit(2)
print(f'bo qua {len(res.unexpected_keys) - len(unexp)} head phu (chi dung luc train)')
print(f'khop 100% | gate={has_vision_gate} type_task={has_type_task} '
      f'vm={sa.get("vision_model")} seed={sa.get("seed")}', flush=True)
for q in model.parameters():
    q.requires_grad_(False)
tok = model.tokenizer

TRAIN = sa.get('train_csv', 'archive/train_split_original.csv')
CAND = sorted({norm(x) for x in pd.read_csv(TRAIN).answer})
print(f'{len(CAND)} ung vien tu {TRAIN}', flush=True)
enc = tok(CAND, return_tensors='pt', padding='max_length', truncation=True, max_length=10)
LB = enc.input_ids.to(DEV).clone()
LB[LB == tok.pad_token_id] = -100
NTOK = (LB != -100).sum(1).float()

te = pd.read_csv(a.csv)
vp = AutoProcessor.from_pretrained(sa.get('vision_model', 'google/siglip-base-patch16-224'))
ds = VQAGenDataset(csv_path=a.csv, image_folder=a.image_folder, vision_processor=vp,
                   tokenizer_name='vinai/bartpho-syllable', max_q_len=32, max_a_len=10,
                   include_question_type=True, auto_detect_type=False)
loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)

rows, t0 = [], time.time()
for j, b in enumerate(loader):
    pv = b['pixel_values'].to(DEV); ii = b['input_ids'].to(DEV); am = b['attention_mask'].to(DEV)
    qt = b['question_type'].to(DEV).long() if b.get('question_type') is not None else None
    sums = torch.empty(len(CAND), device=DEV)
    with torch.no_grad():
        for s in range(0, len(CAND), a.chunk):
            lb = LB[s:s + a.chunk]; k = lb.size(0)
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
    rows.append(dict(idx=j, type=te.type.iloc[j], pick=CAND[int(o[0])], pickN=CAND[int(oN[0])],
                     gold=g, hit=int(CAND[int(o[0])] == g), hitN=int(CAND[int(oN[0])] == g),
                     margin=float(sums[o[0]] - sums[o[1]]), top1=float(sums[o[0]]),
                     marginN=float(sN[oN[0]] - sN[oN[1]]), top1N=float(sN[oN[0]])))
    if (j + 1) % 200 == 0:
        el = time.time() - t0
        print(f'  {j+1}/{len(ds)} ({el/60:.1f}p, con ~{el/(j+1)*(len(ds)-j-1)/60:.0f}p)', flush=True)

d = pd.DataFrame(rows)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
d.to_csv(a.out, index=False)


def aurc(conf, hit):
    """AURC va E-AURC. Risk = ty le sai tren phan da chap nhan, quet theo do tin cay giam dan."""
    o = np.argsort(-np.asarray(conf, float))
    h = np.asarray(hit, float)[o]
    cov = np.arange(1, len(h) + 1)
    risk = 1.0 - np.cumsum(h) / cov
    A = float(risk.mean())
    acc = h.mean(); r = 1 - acc
    # AURC toi uu: moi cau dung xep truoc moi cau sai
    ho = np.sort(h)[::-1]
    Ao = float((1.0 - np.cumsum(ho) / cov).mean())
    return A, A - Ao, r


def auroc(conf, hit):
    c = np.asarray(conf, float); h = np.asarray(hit, int)
    if h.sum() == 0 or h.sum() == len(h):
        return float('nan')
    r = pd.Series(c).rank().values
    n1 = h.sum(); n0 = len(h) - n1
    return float((r[h == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


print(f'\nda luu {a.out} ({len(d)} dong)')
print(f'{"do do":<10}{"acc":>8}{"AUROC":>9}{"AURC":>9}{"E-AURC":>9}')
for name, conf, hit in [('margin', d.margin, d.hit), ('top1', d.top1, d.hit),
                        ('marginN', d.marginN, d.hitN), ('top1N', d.top1N, d.hitN)]:
    A, E, r = aurc(conf, hit)
    print(f'{name:<10}{1-r:>8.4f}{auroc(conf, hit):>9.4f}{A:>9.4f}{E:>9.4f}')
