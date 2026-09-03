"""PHA A1 — GOLD DUNG HANG MAY, VA CACH TOP-1 BAO XA?

Quyet dinh pha B co ton tai hay khong. Neu gold thuong nam hang 2-3 voi khoang cach nho thi mot
ham margin (day gold vuot len tren ung vien sai ma model tu chon) la dung don bay. Neu gold nam
hang > 5 hoac cach xa thi margin khong cuu duoc gi va pha B bi bo.

TIEU CHI GIET (ghi TRUOC khi chay):
  - trung vi hang cua gold tren cac mau SAI <= 3   -> pha B song
  - trung vi hang > 5                              -> pha B chet

GIOI HAN PHAI NOI RO: cho nay cham bang LIKELIHOOD ep buoc (teacher forcing), con model khi
trien khai sinh bang beam 3 + trie. Hai dai luong KHAC NHAU — bai hoc tu phep thu steer 2026-08-15,
luc do toi da dung mot phep do sinh de ket luan cho mot phep do likelihood. O day chi ket luan ve
likelihood, va do chinh la dai luong ma ham mat mat tac dong vao, nen dung cho muc dich pha B.
"""
import sys, torch, pandas as pd, numpy as np, unicodedata as ud, json
import torch.nn.functional as F
sys.path.insert(0, 'src')
from model import DeterministicVQA
from dataset import VQAGenDataset
from torch.utils.data import DataLoader, Subset
from transformers import AutoProcessor

CKPT = 'checkpoints_run87/best_model.pt'      # SigLIP1, base 72.34
DEV = 'cuda'; CHUNK = 48
norm = lambda s: ud.normalize('NFC', str(s)).strip().lower()

tr = pd.read_csv('archive/train_split_original.csv')
CAND = sorted({norm(a) for a in tr.answer})
print(f'{len(CAND)} ung vien = toan bo tu vung dap an cua train', flush=True)

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

te = pd.read_csv('archive/test.csv')
base = pd.read_csv('beam3fixed/seed42_ep40.csv')
ok = base.exact_match.values > .5
vp = AutoProcessor.from_pretrained(sa.get('vision_model', 'google/siglip-base-patch16-224'))
ds = VQAGenDataset(csv_path='archive/test.csv', image_folder='archive/data/images/test',
                   vision_processor=vp, tokenizer_name='vinai/bartpho-syllable',
                   max_q_len=32, max_a_len=10, include_question_type=True, auto_detect_type=False)

# nhan cho MOI ung vien, dung mot lan
enc = tok(CAND, return_tensors='pt', padding='max_length', truncation=True, max_length=10)
LB = enc.input_ids.to(DEV).clone()
LB[LB == tok.pad_token_id] = -100
NTOK = (LB != -100).sum(1).float()                       # so token thuc cua moi ung vien

# lay CA mau sai VA mot mau doi chieu cac mau dung
rng = np.random.RandomState(0)
idx_wrong = np.where(~ok)[0]
idx_right = rng.choice(np.where(ok)[0], size=400, replace=False)
IDX = np.concatenate([idx_wrong, idx_right])
print(f'cham {len(idx_wrong)} mau SAI + {len(idx_right)} mau DUNG (doi chieu)', flush=True)

rows = []
for n, j in enumerate(IDX):
    b = next(iter(DataLoader(Subset(ds, [int(j)]), batch_size=1)))
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
            sums[s:s + k] = lp.sum(1)                    # log-likelihood ca chuoi
    g = norm(te.answer.iloc[int(j)])
    if g not in CAND:                                    # gold ngoai tu vung train -> khong xep hang duoc
        continue
    gi = CAND.index(g)
    for mode, sc in (('sum', sums), ('norm', sums / NTOK)):
        order = torch.argsort(sc, descending=True)
        rank = int((order == gi).nonzero()[0]) + 1
        top1 = int(order[0])
        rows.append(dict(idx=int(j), correct=bool(ok[j]), mode=mode, rank=rank,
                         gold=g, top1=CAND[top1],
                         margin=float(sc[top1] - sc[gi]),
                         type=te.type.iloc[int(j)]))
    if (n + 1) % 100 == 0:
        d = pd.DataFrame(rows); d = d[(d['mode'] == 'sum') & (~d.correct)]
        print(f'  {n+1}/{len(IDX)}  trung vi hang gold tren mau SAI = {d["rank"].median():.0f}', flush=True)

df = pd.DataFrame(rows)
df.to_csv('analysis/gold_rank.csv', index=False)
TN = {0: 'OBJECT', 1: 'COUNT', 2: 'COLOR', 3: 'LOCATION'}
for mode in ('sum', 'norm'):
    d = df[df['mode'] == mode]
    print(f'\n===== cham bang {"tong log-likelihood" if mode=="sum" else "log-likelihood chia do dai"}')
    for nm, m in (('mau SAI', ~d.correct), ('mau DUNG (doi chieu)', d.correct)):
        s = d[m]
        if not len(s): continue
        print(f'  {nm}: n={len(s)} | hang gold: trung vi {s["rank"].median():.0f} '
              f'| hang 1 {100*(s["rank"]==1).mean():.1f}% | <=3 {100*(s["rank"]<=3).mean():.1f}% '
              f'| <=5 {100*(s["rank"]<=5).mean():.1f}% | khoang cach toi top1 trung vi {s["margin"].median():.3f}')
    s = d[~d.correct]
    print('  tren mau SAI, tach theo loai:')
    for t, nm in TN.items():
        x = s[s['type'] == t]
        if not len(x): continue
        print(f'     {nm:9s} n={len(x):4d} | trung vi hang {x["rank"].median():4.0f} '
              f'| <=3 {100*(x["rank"]<=3).mean():5.1f}% | khoang cach {x["margin"].median():.3f}')
print('\nTIEU CHI GIET: trung vi hang tren mau SAI <= 3 -> pha B song; > 5 -> pha B chet')
