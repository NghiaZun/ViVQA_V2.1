"""BUOC 3 — can thiep TRONG CUNG MOT MODEL: gate TAT (alpha=1) vs gate BAT (alpha cua model).

Khac buoc 2 o cho: T0 va T2 la HAI LAN TRAIN KHAC NHAU, nen RESCUE/DAMAGE giua chung lan voi
nhieu seed (do duoc: churn T0->T2 = 7.86% vs churn khi CHI doi seed = 8.10%). O day CUNG BO
TRONG SO, chi bat/tat toan tu, nen moi thay doi deu quy duoc cho TCVG.

alpha_override=1.0 -> v_hat = LN(v), khong tiem text = TRUOC CAN THIEP.
alpha=cua model     -> TCVG day du = SAU CAN THIEP.

Dac trung truoc-can-thiep lay tu chinh nhanh gate-TAT, nen khong con dung model thay the lam proxy.
Cham diem tap dong theo tu vung cua loai (giong moi probe khac trong repo).
"""
import sys, argparse, unicodedata as ud
import torch, torch.nn.functional as F, numpy as np, pandas as pd
sys.path.insert(0, 'src')
from probe_evidence_readout import build_model
norm = lambda s: ud.normalize('NFC', str(s)).strip().lower()

p = argparse.ArgumentParser()
p.add_argument('--checkpoint', required=True); p.add_argument('--out', required=True)
p.add_argument('--train_csv', default='archive/train_split_original.csv')
p.add_argument('--test_csv', default='archive/test.csv')
p.add_argument('--image_folder', default='archive/data/images/test')
p.add_argument('--chunk', type=int, default=48)
a = p.parse_args()

from dataset import VQAGenDataset
from torch.utils.data import DataLoader, Subset
from transformers import AutoProcessor
DEV = 'cuda'
m, sa = build_model(a.checkpoint, DEV); tok = m.tokenizer
tr = pd.read_csv(a.train_csv); tr['an'] = tr.answer.map(norm)
TVOC = {int(t): sorted(set(g.an)) for t, g in tr.groupby('type')}
LB = {}
for t, voc in TVOC.items():
    e = tok(voc, return_tensors='pt', padding='max_length', truncation=True, max_length=10)
    x = e.input_ids.to(DEV).clone(); x[x == tok.pad_token_id] = -100; LB[t] = x
te = pd.read_csv(a.test_csv); gold = te.answer.map(norm)
vp = AutoProcessor.from_pretrained(sa.get('vision_model'))
ds = VQAGenDataset(csv_path=a.test_csv, image_folder=a.image_folder, vision_processor=vp,
                   tokenizer_name='vinai/bartpho-syllable', max_q_len=32, max_a_len=10,
                   include_question_type=True, auto_detect_type=False)
def logp(lg, lb):
    return -F.cross_entropy(lg.reshape(-1, lg.size(-1)).float(), lb.reshape(-1),
                            ignore_index=-100, reduction='none').view(lb.shape).sum(1)

R = []
for j in range(len(te)):
    t = int(te.type.iloc[j])
    if gold.iloc[j] not in TVOC[t]:
        continue
    L = LB[t]; voc = TVOC[t]; gi = voc.index(gold.iloc[j])
    b = next(iter(DataLoader(Subset(ds, [j]), batch_size=1)))
    pv = b['pixel_values'].to(DEV); ii = b['input_ids'].to(DEV)
    am = b['attention_mask'].to(DEV); qt = b['question_type'].to(DEV).long()
    S = {}
    for arm in ('off', 'on'):
        s = torch.empty(L.size(0), device=DEV)
        with torch.no_grad():
            for st in range(0, L.size(0), a.chunk):
                x = L[st:st + a.chunk]; k = x.size(0)
                m.vision_gating.alpha_override = (torch.ones(k, 1, device=DEV)
                                                  if arm == 'off' else None)
                o = m(pixel_values=pv.expand(k, -1, -1, -1), input_ids=ii.expand(k, -1),
                      attention_mask=am.expand(k, -1), labels=x, question_types=qt.expand(k))
                s[st:st + k] = logp(o.answer_logits, x)
        S[arm] = s.float()
    m.vision_gating.alpha_override = None
    al = m.vision_gating.last_alpha.detach().float()
    if al.dim() == 3: al = al.squeeze(-1)

    d = {}
    for arm in ('off', 'on'):
        s = S[arm]; o_ = torch.argsort(s, descending=True)
        c1, c2 = int(o_[0]), int(o_[1])
        sn = torch.softmax(s, 0)
        d[f'hit_{arm}'] = int(c1 == gi)
        d[f'margin_{arm}'] = float(s[c1] - s[c2])
        d[f'ent_{arm}'] = float(-(sn * (sn + 1e-9).log()).sum())
        d[f'sgold_{arm}'] = float(s[gi]); d[f'stop1_{arm}'] = float(s[c1])
        d[f'rank_{arm}'] = int((o_ == gi).nonzero()[0, 0]) + 1
        d[f'top1_{arm}'] = voc[c1]
    R.append(dict(idx=j, type=t, nvoc=len(voc), gold=voc[gi],
                  alpha_mean=float(al[0].mean()), alpha_std=float(al[0].std()), **d))
    if len(R) % 200 == 0:
        x = pd.DataFrame(R)
        print(f'  {len(R)}  EM off {x.hit_off.mean():.4f} on {x.hit_on.mean():.4f}', flush=True)

df = pd.DataFrame(R)
df['dmargin'] = df.margin_on - df.margin_off
df['cell'] = np.where(df.hit_off.eq(1) & df.hit_on.eq(1), 'keep',
             np.where(df.hit_off.eq(0) & df.hit_on.eq(1), 'RESCUE',
             np.where(df.hit_off.eq(1) & df.hit_on.eq(0), 'DAMAGE', 'hard')))
df.to_csv(a.out, index=False)
print(f'\nluu {len(df)} -> {a.out}')
print(df.cell.value_counts())
print(f'EM gate-TAT {df.hit_off.mean()*100:.2f}  gate-BAT {df.hit_on.mean()*100:.2f}')
