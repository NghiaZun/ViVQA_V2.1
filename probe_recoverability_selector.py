"""Can a LABEL-FREE confidence signal identify the RECOVERABLE error subset?

probe_reach_law.py established (with labels) that oracle alpha rescues errors whose gold sits at
rank <= 3 and almost none beyond. A deployable selector cannot see the gold, so the question this
script answers is the deployable half:

    among the samples the model gets WRONG, does a label-free score separate the ones the oracle
    can rescue from the ones it cannot?

Method: re-score the SAME 320 samples probe_reach_law.py used, over the SAME per-type train
vocabulary, and record only label-free quantities of the BASE model (no alpha fitting, so this is
cheap). Join on idx with the reach CSV, which supplies hit_base/hit_fit/rank_base.

Label-free features recorded per sample:
  margin   = logp(top1) - logp(top2)            over the type vocabulary
  top1     = logp(top1)
  entropy  = entropy of softmax over the vocabulary
  alpha_base = the model's own gate alpha (mean over patches)

Reported: AUROC of each feature for (a) predicting the model is WRONG, and (b) among wrong
samples, predicting the oracle RESCUES it. (b) is the number that decides whether an error-aware
selector is buildable; (a) is the already-known control.
"""
import sys, time, argparse, unicodedata as ud
import torch, torch.nn.functional as F, numpy as np, pandas as pd
sys.path.insert(0, 'src')
from model import DeterministicVQA
from dataset import VQAGenDataset
from torch.utils.data import DataLoader, Subset
from transformers import AutoProcessor

p = argparse.ArgumentParser()
p.add_argument('--checkpoint', required=True)
p.add_argument('--reach_csv', required=True, help='output of probe_reach_law.py; supplies idx + hit_base/hit_fit')
p.add_argument('--train_csv', default=None)
p.add_argument('--out', required=True)
a = p.parse_args()

DEV = 'cuda'
norm = lambda s: ud.normalize('NFC', str(s)).strip().lower()
T = {0: 'OBJECT', 1: 'COUNT', 2: 'COLOR', 3: 'LOCATION'}

ck = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
sa = ck['args']; sa = sa if isinstance(sa, dict) else vars(sa)
sd = ck['model_state_dict']; K = list(sd.keys())
tlr = next((sd[k].shape[0] for k in K
            if k.startswith('encoder.base_model.model.layers.0.self_attn.q_proj.lora_A')), 16)
m = DeterministicVQA(
    vision_model_name=sa.get('vision_model'), bartpho_model_name='vinai/bartpho-syllable',
    num_fusion_layers=sa.get('num_fusion_layers', 2), fusion_type=sa.get('fusion_type', 'text2vision'),
    use_text_lora=True, text_lora_r=tlr, text_lora_alpha=sa.get('text_lora_alpha', 32),
    use_decoder_lora=sa.get('use_decoder_lora', False), decoder_lora_r=sa.get('decoder_lora_r', 16),
    decoder_lora_alpha=sa.get('decoder_lora_alpha', 32), use_vision_gate=True,
    vision_gate_init=sa.get('vision_gate_init', 1.5),
    vision_gate_min_alpha=sa.get('vision_gate_min_alpha', 0.0),
    vision_gate_max_alpha=sa.get('vision_gate_max_alpha', 1.0),
    use_type_task=any(k.startswith('type_head.') or k.startswith('type_classifier.') for k in K),
    use_siglip_pooler=sa.get('use_siglip_pooler', True)).to(DEV).eval()
r = m.load_state_dict(sd, strict=False)
assert not [k for k in r.missing_keys if 'teacher' not in k], 'thieu key'
for q in m.parameters(): q.requires_grad_(False)
tok = m.tokenizer

tr = pd.read_csv(a.train_csv or sa.get('train_csv', 'archive/train_split_original.csv'))
tr['an'] = tr.answer.map(norm)
TVOC = {t: sorted(set(g.an)) for t, g in tr.groupby('type')}
LB = {}
for t, voc in TVOC.items():
    e = tok(voc, return_tensors='pt', padding='max_length', truncation=True, max_length=10)
    x = e.input_ids.to(DEV).clone(); x[x == tok.pad_token_id] = -100
    LB[t] = x

te = pd.read_csv('archive/test.csv'); gold = te.answer.map(norm)
vp = AutoProcessor.from_pretrained(sa.get('vision_model'))
ds = VQAGenDataset(csv_path='archive/test.csv', image_folder='archive/data/images/test',
                   vision_processor=vp, tokenizer_name='vinai/bartpho-syllable', max_q_len=32,
                   max_a_len=10, include_question_type=True, auto_detect_type=False)

reach = pd.read_csv(a.reach_csv)
sel = reach.idx.tolist()

def logp(lg, lb):
    return -F.cross_entropy(lg.reshape(-1, lg.size(-1)).float(), lb.reshape(-1),
                            ignore_index=-100, reduction='none').view(lb.shape).sum(1)

def score_all(pv, ii, am, qt, lb, chunk=256):
    s = torch.empty(lb.size(0), device=DEV)
    with torch.no_grad():
        for st in range(0, lb.size(0), chunk):
            x = lb[st:st + chunk]; k = x.size(0)
            o = m(pixel_values=pv.expand(k, -1, -1, -1), input_ids=ii.expand(k, -1),
                  attention_mask=am.expand(k, -1), labels=x, question_types=qt.expand(k))
            s[st:st + k] = logp(o.answer_logits, x)
    return s

rows, t0 = [], time.time()
for c, j in enumerate(sel):
    b = next(iter(DataLoader(Subset(ds, [j]), batch_size=1)))
    pv = b['pixel_values'].to(DEV); ii = b['input_ids'].to(DEV); am = b['attention_mask'].to(DEV)
    qt = b['question_type'].to(DEV).long()
    t = int(te.type.iloc[j])
    m.vision_gating.alpha_override = None
    s = score_all(pv, ii, am, qt, LB[t])
    a0 = m.vision_gating.last_alpha.detach().float()
    if a0.dim() == 3: a0 = a0.squeeze(-1)
    sn = s.cpu().numpy()
    o = np.argsort(-sn)
    pr = torch.softmax(s.float(), 0).cpu().numpy()
    rows.append(dict(idx=j, type=t,
                     margin=float(sn[o[0]] - sn[o[1]]),
                     top1=float(sn[o[0]]),
                     entropy=float(-(pr * np.log(pr + 1e-12)).sum()),
                     alpha_base=float(a0.mean())))
    if (c + 1) % 80 == 0:
        el = time.time() - t0
        print(f'  {c+1}/{len(sel)} ({el/60:.1f}p)', flush=True)

d = pd.DataFrame(rows).merge(reach[['idx', 'hit_base', 'hit_fit', 'rank_base', 'nvoc']], on='idx')
d.to_csv(a.out, index=False)
print(f'\nsaved {a.out} ({len(d)} rows)\n')

from sklearn.metrics import roc_auc_score
FEATS = ['margin', 'top1', 'entropy', 'alpha_base']
print('(a) CONTROL — predict the model is WRONG (higher = more likely wrong):')
y = 1 - d.hit_base.values
for f in FEATS:
    print(f'    AUROC({f:<10}) = {roc_auc_score(y, -d[f].values):.3f}')
e = d[d.hit_base == 0]
print(f'\n(b) THE DECIDING NUMBER — among the {len(e)} WRONG samples, predict oracle RESCUE:')
if e.hit_fit.nunique() > 1:
    for f in FEATS:
        auc = roc_auc_score(e.hit_fit.values, e[f].values)
        print(f'    AUROC({f:<10}) = {auc:.3f}   (mirrored {1-auc:.3f})')
    print(f'    AUROC(rank_base, LABEL-DEPENDENT upper reference) = {roc_auc_score(e.hit_fit.values, -e.rank_base.values):.3f}')
else:
    print('    degenerate: no variation in hit_fit')
