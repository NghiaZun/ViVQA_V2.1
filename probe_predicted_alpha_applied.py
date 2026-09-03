"""FAIR TEST of per-patch routing: does a PREDICTED alpha field actually RESCUE?

Regressing onto the oracle's alpha and getting R2 < 0 is not by itself proof that per-patch
routing is impossible: the oracle's field is one member of a possibly large set of fields that
would rescue the same sample, and a regression to one arbitrary member can fail even when the set
is easy to hit. The decisive test is behavioural, not geometric:

    fit alpha-predictor on OTHER samples -> apply its prediction to this sample -> did EM improve?

Predictor input is exactly `gate_net`'s own input [v_proj_i ; q], so a positive result would mean
the CURRENT TCVG architecture can express the fix and only lacks the training signal (-> alpha
distillation is the fix). A negative result closes per-patch routing on the current gate input.

Held out by SAMPLE (GroupKFold), never by patch.
Controls: (1) the model's own alpha = no-op; (2) a shuffled prediction from another sample.
"""
import sys, argparse, unicodedata as ud
import torch, torch.nn.functional as F, numpy as np, pandas as pd
sys.path.insert(0, 'src')
from model import DeterministicVQA
from dataset import VQAGenDataset
from torch.utils.data import DataLoader, Subset
from transformers import AutoProcessor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

p = argparse.ArgumentParser()
p.add_argument('--checkpoint', required=True)
p.add_argument('--gateinput_npz', required=True)
p.add_argument('--mech_csv', required=True)
p.add_argument('--train_csv', default=None)
p.add_argument('--ridge', type=float, default=100.0)
p.add_argument('--out', required=True)
a = p.parse_args()
DEV = 'cuda'
norm = lambda s: ud.normalize('NFC', str(s)).strip().lower()
T = {0: 'OBJECT', 1: 'COUNT', 2: 'COLOR', 3: 'LOCATION'}

Z = np.load(a.gateinput_npz)
X, Y, S, TY, RS = Z['X'], Z['Y'], Z['sample'], Z['type'], Z['rescued']
N, P, D = X.shape
# ---- cross-fitted per-patch alpha prediction, grouped by sample ----
Xf = X.reshape(-1, D).astype(np.float32); Yf = Y.reshape(-1).astype(np.float32)
g = np.repeat(np.arange(N), P)
pred = np.zeros_like(Yf)
for tr, te in GroupKFold(n_splits=5).split(Xf, Yf, groups=g):
    pred[te] = Ridge(alpha=a.ridge).fit(Xf[tr], Yf[tr]).predict(Xf[te])
PRED = pred.reshape(N, P)
print(f'predicted delta: mean={PRED.mean():+.4f} sd={PRED.std():.4f} | oracle delta sd={Y.std():.4f}', flush=True)

ck = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
sa = ck['args']; sa = sa if isinstance(sa, dict) else vars(sa)
sd = ck['model_state_dict']; K = list(sd.keys())
tlr = next((sd[k].shape[0] for k in K
            if k.startswith('encoder.base_model.model.layers.0.self_attn.q_proj.lora_A')), 16)
m = DeterministicVQA(
    vision_model_name=sa.get('vision_model'), bartpho_model_name='vinai/bartpho-syllable',
    num_fusion_layers=sa.get('num_fusion_layers', 2), fusion_type=sa.get('fusion_type', 'text2vision'),
    use_text_lora=True, text_lora_r=tlr, text_lora_alpha=sa.get('text_lora_alpha', 32),
    use_vision_gate=True, vision_gate_init=sa.get('vision_gate_init', 1.5),
    vision_gate_min_alpha=sa.get('vision_gate_min_alpha', 0.0),
    vision_gate_max_alpha=sa.get('vision_gate_max_alpha', 1.0),
    use_type_task=any(k.startswith('type_head.') or k.startswith('type_classifier.') for k in K),
    use_siglip_pooler=sa.get('use_siglip_pooler', True)).to(DEV).eval()
r = m.load_state_dict(sd, strict=False)
assert not [k for k in r.missing_keys if 'teacher' not in k]
for q in m.parameters(): q.requires_grad_(False)
tok = m.tokenizer

tr_ = pd.read_csv(a.train_csv or sa.get('train_csv', 'archive/train_split_original.csv'))
tr_['an'] = tr_.answer.map(norm)
TVOC = {t: sorted(set(gg.an)) for t, gg in tr_.groupby('type')}
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

def logp(lg, lb):
    return -F.cross_entropy(lg.reshape(-1, lg.size(-1)).float(), lb.reshape(-1),
                            ignore_index=-100, reduction='none').view(lb.shape).sum(1)
def score_all(pv, ii, am, qt, lb, chunk=64):
    s = torch.empty(lb.size(0), device=DEV)
    with torch.no_grad():
        for st in range(0, lb.size(0), chunk):
            x = lb[st:st + chunk]; k = x.size(0)
            o = m(pixel_values=pv.expand(k, -1, -1, -1), input_ids=ii.expand(k, -1),
                  attention_mask=am.expand(k, -1), labels=x, question_types=qt.expand(k))
            s[st:st + k] = logp(o.answer_logits, x)
    return s.cpu().numpy()

rng = np.random.default_rng(0)
perm = rng.permutation(N)
rows = []
for n in range(N):
    j = int(S[n]); t = int(TY[n]); voc = TVOC[t]; gi = voc.index(gold.iloc[j]); L = LB[t]
    b = next(iter(DataLoader(Subset(ds, [j]), batch_size=1)))
    pv = b['pixel_values'].to(DEV); ii = b['input_ids'].to(DEV); am = b['attention_mask'].to(DEV)
    qt = b['question_type'].to(DEV).long()
    m.vision_gating.alpha_override = None
    sb = score_all(pv, ii, am, qt, L)
    a0 = m.vision_gating.last_alpha.detach().float()
    if a0.dim() == 3: a0 = a0.squeeze(-1)
    a0 = a0[:1]
    rec = dict(idx=j, type=t, hit_base=int(sb.argmax() == gi), rescuable=int(RS[n]))
    for nm, dl in [('pred', PRED[n]), ('shuf', PRED[perm[n]])]:
        av = (a0 + torch.tensor(dl, device=DEV, dtype=a0.dtype).unsqueeze(0)).clamp(0, 1)
        m.vision_gating.alpha_override = av
        s = score_all(pv, ii, am, qt, L)
        rec[f'hit_{nm}'] = int(s.argmax() == gi)
    m.vision_gating.alpha_override = None
    rows.append(rec)
    if (n + 1) % 50 == 0: print(f'  {n+1}/{N}', flush=True)

d = pd.DataFrame(rows); d.to_csv(a.out, index=False)
print(f'\nsaved {a.out}\n')
print(f'{"arm":<34}{"EM on this error set":>22}{"rescued":>10}{"broken":>9}')
b0 = d.hit_base.values
for nm in ['pred', 'shuf']:
    h = d[f'hit_{nm}'].values
    print(f'{("cross-fitted PREDICTED alpha" if nm=="pred" else "SHUFFLED prediction (null)"):<34}'
          f'{100*h.mean():>21.2f}%{int(((b0==0)&(h==1)).sum()):>10}{int(((b0==1)&(h==0)).sum()):>9}')
print(f'{"base (model own alpha)":<34}{100*b0.mean():>21.2f}%{"-":>10}{"-":>9}')
sub = d[d.rescuable == 1]
print(f'\nRestricted to the {len(sub)} samples the ORACLE could rescue:')
print(f'  predicted alpha rescues {int(sub.hit_pred.sum())}/{len(sub)}   shuffled rescues {int(sub.hit_shuf.sum())}/{len(sub)}')
