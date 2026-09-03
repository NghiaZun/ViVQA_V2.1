"""WHAT IS THE SUCCESSFUL INTERVENTION ACTUALLY DOING?

Stops asking how to predict alpha. Asks instead what changes in the representation the decoder
reads when an intervention succeeds, so that a more fundamental controllable quantity than alpha
can be identified.

Standard TCVG path (model.py, ln_mode='post'):
    pre_i  = alpha_i * v_i + (1 - alpha_i) * t_bar          t_bar shared by ALL patches
    v_hat  = LayerNorm(pre)                                  <- what decoder cross-attention sees

Because t_bar is one shared vector, a per-patch alpha is a DIFFERENTIAL CONTRACTION of the patch
cloud toward a single common point. Patches with low alpha collapse toward t_bar and toward each
other; patches with high alpha keep their identity. So the quantity TCVG physically controls is
not "how much vision" but the SHAPE of the patch cloud -- how many mutually distinguishable visual
tokens the decoder is offered.

Captured per sample, for the model's own alpha and for the fitted oracle alpha:
  geometry of the post-LN cloud   mean pairwise cosine, participation-ratio effective rank,
                                  norm dispersion (the amplitude channel the decoder attends on),
                                  mean cosine to t_bar
  candidate scores                the FULL per-type vocabulary score vector, so that
                                  delta_s = s_fit - s_base lives in a space that is IDENTICAL
                                  across all samples of a type -> "is there a consistent direction
                                  in candidate score space?" becomes directly testable.
"""
import sys, argparse, unicodedata as ud
import torch, torch.nn.functional as F, numpy as np, pandas as pd
sys.path.insert(0, 'src')
from model import DeterministicVQA
from dataset import VQAGenDataset
from torch.utils.data import DataLoader, Subset
from transformers import AutoProcessor

p = argparse.ArgumentParser()
p.add_argument('--checkpoint', required=True)
p.add_argument('--mech_csv', required=True)
p.add_argument('--alpha_npz', required=True)
p.add_argument('--train_csv', default=None)
p.add_argument('--tag', required=True)
p.add_argument('--outdir', default='analysis/mech')
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
    use_vision_gate=True, vision_gate_init=sa.get('vision_gate_init', 1.5),
    vision_gate_min_alpha=sa.get('vision_gate_min_alpha', 0.0),
    vision_gate_max_alpha=sa.get('vision_gate_max_alpha', 1.0),
    use_type_task=any(k.startswith('type_head.') or k.startswith('type_classifier.') for k in K),
    use_siglip_pooler=sa.get('use_siglip_pooler', True)).to(DEV).eval()
r = m.load_state_dict(sd, strict=False)
assert not [k for k in r.missing_keys if 'teacher' not in k]
for q in m.parameters(): q.requires_grad_(False)
tok = m.tokenizer

CAP = {}
m.vision_gating.gate_net.register_forward_pre_hook(lambda mod, i: CAP.__setitem__('gin', i[0].detach()))
m.vision_gating.layer_norm.register_forward_pre_hook(lambda mod, i: CAP.__setitem__('pre', i[0].detach()))
m.vision_gating.layer_norm.register_forward_hook(lambda mod, i, o: CAP.__setitem__('post', o.detach()))

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

def geom(post, tbar):
    """Geometry of the post-LN patch cloud, i.e. exactly what decoder cross-attention reads."""
    Xc = post[0].float()                                   # [P, D]
    Xn = F.normalize(Xc, dim=-1)
    C = Xn @ Xn.T
    P = Xc.size(0)
    off = (C.sum() - C.diag().sum()) / (P * (P - 1))       # mean pairwise cosine
    s = torch.linalg.svdvals(Xc - Xc.mean(0, keepdim=True))
    er = (s.sum() ** 2 / (s ** 2).sum())                   # participation-ratio effective rank
    nn_ = Xc.norm(dim=-1)
    tb = F.normalize(tbar.float().view(1, -1), dim=-1)
    return dict(pair_cos=float(off), eff_rank=float(er),
                norm_mean=float(nn_.mean()), norm_disp=float(nn_.std() / nn_.mean().clamp(min=1e-9)),
                cos_tbar=float((Xn @ tb.T).mean()))

mech = pd.read_csv(a.mech_csv); err = mech[mech.hit_base == 0]
Z = np.load(a.alpha_npz)
rows, DS = [], {}
for n, row in enumerate(err.itertuples()):
    j = int(row.idx)
    if str(j) not in Z.files: continue
    af = torch.tensor(Z[str(j)], device=DEV).unsqueeze(0)
    t = int(row.type); voc = TVOC[t]; gi = voc.index(gold.iloc[j]); L = LB[t]
    b = next(iter(DataLoader(Subset(ds, [j]), batch_size=1)))
    pv = b['pixel_values'].to(DEV); ii = b['input_ids'].to(DEV); am = b['attention_mask'].to(DEV)
    qt = b['question_type'].to(DEV).long()
    rec = dict(idx=j, type=t, rescued=int(row.hit_fit), rank_base=int(row.rank_base))
    got = {}
    for nm, ov in [('base', None), ('fit', af)]:
        m.vision_gating.alpha_override = ov
        s = score_all(pv, ii, am, qt, L)
        al = m.vision_gating.last_alpha.detach().float()
        if al.dim() == 3: al = al.squeeze(-1)
        pre = CAP['pre']; v = CAP['gin'][:, :, :CAP['gin'].size(-1) // 2]
        # t_bar solved from the blend identity on the patch with the smallest alpha
        k = int(al[0].argmin()); ak = al[0, k].clamp(max=0.999)
        tbar = (pre[0, k] - ak * v[0, k]) / (1 - ak)
        g = geom(CAP['post'], tbar)
        for kk, vv in g.items(): rec[f'{kk}_{nm}'] = vv
        got[nm] = s
    m.vision_gating.alpha_override = None
    DS[f'{t}_{j}'] = np.stack([got['base'], got['fit']])
    rec['gold_i'] = gi
    rows.append(rec)
    if (n + 1) % 50 == 0: print(f'  {n+1}/{len(err)}', flush=True)

d = pd.DataFrame(rows); d.to_csv(f'{a.outdir}/geom_{a.tag}.csv', index=False)
np.savez_compressed(f'{a.outdir}/scores_{a.tag}.npz', **DS)
print(f'\nsaved {a.outdir}/geom_{a.tag}.csv ({len(d)}) and scores_{a.tag}.npz\n')
from scipy import stats as st
print(f'{"geometry of the post-LN patch cloud":<26}{"RESCUED base->fit":>26}{"NOT-rescued base->fit":>26}{"p(diff of deltas)":>19}')
for k in ['pair_cos', 'eff_rank', 'norm_disp', 'cos_tbar', 'norm_mean']:
    R = d[d.rescued == 1]; Nn = d[d.rescued == 0]
    dr = R[f'{k}_fit'] - R[f'{k}_base']; dn = Nn[f'{k}_fit'] - Nn[f'{k}_base']
    print(f'  {k:<24}{R[f"{k}_base"].mean():>11.4f}->{R[f"{k}_fit"].mean():<10.4f}'
          f'{Nn[f"{k}_base"].mean():>13.4f}->{Nn[f"{k}_fit"].mean():<10.4f}'
          f'{st.mannwhitneyu(dr, dn).pvalue:>19.3g}')
