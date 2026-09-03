"""MECHANISM PROBE — WHEN / WHERE / HOW for visual intervention.

Runs three analyses in one pass because they share the expensive part (closed-set rescoring
over the per-type train vocabulary). Reuses probe_reach_law.py's model construction, vocabulary
and oracle-alpha fitting verbatim; adds three measurements on top.

  [HOW-1  constant-alpha grid]  For every sample, rescore under alpha forced to each of a few
      CONSTANTS. If low-margin errors are rescued by a fixed constant, no per-sample alpha
      prediction is needed at all and the method is deployable today. Correct samples are scored
      at the same constants, so the BREAKAGE cost is measured on the same pass -- a rescue rate
      without a breakage rate is meaningless (97% of TCVG's rewrites cancel).

  [HOW-2  oracle fit + predictability]  Per-sample oracle per-patch alpha (25 Adam steps on the
      gold, identical to probe_reach_law.py). Dumps the fitted alpha so alpha-predictability can
      be tested RESTRICTED TO THE LOW-MARGIN SUBSET -- the seven earlier probes all fit on the
      whole error distribution, where ~76% of errors are unrescuable and act as label noise.

  [WHERE  patch counterfactual]  For each rescued sample, keep the fitted alpha on only the top-k
      patches by |alpha_fit - alpha_base| and revert every other patch to the model's own alpha
      (via the NaN convention of VisionGating.alpha_override, model.py:1263). Compare against k
      RANDOM patches, matched k. If top-k survives at small k and random-k does not, the
      intervention is spatially localised and patch-level routing is the right object.

  [WHAT  candidate transition]  Records the top-1 candidate before and after the fit, so it is
      possible to say WHICH competitor the gold overtakes.

Nothing here is a method: the oracle uses gold labels at inference. The constant-alpha arm is the
only arm that is label-free, and that is exactly why it is the one that matters.
"""
import sys, time, argparse, json, unicodedata as ud
import torch, torch.nn.functional as F, numpy as np, pandas as pd
sys.path.insert(0, 'src')
from model import DeterministicVQA
from dataset import VQAGenDataset
from torch.utils.data import DataLoader, Subset
from transformers import AutoProcessor

p = argparse.ArgumentParser()
p.add_argument('--checkpoint', required=True)
p.add_argument('--train_csv', default=None)
p.add_argument('--per_type', type=int, default=150)
p.add_argument('--steps', type=int, default=25)
p.add_argument('--lr', type=float, default=0.1)
p.add_argument('--chunk', type=int, default=64)
p.add_argument('--consts', default='0.0,0.25,0.5,0.75')
p.add_argument('--topk', default='1,5,10,25,50')
p.add_argument('--tag', required=True)
p.add_argument('--outdir', default='analysis/mech')
a = p.parse_args()

DEV = 'cuda'
norm = lambda s: ud.normalize('NFC', str(s)).strip().lower()
T = {0: 'OBJECT', 1: 'COUNT', 2: 'COLOR', 3: 'LOCATION'}
CONSTS = [float(x) for x in a.consts.split(',')]
TOPKS = [int(x) for x in a.topk.split(',')]
import os; os.makedirs(a.outdir, exist_ok=True)

ck = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
sa = ck['args']; sa = sa if isinstance(sa, dict) else vars(sa)
sd = ck['model_state_dict']; K = list(sd.keys())
tlr = next((sd[k].shape[0] for k in K
            if k.startswith('encoder.base_model.model.layers.0.self_attn.q_proj.lora_A')), 16)
assert float(sa.get('vision_gate_min_alpha', 0.0)) == 0.0 and float(sa.get('vision_gate_max_alpha', 1.0)) == 1.0, \
    'alpha_override is pre-scaling; this probe assumes the identity scaling [0,1]'
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
assert not [k for k in r.missing_keys if 'teacher' not in k], 'missing keys'
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
print({T[t]: len(v) for t, v in TVOC.items()}, flush=True)

te = pd.read_csv('archive/test.csv'); gold = te.answer.map(norm)
vp = AutoProcessor.from_pretrained(sa.get('vision_model'))
ds = VQAGenDataset(csv_path='archive/test.csv', image_folder='archive/data/images/test',
                   vision_processor=vp, tokenizer_name='vinai/bartpho-syllable', max_q_len=32,
                   max_a_len=10, include_question_type=True, auto_detect_type=False)
rng = np.random.default_rng(0)
sel = []
for t in [0, 1, 2, 3]:
    pool = [i for i in range(len(te)) if int(te.type.iloc[i]) == t and gold.iloc[i] in TVOC[t]]
    sel += rng.choice(pool, min(a.per_type, len(pool)), replace=False).tolist()
sel = sorted(sel)
print(f'{len(sel)} samples', flush=True)

def logp(lg, lb):
    return -F.cross_entropy(lg.reshape(-1, lg.size(-1)).float(), lb.reshape(-1),
                            ignore_index=-100, reduction='none').view(lb.shape).sum(1)

def score_all(pv, ii, am, qt, lb):
    s = torch.empty(lb.size(0), device=DEV)
    with torch.no_grad():
        for st in range(0, lb.size(0), a.chunk):
            x = lb[st:st + a.chunk]; k = x.size(0)
            o = m(pixel_values=pv.expand(k, -1, -1, -1), input_ids=ii.expand(k, -1),
                  attention_mask=am.expand(k, -1), labels=x, question_types=qt.expand(k))
            s[st:st + k] = logp(o.answer_logits, x)
    return s.cpu().numpy()

def rank_of(sc, gi):
    return int((sc > sc[gi]).sum()) + 1

rows, alphas, t0 = [], {}, time.time()
for c, j in enumerate(sel):
    b = next(iter(DataLoader(Subset(ds, [j]), batch_size=1)))
    pv = b['pixel_values'].to(DEV); ii = b['input_ids'].to(DEV); am = b['attention_mask'].to(DEV)
    qt = b['question_type'].to(DEV).long()
    t = int(te.type.iloc[j]); voc = TVOC[t]; gi = voc.index(gold.iloc[j]); L = LB[t]

    # ---- base ----
    m.vision_gating.alpha_override = None
    sb = score_all(pv, ii, am, qt, L)
    a0 = m.vision_gating.last_alpha.detach().float()
    if a0.dim() == 3: a0 = a0.squeeze(-1)
    a0 = a0[:1]                                   # [1, P]
    P = a0.size(1)
    o = np.argsort(-sb)
    rb = rank_of(sb, gi); hb = int(o[0] == gi)
    rec = dict(idx=j, type=t, nvoc=len(voc), P=P, rank_base=rb, hit_base=hb,
               margin=float(sb[o[0]] - sb[o[1]]), alpha_base=float(a0.mean()),
               top1_base=voc[o[0]], top2_base=voc[o[1]], gold=voc[gi])

    # ---- HOW-1: constant alpha, on EVERY sample (rescue AND breakage) ----
    for cv in CONSTS:
        m.vision_gating.alpha_override = torch.full((1, P), cv, device=DEV)
        sc = score_all(pv, ii, am, qt, L)
        rec[f'hit_c{cv}'] = int(sc.argmax() == gi); rec[f'rank_c{cv}'] = rank_of(sc, gi)
    m.vision_gating.alpha_override = None

    # ---- HOW-2 + WHERE: only on errors (nothing to rescue otherwise) ----
    if hb == 0:
        gl = L[gi:gi + 1]
        th = torch.logit(a0.clamp(1e-4, 1 - 1e-4)).clone().requires_grad_(True)
        opt = torch.optim.Adam([th], lr=a.lr)
        with torch.enable_grad():
            for _ in range(a.steps):
                opt.zero_grad(set_to_none=True)
                m.vision_gating.alpha_override = torch.sigmoid(th)
                out = m(pixel_values=pv, input_ids=ii, attention_mask=am, labels=gl, question_types=qt)
                (-logp(out.answer_logits, gl).sum()).backward(); opt.step()
        af = torch.sigmoid(th).detach()                    # [1, P]
        m.vision_gating.alpha_override = af
        sf = score_all(pv, ii, am, qt, L)
        m.vision_gating.alpha_override = None
        of = np.argsort(-sf)
        rec['rank_fit'] = rank_of(sf, gi); rec['hit_fit'] = int(of[0] == gi)
        rec['alpha_fit'] = float(af.mean()); rec['top1_fit'] = voc[of[0]]
        dev = (af - a0).abs()[0]                            # [P]
        rec['dev_mean'] = float(dev.mean()); rec['dev_max'] = float(dev.max())
        # concentration: share of total deviation held by the top 10% of patches
        sd_, _ = torch.sort(dev, descending=True)
        rec['dev_top10pct_share'] = float(sd_[:max(1, P // 10)].sum() / sd_.sum().clamp(min=1e-9))
        alphas[j] = dict(a0=a0[0].cpu().numpy().tolist(), af=af[0].cpu().numpy().tolist())

        if rec['hit_fit'] == 1:                             # WHERE: only where a rescue exists
            order = torch.argsort(dev, descending=True)
            for k in TOPKS:
                if k > P: continue
                for mode in ['top', 'rand']:
                    keep = order[:k] if mode == 'top' else torch.tensor(
                        rng.choice(P, k, replace=False), device=DEV)
                    ov = torch.full((1, P), float('nan'), device=DEV)
                    ov[0, keep] = af[0, keep]               # NaN elsewhere = keep model's own alpha
                    m.vision_gating.alpha_override = ov
                    sk = score_all(pv, ii, am, qt, L)
                    rec[f'hit_{mode}{k}'] = int(sk.argmax() == gi)
                    rec[f'rank_{mode}{k}'] = rank_of(sk, gi)
            m.vision_gating.alpha_override = None
    rows.append(rec)
    if (c + 1) % 50 == 0:
        el = time.time() - t0
        pd.DataFrame(rows).to_csv(f'{a.outdir}/mech_{a.tag}.csv', index=False)
        print(f'  {c+1}/{len(sel)} ({el/60:.1f}m, eta {el/(c+1)*(len(sel)-c-1)/60:.0f}m)', flush=True)

d = pd.DataFrame(rows)
d.to_csv(f'{a.outdir}/mech_{a.tag}.csv', index=False)
np.savez_compressed(f'{a.outdir}/alpha_{a.tag}.npz', **{str(k): np.array(v['af']) for k, v in alphas.items()},
                    **{f'base_{k}': np.array(v['a0']) for k, v in alphas.items()})
print(f'\nsaved {a.outdir}/mech_{a.tag}.csv ({len(d)} rows) and alpha_{a.tag}.npz\n')
