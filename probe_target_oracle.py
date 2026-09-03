"""COUNTERFACTUAL: does patch-specific TEXT CONTENT beat patch-specific GATING STRENGTH?

Current TCVG :  v_hat_i = a_i * v_i + (1 - a_i) * t_bar      t_bar = ONE masked-mean text vector
Proposed     :  v_hat_i = a_i * v_i + (1 - a_i) * c_i        c_i   = patch-specific question target

FAIRNESS. A freely fitted c_i would have P x 1024 = 201,728 dof per sample against 197 for
per-patch alpha, and would "win" on capacity alone. So c_i is constrained to what the proposed
architecture can actually express: a convex combination of the question's OWN token vectors,

    c_i = sum_j softmax(W_i)_j * t_proj_j          W in R^{P x L_real}

which has P x L_real dof (~2-6k), and, crucially, is NESTED: at W = 0 the softmax is uniform over
the real tokens, so c_i collapses to exactly the masked mean t_bar and the model is bit-identical
to the current one. Step 0 of every arm is therefore the same model.
l
INJECTION. No model edit. Since
    a*v + (1-a)*c  =  [a*v + (1-a)*t_bar]  +  (1-a)*(c - t_bar)
a forward_pre_hook on `vision_gating.layer_norm` (whose input is exactly the pre-LN blend for
ln_mode='post') adds (1-a)*(c - t_bar). Verified nested: W=0 gives a zero offset.

ARMS (same samples, same steps, same scoring):
  A1 scalar alpha        1 dof/sample            fit alpha, target fixed at t_bar
  A2 per-patch alpha     P dof/sample            fit alpha, target fixed at t_bar
  A3 per-patch target    P x L_real dof/sample   fit c_i, ALPHA FROZEN at the model's own value
  A4 null                shuffled W from another sample of the same type  (capacity control)

A3 freezes alpha so the comparison isolates TARGET CONTENT from GATING STRENGTH.
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
p.add_argument('--train_csv', default=None)
p.add_argument('--per_type', type=int, default=150)
p.add_argument('--steps', type=int, default=25)
p.add_argument('--lr_alpha', type=float, default=0.1)
p.add_argument('--lr_target', type=float, default=0.1)
p.add_argument('--lr_target2', type=float, default=0.5)
p.add_argument('--chunk', type=int, default=24)
p.add_argument('--limit', type=int, default=0)
p.add_argument('--tag', required=True)
a = p.parse_args()
DEV = 'cuda'
norm = lambda s: ud.normalize('NFC', str(s)).strip().lower()
T = {0: 'OBJECT', 1: 'COUNT', 2: 'COLOR', 3: 'LOCATION'}

ck = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
sa = ck['args']; sa = sa if isinstance(sa, dict) else vars(sa); sd = ck['model_state_dict']; K = list(sd.keys())
assert sa.get('tcvg_ln_mode', 'post') in ('post', None), 'this injection assumes ln_mode=post'
assert not sa.get('gate_layerscale_pertype', False), 'layerscale would rescale the blend'
tlr = next((sd[k].shape[0] for k in K if k.startswith('encoder.base_model.model.layers.0.self_attn.q_proj.lora_A')), 16)
m = DeterministicVQA(vision_model_name=sa.get('vision_model'), bartpho_model_name='vinai/bartpho-syllable',
    num_fusion_layers=sa.get('num_fusion_layers', 2), fusion_type=sa.get('fusion_type', 'text2vision'),
    use_text_lora=True, text_lora_r=tlr, text_lora_alpha=sa.get('text_lora_alpha', 32),
    use_vision_gate=True, vision_gate_init=sa.get('vision_gate_init', 1.5),
    vision_gate_min_alpha=sa.get('vision_gate_min_alpha', 0.0), vision_gate_max_alpha=sa.get('vision_gate_max_alpha', 1.0),
    use_type_task=any(k.startswith('type_head.') or k.startswith('type_classifier.') for k in K),
    use_siglip_pooler=sa.get('use_siglip_pooler', True)).to(DEV).eval()
r = m.load_state_dict(sd, strict=False); assert not [k for k in r.missing_keys if 'teacher' not in k]
for q in m.parameters(): q.requires_grad_(False)
tok = m.tokenizer

ST = {'tproj': None, 'W': None, 'alpha': None, 'mask': None}
m.vision_gating.text_proj.register_forward_hook(lambda mod, i, o: ST.__setitem__('tproj', o))
def ln_pre(mod, inp):
    """Adds (1-alpha)*(c_i - t_bar) to the pre-LN blend. Computed HERE because t_proj is produced
    earlier inside this same VisionGating.forward call. W=0 -> uniform attention -> c_i = t_bar
    -> offset exactly 0 -> bit-identical to the current model."""
    W = ST['W']
    if W is None: return None
    tp = ST['tproj']                                  # [B, L, D]
    lm = ST['mask'].bool()[0]                         # [L]
    tpr = tp[:, lm, :]                                # [B, Lreal, D]
    att = torch.softmax(W, dim=-1)                    # [P, Lreal]
    c = torch.einsum('pl,bld->bpd', att, tpr)         # [B, P, D]
    tbar = tpr.mean(1, keepdim=True)                  # [B, 1, D]
    off = (1 - ST['alpha']).unsqueeze(-1) * (c - tbar)
    return (inp[0] + off,)
m.vision_gating.layer_norm.register_forward_pre_hook(ln_pre)

tr = pd.read_csv(a.train_csv or sa.get('train_csv', 'archive/train_split_original.csv')); tr['an'] = tr.answer.map(norm)
TVOC = {t: sorted(set(g.an)) for t, g in tr.groupby('type')}
LB = {}
for t, voc in TVOC.items():
    e = tok(voc, return_tensors='pt', padding='max_length', truncation=True, max_length=10)
    x = e.input_ids.to(DEV).clone(); x[x == tok.pad_token_id] = -100; LB[t] = x
te = pd.read_csv('archive/test.csv'); gold = te.answer.map(norm)
vp = AutoProcessor.from_pretrained(sa.get('vision_model'))
ds = VQAGenDataset(csv_path='archive/test.csv', image_folder='archive/data/images/test', vision_processor=vp,
    tokenizer_name='vinai/bartpho-syllable', max_q_len=32, max_a_len=10, include_question_type=True, auto_detect_type=False)
rng = np.random.default_rng(0); sel = []
for t in [0, 1, 2, 3]:
    pool = [i for i in range(len(te)) if int(te.type.iloc[i]) == t and gold.iloc[i] in TVOC[t]]
    sel += rng.choice(pool, min(a.per_type, len(pool)), replace=False).tolist()
sel = sorted(sel)
if a.limit: sel = sel[:a.limit]

def logp(lg, lb):
    return -F.cross_entropy(lg.reshape(-1, lg.size(-1)).float(), lb.reshape(-1), ignore_index=-100, reduction='none').view(lb.shape).sum(1)
def fwd(pv, ii, am, qt, lb):
    return m(pixel_values=pv.expand(lb.size(0), -1, -1, -1), input_ids=ii.expand(lb.size(0), -1),
             attention_mask=am.expand(lb.size(0), -1), labels=lb, question_types=qt.expand(lb.size(0)))
def score(pv, ii, am, qt, lb):
    s = torch.empty(lb.size(0), device=DEV)
    with torch.no_grad():
        for st in range(0, lb.size(0), a.chunk):
            x = lb[st:st + a.chunk]; s[st:st + x.size(0)] = logp(fwd(pv, ii, am, qt, x).answer_logits, x)
    return s.cpu().numpy()

rows, Wstore = [], {}
for n, j in enumerate(sel):
    b = next(iter(DataLoader(Subset(ds, [j]), batch_size=1)))
    pv = b['pixel_values'].to(DEV); ii = b['input_ids'].to(DEV); am = b['attention_mask'].to(DEV)
    qt = b['question_type'].to(DEV).long()
    t = int(te.type.iloc[j]); voc = TVOC[t]; gi = voc.index(gold.iloc[j]); L = LB[t]; gl = L[gi:gi + 1]
    ST['W'] = None; ST['mask'] = am; m.vision_gating.alpha_override = None
    sb = score(pv, ii, am, qt, L)
    a0 = m.vision_gating.last_alpha.detach().float()
    if a0.dim() == 3: a0 = a0.squeeze(-1)
    a0 = a0[:1]; P = a0.size(1)
    Lreal = int(am[0].sum().item())
    rec = dict(idx=j, type=t, P=P, Lreal=Lreal, hit_base=int(sb.argmax() == gi))

    for arm in ['A1', 'A2']:
        init = a0.mean(1, keepdim=True) if arm == 'A1' else a0
        th = torch.logit(init.clamp(1e-4, 1 - 1e-4)).clone().requires_grad_(True)
        opt = torch.optim.Adam([th], lr=a.lr_alpha)
        ST['W'] = None
        with torch.enable_grad():
            for _ in range(a.steps):
                opt.zero_grad(set_to_none=True)
                m.vision_gating.alpha_override = torch.sigmoid(th)
                (-logp(fwd(pv, ii, am, qt, gl).answer_logits, gl).sum()).backward(); opt.step()
        m.vision_gating.alpha_override = torch.sigmoid(th).detach()
        s = score(pv, ii, am, qt, L); rec[f'hit_{arm}'] = int(s.argmax() == gi)
        m.vision_gating.alpha_override = None

    W = torch.zeros(P, Lreal, device=DEV, requires_grad=True)
    ST['alpha'] = a0; ST['W'] = W
    if n == 0:   # NESTEDNESS ASSERTION: W=0 must reproduce the base scores exactly
        s0 = score(pv, ii, am, qt, L)
        assert np.allclose(s0, sb, atol=1e-3), f'W=0 not nested: max|d|={np.abs(s0-sb).max():.4g}'
        print('  [check] W=0 reproduces base scores exactly -> injection is nested', flush=True)
    opt = torch.optim.Adam([W], lr=a.lr_target)
    with torch.enable_grad():
        for _ in range(a.steps):
            opt.zero_grad(set_to_none=True)
            (-logp(fwd(pv, ii, am, qt, gl).answer_logits, gl).sum()).backward(); opt.step()
    Wd = W.detach(); ST['W'] = Wd
    s = score(pv, ii, am, qt, L); rec['hit_A3'] = int(s.argmax() == gi)
    rec['att_entropy'] = float(-(torch.softmax(Wd, -1) * torch.log_softmax(Wd, -1)).sum(-1).mean())
    rec['att_maxw'] = float(torch.softmax(Wd, -1).max(-1).values.mean())
    # A3b: same arm at a larger lr, so a negative cannot be blamed on under-optimisation
    W2 = torch.zeros(P, Lreal, device=DEV, requires_grad=True)
    ST['W'] = W2; opt2 = torch.optim.Adam([W2], lr=a.lr_target2)
    with torch.enable_grad():
        for _ in range(a.steps):
            opt2.zero_grad(set_to_none=True)
            (-logp(fwd(pv, ii, am, qt, gl).answer_logits, gl).sum()).backward(); opt2.step()
    ST['W'] = W2.detach()
    s2 = score(pv, ii, am, qt, L); rec['hit_A3b'] = int(s2.argmax() == gi)
    Wstore[j] = (Wd.cpu().numpy(), t)
    ST['W'] = None
    rows.append(rec)
    if (n + 1) % 50 == 0:
        pd.DataFrame(rows).to_csv(f'analysis/mech/target_{a.tag}.csv', index=False)
        print(f'  {n+1}/{len(sel)}', flush=True)

# A4 null: another sample's fitted W, same type
keys = list(Wstore.keys()); bytype = {}
for k, (w, t) in Wstore.items(): bytype.setdefault(t, []).append(k)
rows2 = []
for n, rec in enumerate(rows):
    j = rec['idx']; t = rec['type']; pool = [k for k in bytype[t] if k != j]
    if not pool: rec['hit_A4'] = rec['hit_base']; rows2.append(rec); continue
    k = pool[rng.integers(len(pool))]
    b = next(iter(DataLoader(Subset(ds, [j]), batch_size=1)))
    pv = b['pixel_values'].to(DEV); ii = b['input_ids'].to(DEV); am = b['attention_mask'].to(DEV)
    qt = b['question_type'].to(DEV).long(); voc = TVOC[t]; gi = voc.index(gold.iloc[j]); L = LB[t]
    ST['W'] = None; ST['mask'] = am; m.vision_gating.alpha_override = None
    score(pv, ii, am, qt, L[:1])
    a0 = m.vision_gating.last_alpha.detach().float()
    if a0.dim() == 3: a0 = a0.squeeze(-1)
    a0 = a0[:1]
    Wn = torch.tensor(Wstore[k][0], device=DEV)
    Lr = int(am[0].sum().item())
    Wn = Wn[:, :Lr] if Wn.size(1) >= Lr else F.pad(Wn, (0, Lr - Wn.size(1)))
    ST['alpha'] = a0; ST['W'] = Wn
    s = score(pv, ii, am, qt, L); rec['hit_A4'] = int(s.argmax() == gi)
    ST['W'] = None; rows2.append(rec)
d = pd.DataFrame(rows2); d.to_csv(f'analysis/mech/target_{a.tag}.csv', index=False)
print(f'\nsaved analysis/mech/target_{a.tag}.csv ({len(d)})\n')
NAM = {'hit_A1': 'A1 scalar alpha (1 dof)', 'hit_A2': 'A2 per-patch alpha (P dof)',
       'hit_A3': 'A3 per-patch TARGET c_i lr=0.1', 'hit_A3b': 'A3b per-patch TARGET c_i lr=0.5',
       'hit_A4': 'A4 null: shuffled c_i'}
bb = d.hit_base.values
print(f'base EM = {100*bb.mean():.2f}   (n={len(d)}, mean L_real={d.Lreal.mean():.1f}, P={int(d.P.iloc[0])})\n')
print(f'{"arm":<38}{"EM":>8}{"headroom":>11}{"rescued":>9}{"broken":>8}')
for c, nm in NAM.items():
    h = d[c].values
    print(f'{nm:<38}{100*h.mean():>8.2f}{100*(h.mean()-bb.mean()):>+11.2f}{int(((bb==0)&(h==1)).sum()):>9}{int(((bb==1)&(h==0)).sum()):>8}')
print(f'\n{"type":<10}{"base":>8}' + ''.join(f'{k[4:]:>12}' for k in NAM))
for t in [0, 1, 2, 3]:
    s = d[d.type == t]
    if not len(s): continue
    print(f'{T[t]:<10}{100*s.hit_base.mean():>8.2f}' + ''.join(f'{100*(s[c].mean()-s.hit_base.mean()):>+12.2f}' for c in NAM))
print(f'\nfitted attention: mean entropy {d.att_entropy.mean():.3f} (uniform = {np.log(d.Lreal.mean()):.3f}), mean max weight {d.att_maxw.mean():.3f}')
