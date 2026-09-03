"""Is the ORACLE per-patch alpha a learnable function of what TCVG's gate ALREADY SEES?

This is the sharpest possible test of the current architecture, because it does not invent new
features: it captures the literal input tensor of `VisionGating.gate_net`, which is

    gate_input_i = [ v_proj_i ; W_q[t_cls ; e_type] ]        (model.py, VisionGating.forward)

and asks whether the oracle's per-patch alpha move can be regressed from it, held out BY SAMPLE.

  R2 > 0 across held-out samples  -> the information IS in gate_net's input; TCVG's failure is an
                                     optimisation/learning-signal problem, and alpha-distillation
                                     or a per-patch head on the SAME input is the right fix.
  R2 ~ 0                          -> the oracle alpha is not a function of gate_net's input at all;
                                     no reparameterisation of the current gate can reach it, and the
                                     gate needs a genuinely new input, not a bigger MLP.

Held-out unit is the SAMPLE, never the patch: patches inside one image are highly correlated, so a
patch-level split would leak and report a fake positive.
"""
import sys, argparse, time, unicodedata as ud
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
p.add_argument('--out', required=True)
a = p.parse_args()
DEV = 'cuda'
norm = lambda s: ud.normalize('NFC', str(s)).strip().lower()

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

CAP = {}
def hook(mod, inp): CAP['x'] = inp[0].detach()      # [B, P, 2D] = [v_proj ; query]
m.vision_gating.gate_net.register_forward_pre_hook(hook)

te = pd.read_csv('archive/test.csv')
vp = AutoProcessor.from_pretrained(sa.get('vision_model'))
ds = VQAGenDataset(csv_path='archive/test.csv', image_folder='archive/data/images/test',
                   vision_processor=vp, tokenizer_name='vinai/bartpho-syllable', max_q_len=32,
                   max_a_len=10, include_question_type=True, auto_detect_type=False)

mech = pd.read_csv(a.mech_csv); err = mech[mech.hit_base == 0]
Z = np.load(a.alpha_npz)
X, Y, S, TY, RS = [], [], [], [], []
t0 = time.time()
for n, row in enumerate(err.itertuples()):
    j = int(row.idx)
    if str(j) not in Z.files: continue
    af = Z[str(j)]; a0 = Z[f'base_{j}']
    b = next(iter(DataLoader(Subset(ds, [j]), batch_size=1)))
    with torch.no_grad():
        m.vision_gating.alpha_override = None
        m(pixel_values=b['pixel_values'].to(DEV), input_ids=b['input_ids'].to(DEV),
          attention_mask=b['attention_mask'].to(DEV),
          labels=b['labels'].to(DEV) if 'labels' in b else None,
          question_types=b['question_type'].to(DEV).long())
    gi = CAP['x'][0].float().cpu().numpy()                 # [P, 2D]
    X.append(gi.astype(np.float16)); Y.append((af - a0).astype(np.float32))
    S.append(j); TY.append(int(row.type)); RS.append(int(row.hit_fit))
    if (n + 1) % 50 == 0: print(f'  {n+1}/{len(err)} ({(time.time()-t0)/60:.1f}m)', flush=True)

X = np.stack(X); Y = np.stack(Y)
np.savez_compressed(a.out, X=X, Y=Y, sample=np.array(S), type=np.array(TY), rescued=np.array(RS))
print(f'\nsaved {a.out}  X={X.shape} Y={Y.shape}\n')

from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
S = np.array(S); TY = np.array(TY); RS = np.array(RS)
N, P, D = X.shape
def evaluate(mask, name):
    idx = np.where(mask)[0]
    if len(idx) < 10: print(f'  {name}: too few samples ({len(idx)})'); return
    Xf = X[idx].reshape(-1, D).astype(np.float32)
    Yf = Y[idx].reshape(-1)
    g = np.repeat(np.arange(len(idx)), P)                  # group = SAMPLE
    pred = np.zeros_like(Yf)
    for tr, teI in GroupKFold(n_splits=5).split(Xf, Yf, groups=g):
        mo = Ridge(alpha=100.0).fit(Xf[tr], Yf[tr]); pred[teI] = mo.predict(Xf[teI])
    # within-sample R2: can it predict the SHAPE of the alpha field, not just its mean?
    Yr = Y[idx]; Pr = pred.reshape(len(idx), P)
    wr = 1 - ((Yr - Pr) ** 2).sum() / ((Yr - Yr.mean(1, keepdims=True)) ** 2).sum()
    print(f'  {name:<28} n={len(idx):>3}  R2(all patches)={r2_score(Yf,pred):>7.3f}   R2(within-sample shape)={wr:>7.3f}')
print('Ridge on gate_net\'s OWN input -> oracle per-patch alpha move, grouped 5-fold by SAMPLE:')
evaluate(np.ones(N, bool), 'all errors')
evaluate(RS == 1, 'oracle-RESCUED errors only')
for t, nm in [(1, 'COUNT'), (2, 'COLOR'), (3, 'LOCATION'), (0, 'OBJECT')]:
    evaluate(TY == t, nm)
