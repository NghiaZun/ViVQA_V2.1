#!/usr/bin/env python3
"""
Phan tich Giai doan 1 cua experiment.md (khong can train lai):
  Muc 4 — type-head metrics: accuracy, macro/weighted P-R-F1, per-class, confusion matrix
  Muc 5 — predicted-type vs gold-type oracle gap, 11 checkpoint
  Muc 7 — dem tham so theo thanh phan

Ghi ket qua ra analysis/final/*.csv va in bang de dan vao luan van.
"""
import json, os, re, sys
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (classification_report, confusion_matrix,
                             precision_recall_fscore_support, accuracy_score)

BASE = '/home/user/workspace/nghia.duong/thesis'
OUT = os.path.join(BASE, 'analysis/final')
os.makedirs(OUT, exist_ok=True)
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
TYPES = ['OBJECT', 'COUNT', 'COLOR', 'LOCATION']


def em_from_log(path):
    if not os.path.exists(path):
        return None
    t = open(path, errors='ignore').read().replace('\r', '\n')
    m = re.search(r'Exact Match:\s*([0-9.]+)%', t)
    return float(m.group(1)) if m else None


def per_type_from_log(path):
    t = open(path, errors='ignore').read().replace('\r', '\n')
    out = {}
    for ty in TYPES:
        m = re.search(rf'^\s*{ty}\s+([0-9.]+)\s+([0-9.]+)\s+(\d+)', t, re.M)
        if m:
            out[ty] = float(m.group(1))
    return out


# ---------------------------------------------------------------- Muc 5
print('=' * 78)
print('MUC 5 — Predicted-type vs Gold-type oracle  (test EM, epoch 40, beam3 da sua)')
print('=' * 78)
rows = []
for s in SEEDS + [42]:
    p = em_from_log(f'{BASE}/analysis/oracle_gap/vivqa_predicted_seed{s}.log')
    g = em_from_log(f'{BASE}/analysis/oracle_gap/vivqa_gold_seed{s}.log')
    if p is None or g is None:
        continue
    rows.append({'seed': s, 'predicted_EM': p, 'gold_EM': g, 'oracle_gap': g - p})
og = pd.DataFrame(rows)
print(og.to_string(index=False, float_format=lambda x: f'{x:7.2f}'))
m = og[og.seed != 42]
print(f'\nseed 0-9 (n={len(m)}):')
print(f'  predicted-type EM : {m.predicted_EM.mean():.2f} +/- {m.predicted_EM.std(ddof=1):.2f}')
print(f'  gold-type EM      : {m.gold_EM.mean():.2f} +/- {m.gold_EM.std(ddof=1):.2f}')
print(f'  oracle gap        : {m.oracle_gap.mean():+.3f} +/- {m.oracle_gap.std(ddof=1):.3f}'
      f'  (min {m.oracle_gap.min():+.2f}, max {m.oracle_gap.max():+.2f})')
t, p = stats.ttest_rel(m.gold_EM, m.predicted_EM)
w = stats.wilcoxon(m.gold_EM, m.predicted_EM)
ci = stats.t.interval(0.95, len(m) - 1, loc=m.oracle_gap.mean(),
                      scale=stats.sem(m.oracle_gap))
print(f'  95% CI cua gap    : [{ci[0]:+.3f}, {ci[1]:+.3f}]')
print(f'  paired t-test     : t={t:.2f}, p={p:.4f}')
print(f'  Wilcoxon          : p={w.pvalue:.4f}')
og.to_csv(f'{OUT}/vivqa_oracle_gap.csv', index=False)

# ---------------------------------------------------------------- Muc 4
print()
print('=' * 78)
print('MUC 4 — Type-head metrics (ViVQA test, 3001 mau)')
print('=' * 78)
allrows = []
for s in SEEDS + [42]:
    f = f'{BASE}/analysis/oracle_gap/vivqa_predicted_seed{s}.csv'
    if not os.path.exists(f):
        continue
    d = pd.read_csv(f)
    if 'pred_question_type' not in d.columns or d.pred_question_type.isna().all():
        continue
    y_true, y_pred = d.question_type.astype(str), d.pred_question_type.astype(str)
    if (y_pred == '').all():
        continue
    acc = accuracy_score(y_true, y_pred)
    mp, mr, mf, _ = precision_recall_fscore_support(y_true, y_pred, average='macro',
                                                    labels=TYPES, zero_division=0)
    wp, wr, wf, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted',
                                                    labels=TYPES, zero_division=0)
    allrows.append({'seed': s, 'accuracy': acc * 100, 'macro_P': mp * 100, 'macro_R': mr * 100,
                    'macro_F1': mf * 100, 'weighted_F1': wf * 100})
    if s == 42:
        print('\nPer-class (seed 42):')
        print(classification_report(y_true, y_pred, labels=TYPES, digits=4, zero_division=0))
        cm = confusion_matrix(y_true, y_pred, labels=TYPES)
        cmdf = pd.DataFrame(cm, index=[f'gold_{t}' for t in TYPES],
                            columns=[f'pred_{t}' for t in TYPES])
        print('Confusion matrix (seed 42):')
        print(cmdf.to_string())
        cmdf.to_csv(f'{OUT}/confusion_matrix_vivqa_seed42.csv')

if allrows:
    th = pd.DataFrame(allrows)
    print('\nTong hop qua cac seed:')
    print(th.to_string(index=False, float_format=lambda x: f'{x:7.3f}'))
    mm = th[th.seed != 42]
    print(f'\nseed 0-9: accuracy={mm.accuracy.mean():.2f}+/-{mm.accuracy.std(ddof=1):.2f}'
          f'  macro_F1={mm.macro_F1.mean():.2f}+/-{mm.macro_F1.std(ddof=1):.2f}')
    th.to_csv(f'{OUT}/vivqa_type_metrics.csv', index=False)
else:
    print('Chua co cot pred_question_type — CSV phase1 duoc tao truoc khi them cot nay.')

# ---------------------------------------------------------------- Muc 7
print()
print('=' * 78)
print('MUC 7 — Dem tham so theo thanh phan (tu state_dict cua checkpoint T2)')
print('=' * 78)
import torch
ck = torch.load(f'{BASE}/checkpoints_run87_rerun/epoch_40.pt', map_location='meta', weights_only=False)
sd = ck['model_state_dict']
GROUPS = [
    ('SigLIP (vision encoder)', lambda k: k.startswith('vision_encoder')),
    ('BARTpho encoder (base)', lambda k: k.startswith('encoder.') and 'lora_' not in k),
    ('Encoder LoRA', lambda k: 'lora_' in k),
    ('GCA (flamingo_fusion)', lambda k: k.startswith('flamingo_fusion')),
    ('TCVG (vision_gating)', lambda k: k.startswith('vision_gating')),
    ('Type head', lambda k: k.startswith('type_head')),
    ('Decoder', lambda k: k.startswith('decoder')),
    ('LM head', lambda k: k.startswith('lm_head')),
]
seen, prows = set(), []
for name, pred in GROUPS:
    keys = [k for k in sd if pred(k) and k not in seen]
    seen.update(keys)
    n = sum(sd[k].numel() for k in keys)
    prows.append({'component': name, 'params': n, 'n_tensors': len(keys)})
rest = [k for k in sd if k not in seen]
prows.append({'component': 'Khac (' + ', '.join(sorted({k.split('.')[0] for k in rest})[:6]) + ')',
              'params': sum(sd[k].numel() for k in rest), 'n_tensors': len(rest)})
pc = pd.DataFrame(prows)
total = sum(v.numel() for v in sd.values())
pc['pct_of_total'] = pc.params / total * 100
print(pc.to_string(index=False, float_format=lambda x: f'{x:10.2f}'))
print(f'\nTong tham so trong state_dict: {total:,} ({total/1e6:.1f}M)')
print('Luu y: state_dict khong ghi requires_grad, nen cot trainable/frozen phai lay tu')
print('cau hinh (SigLIP frozen, BARTpho encoder base frozen tru LoRA, phan con lai trainable).')
pc.to_csv(f'{OUT}/parameter_count.csv', index=False)
print(f'\nDa ghi: {OUT}/vivqa_oracle_gap.csv, vivqa_type_metrics.csv, parameter_count.csv')
