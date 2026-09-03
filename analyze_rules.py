#!/usr/bin/env python3
"""
Tinh mean +/- std test EM cho tung rule chon best_model.
  - Thong ke chinh: seed 0-9 (n=10)
  - seed42 (= run87) bao rieng, KHONG gop vao mean/std
"""
import json, os
import numpy as np
import pandas as pd

BASE = '/home/user/workspace/nghia.duong/thesis'
RES = json.load(open(os.path.join(BASE, 'rule_study/results.json')))
PLAN = json.load(open('/tmp/claude-1000/-home-user-workspace/e04d8206-0198-4a27-b0b9-2cca49ef2296/scratchpad/need_evals.json'))
PICKS = PLAN['picks']
MAIN = [f'seed{s}' for s in range(10)]
REF = 'seed42'


def em(run, key):
    r = RES.get(f'{run}|{key}')
    return r['em'] if isinstance(r, dict) and r.get('em') is not None else np.nan


rows = []
for rule, per_run in PICKS.items():
    vals = {r: em(r, f'ep{per_run[r]}') for r in MAIN + [REF]}
    main = np.array([vals[r] for r in MAIN], dtype=float)
    rows.append({
        'rule': rule,
        'n': int(np.sum(~np.isnan(main))),
        'mean': np.nanmean(main),
        'std': np.nanstd(main, ddof=1),
        'min': np.nanmin(main) if np.any(~np.isnan(main)) else np.nan,
        'max': np.nanmax(main) if np.any(~np.isnan(main)) else np.nan,
        'seed42': vals[REF],
    })

for soup in ('soup_last5', 'soup_last10', 'soup_top5val'):
    vals = {r: em(r, soup) for r in MAIN + [REF]}
    main = np.array([vals[r] for r in MAIN], dtype=float)
    if np.all(np.isnan(main)):
        continue
    rows.append({
        'rule': soup,
        'n': int(np.sum(~np.isnan(main))),
        'mean': np.nanmean(main),
        'std': np.nanstd(main, ddof=1),
        'min': np.nanmin(main),
        'max': np.nanmax(main),
        'seed42': vals[REF],
    })

df = pd.DataFrame(rows).sort_values('mean', ascending=False)
pd.set_option('display.width', 160)
print('=== test EM theo rule chon best_model (protocol test hien tai) ===')
print('mean/std tren seed 0-9; cot seed42 la run87, KHONG nam trong thong ke\n')
print(df.to_string(index=False, float_format=lambda x: f'{x:6.2f}'))

# oracle: epoch tot nhat trong so nhung epoch da eval (chan duoi cua tran that su)
print('\n=== oracle (tot nhat trong cac epoch DA eval, chi la chan duoi) ===')
best = {}
for key, r in RES.items():
    if not isinstance(r, dict) or '|ep' not in key:
        continue
    run, ep = key.split('|')
    if r.get('em') is None:
        continue
    if run not in best or r['em'] > best[run][1]:
        best[run] = (ep, r['em'])
o = np.array([best[r][1] for r in MAIN if r in best], dtype=float)
if len(o):
    print(f'  n={len(o)} mean={o.mean():.2f} std={o.std(ddof=1):.2f}')
    for r in MAIN + [REF]:
        if r in best:
            print(f'    {r:7s} {best[r][0]:>5s} -> {best[r][1]:.2f}')
