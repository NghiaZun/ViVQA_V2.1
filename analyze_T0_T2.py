#!/usr/bin/env python3
"""Muc 3 (P0) experiment.md: so ghep cap T0 vs T2 tren cung tung seed."""
import os, re
import numpy as np
import pandas as pd
from scipy import stats

BASE = '/home/user/workspace/nghia.duong/thesis'
OUT = os.path.join(BASE, 'analysis/final'); os.makedirs(OUT, exist_ok=True)
TYPES = ['OBJECT', 'COUNT', 'COLOR', 'LOCATION']
PAIRED = [0, 1, 2, 3, 4, 5, 6, 7, 8]     # seed co ca T0 lan T2; 42 bao rieng


def parse(path):
    if not os.path.exists(path):
        return None
    t = open(path, errors='ignore').read().replace('\r', '\n')
    m = re.search(r'Exact Match:\s*([0-9.]+)%', t)
    if not m:
        return None
    r = {'Overall': float(m.group(1))}
    for ty in TYPES:
        mm = re.search(rf'^\s*{ty}\s+([0-9.]+)\s+([0-9.]+)\s+(\d+)', t, re.M)
        if mm:
            r[ty] = float(mm.group(1))
    return r


rows = []
for s in PAIRED + [42]:
    t0 = parse(f'{BASE}/analysis/T0/T0_seed{s}.log')
    t2 = parse(f'{BASE}/beam3fixed/seed{s}_ep40.log')
    if not t0 or not t2:
        print(f'thieu seed {s}: T0={bool(t0)} T2={bool(t2)}')
        continue
    row = {'seed': s}
    for k in ['Overall'] + TYPES:
        row[f'T0_{k}'] = t0.get(k)
        row[f'T2_{k}'] = t2.get(k)
        row[f'd_{k}'] = t2.get(k) - t0.get(k)
    rows.append(row)
df = pd.DataFrame(rows)
df.to_csv(f'{OUT}/vivqa_paired_T0_T2.csv', index=False)

p = df[df.seed != 42]
print('=' * 92)
print(f'MUC 3 (P0) — T0 vs T2 ghep cap, seed {PAIRED} (n={len(p)}); seed 42 bao rieng')
print('=' * 92)
print(df[['seed', 'T0_Overall', 'T2_Overall', 'd_Overall']].to_string(
    index=False, float_format=lambda x: f'{x:8.2f}'))

print('\n--- Bang 3.6: mean +/- std tren tung loai ---')
tab = []
for name, sub in [('T0', 'T0_'), ('T2', 'T2_'), ('Delta ghep cap', 'd_')]:
    r = {'Mô hình': name}
    for k in TYPES + ['Overall']:
        v = p[f'{sub}{k}']
        r[k] = f'{v.mean():.2f} ± {v.std(ddof=1):.2f}'
    tab.append(r)
print(pd.DataFrame(tab).to_string(index=False))

print('\n--- Bang kiem dinh (n=%d) ---' % len(p))
st = []
for k in ['Overall'] + TYPES:
    d = p[f'd_{k}'].values
    ci = stats.t.interval(0.95, len(d) - 1, loc=d.mean(), scale=stats.sem(d))
    t, pt = stats.ttest_rel(p[f'T2_{k}'], p[f'T0_{k}'])
    try:
        pw = stats.wilcoxon(p[f'T2_{k}'], p[f'T0_{k}']).pvalue
    except ValueError:
        pw = float('nan')
    st.append({'Chỉ số': k, 'Mean paired diff': f'{d.mean():+.2f}',
               '95% CI': f'[{ci[0]:+.2f}, {ci[1]:+.2f}]',
               'paired t p': f'{pt:.5f}', 'Wilcoxon p': f'{pw:.5f}'})
stdf = pd.DataFrame(st)
print(stdf.to_string(index=False))
stdf.to_csv(f'{OUT}/vivqa_statistical_tests_T0_T2.csv', index=False)

r42 = df[df.seed == 42]
if len(r42):
    r42 = r42.iloc[0]
    print(f'\nseed 42 (rieng): T0={r42.T0_Overall:.2f}  T2={r42.T2_Overall:.2f}  Delta={r42.d_Overall:+.2f}')
print(f'\nDa ghi {OUT}/vivqa_paired_T0_T2.csv va vivqa_statistical_tests_T0_T2.csv')
