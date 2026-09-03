#!/usr/bin/env python3
"""
So cac rule chon best_model tren 11 seed (0-9 + 42), moi seed 40 epoch.

Phase A: eval TEST cho cac (seed, epoch) ma bat ky rule nao chon toi  (52 eval)
Phase B: build + eval 3 loai weight soup cho moi seed                  (33 eval)
           soup_last5   = trung binh ep36-40      (khong chon gi ca)
           soup_last10  = trung binh ep31-40      (khong chon gi ca)
           soup_top5val = trung binh 5 epoch val-EM cao nhat

Ket qua -> rule_study/results.json  (test EM cho tung (run, item))
"""
import json, os, subprocess, sys, time
import torch
import pandas as pd

BASE = '/home/user/workspace/nghia.duong/thesis'
PY = '/home/user/workspace/all_env/vivqa/bin/python3'
OUT = os.path.join(BASE, 'rule_study')
PLAN = '/tmp/claude-1000/-home-user-workspace/e04d8206-0198-4a27-b0b9-2cca49ef2296/scratchpad/need_evals.json'

os.makedirs(OUT, exist_ok=True)
plan = json.load(open(PLAN))
RUNS = plan['runs']


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def eval_ckpt(ckpt, tag, run):
    """Chay eval.py tren test set. Tra ve dict EM/F1 hoac None."""
    d = os.path.join(OUT, run)
    os.makedirs(d, exist_ok=True)
    lg = os.path.join(d, f'{tag}.log')
    if os.path.exists(lg) and 'Exact Match' in open(lg, errors='ignore').read():
        return parse_log(lg)
    cmd = [PY, os.path.join(BASE, 'src/eval.py'),
           '--checkpoint', ckpt,
           '--csv_path', os.path.join(BASE, 'archive/test.csv'),
           '--image_folder', os.path.join(BASE, 'archive/data/images/test'),
           '--output_csv', os.path.join(d, f'{tag}.csv'),
           '--num_beams', '3', '--repetition_penalty', '1.3', '--max_length', '10',
           '--use_synonyms', '--use_constrained',
           '--train_csv_for_trie', os.path.join(BASE, 'archive/train_split.csv')]
    with open(lg, 'w') as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=BASE)
    return parse_log(lg)


def parse_log(path):
    em = f1 = None
    per_type = {}
    for line in open(path, errors='ignore').read().replace('\r', '\n').split('\n'):
        s = line.strip()
        if s.startswith('Exact Match:'):
            em = float(s.split(':')[1].strip().rstrip('%'))
        elif s.startswith('F1 Score:'):
            f1 = float(s.split(':')[1].strip().rstrip('%'))
        else:
            for t in ('COLOR', 'COUNT', 'LOCATION', 'OBJECT'):
                if s.startswith(t) and len(s.split()) >= 3:
                    try:
                        per_type[t] = float(s.split()[1])
                    except ValueError:
                        pass
    return None if em is None else {'em': em, 'f1': f1, 'per_type': per_type}


def build_soup(run_dir, epochs, dest):
    """Trung binh uniform model_state_dict cua cac epoch. Giu nguyen 'args'."""
    acc, n = None, 0
    template = None
    for e in epochs:
        c = torch.load(os.path.join(run_dir, f'epoch_{e}.pt'), map_location='cpu', weights_only=False)
        sd = c['model_state_dict']
        if acc is None:
            template = c
            acc = {k: (v.double().clone() if v.is_floating_point() else v.clone())
                   for k, v in sd.items()}
        else:
            for k, v in sd.items():
                if acc[k].is_floating_point():
                    acc[k] += v.double()
        n += 1
        del c
    souped = {k: (v / n).to(torch.float32) if v.is_floating_point() else v
              for k, v in acc.items()}
    torch.save({'model_state_dict': souped, 'args': template['args'],
                'epoch': f'soup{epochs}'}, dest)
    return n


results = {}
res_path = os.path.join(OUT, 'results.json')
if os.path.exists(res_path):
    results = json.load(open(res_path))


def save():
    json.dump(results, open(res_path, 'w'), indent=1)


# ---------------- Phase A: epoch duoc rule chon ----------------
need = plan['need']
log(f'PHASE A: {len(need)} test evals')
for i, (run, ep) in enumerate(need, 1):
    key = f'{run}|ep{ep}'
    if key in results:
        continue
    ck = os.path.join(BASE, RUNS[run], f'epoch_{ep}.pt')
    r = eval_ckpt(ck, f'test_ep{ep}', run)
    results[key] = r
    save()
    log(f'  [{i}/{len(need)}] {key} EM={r["em"] if r else "FAIL"}')
log('PHASE A done')

# ---------------- Phase B: soups ----------------
SOUPS = {
    'soup_last5':  lambda df: list(range(36, 41)),
    'soup_last10': lambda df: list(range(31, 41)),
    'soup_top5val': lambda df: sorted(df.nlargest(5, 'exact_match').epoch.astype(int).tolist()),
}
log(f'PHASE B: {len(RUNS) * len(SOUPS)} soup evals')
for run, d in RUNS.items():
    rd = os.path.join(BASE, d)
    df = pd.read_csv(os.path.join(rd, 'training_metrics.csv'))
    for sname, fn in SOUPS.items():
        key = f'{run}|{sname}'
        if key in results:
            continue
        eps = fn(df)
        dest = os.path.join(OUT, f'_tmp_{run}_{sname}.pt')
        n = build_soup(rd, eps, dest)
        r = eval_ckpt(dest, f'test_{sname}', run)
        results[key] = r
        results[f'{key}|epochs'] = eps
        save()
        try:
            os.remove(dest)
        except OSError:
            pass
        log(f'  {key} epochs={eps} EM={r["em"] if r else "FAIL"}')
log('PHASE B done')
log('ALL DONE')
