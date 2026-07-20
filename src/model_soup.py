"""
Model Soup: Greedy weight averaging of run25 + run56.

Greedy soup: start with best checkpoint (run25), add run56 only if val EM improves.
Uses the same eval pipeline as train.py to ensure fair comparison.
"""
import os
import sys
import copy
import torch
import csv
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE_DIR = os.path.dirname(__file__)

SOUP_DIR = os.path.join(BASE_DIR, 'checkpoints_soup_r25_r56')
os.makedirs(SOUP_DIR, exist_ok=True)

CANDIDATES = [
    os.path.join(BASE_DIR, 'checkpoints_run25', 'best_model.pt'),
    os.path.join(BASE_DIR, 'checkpoints_run56', 'best_model.pt'),
]

VAL_CSV    = os.path.join(BASE_DIR, 'archive', 'val_split.csv')
TEST_CSV   = os.path.join(BASE_DIR, 'archive', 'test.csv')
IMAGE_DIR_VAL  = os.path.join(BASE_DIR, 'archive', 'data', 'images', 'train')
IMAGE_DIR_TEST = os.path.join(BASE_DIR, 'archive', 'data', 'images', 'test')


def eval_checkpoint(ckpt_path, split='val'):
    """Quick eval using eval.py pipeline. Returns overall EM."""
    import subprocess, tempfile, json
    out_csv = os.path.join(SOUP_DIR, f'_tmp_eval_{split}.csv')
    csv_path  = VAL_CSV  if split == 'val' else TEST_CSV
    img_dir   = IMAGE_DIR_VAL if split == 'val' else IMAGE_DIR_TEST
    cmd = [
        sys.executable, os.path.join(BASE_DIR, 'eval.py'),
        '--checkpoint', ckpt_path,
        '--csv_path', csv_path,
        '--image_folder', img_dir,
        '--output_csv', out_csv,
        '--num_beams', '3',
        '--repetition_penalty', '1.3',
        '--max_length', '10',
        '--use_synonyms',
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    rows = list(csv.DictReader(open(out_csv)))
    by_type = defaultdict(list)
    for r in rows:
        by_type[r['question_type']].append(float(r['exact_match']))
    total = sum(float(r['exact_match']) for r in rows) / len(rows) * 100
    per_type = {t: sum(v)/len(v)*100 for t, v in by_type.items()}
    return total, per_type


def average_state_dicts(sd1, sd2, alpha=0.5):
    """Return (1-alpha)*sd1 + alpha*sd2."""
    out = {}
    for k in sd1:
        v1 = sd1[k].float()
        v2 = sd2[k].float()
        out[k] = (1 - alpha) * v1 + alpha * v2
    return out


def save_soup(state_dict, path):
    torch.save({'model_state_dict': state_dict, 'epoch': 'soup', 'stage': 3}, path)


def main():
    print('='*60)
    print('GREEDY MODEL SOUP: run25 + run56')
    print('='*60)

    # Load state dicts
    print('\n[1] Loading checkpoints...')
    ckpts = []
    for path in CANDIDATES:
        ck = torch.load(path, map_location='cpu')
        ckpts.append(ck['model_state_dict'])
        print(f'  Loaded: {path}')

    # Start: run25 alone
    print('\n[2] Evaluating run25 baseline on val...')
    base_em, base_per = eval_checkpoint(CANDIDATES[0], split='val')
    print(f'  run25 val EM: {base_em:.2f}%  {base_per}')

    best_sd   = ckpts[0]
    best_em   = base_em
    best_name = 'run25'

    # Try adding run56
    for i, (path, sd) in enumerate(zip(CANDIDATES[1:], ckpts[1:]), start=1):
        name = os.path.basename(os.path.dirname(path))
        print(f'\n[3] Trying soup: {best_name} + {name} (α=0.5)...')
        soup_sd = average_state_dicts(best_sd, sd, alpha=0.5)
        tmp_path = os.path.join(SOUP_DIR, '_tmp_soup.pt')
        save_soup(soup_sd, tmp_path)

        soup_em, soup_per = eval_checkpoint(tmp_path, split='val')
        print(f'  Soup val EM: {soup_em:.2f}%  {soup_per}')

        if soup_em > best_em:
            print(f'  ✅ IMPROVED ({best_em:.2f}% → {soup_em:.2f}%), keeping soup')
            best_sd   = soup_sd
            best_em   = soup_em
            best_name = f'{best_name}+{name}'
        else:
            print(f'  ❌ No improvement ({soup_em:.2f}% < {best_em:.2f}%), keeping {best_name}')

    # Save final soup
    final_path = os.path.join(SOUP_DIR, 'best_model.pt')
    save_soup(best_sd, final_path)
    print(f'\n[4] Final soup saved: {final_path}  (components: {best_name})')

    # Test eval
    print('\n[5] Evaluating final soup on TEST...')
    test_em, test_per = eval_checkpoint(final_path, split='test')
    run25_test = {'COLOR': 72.5, 'COUNT': 65.8, 'LOCATION': 69.3, 'OBJECT': 75.2}
    print(f'\n  SOUP test EM: {test_em:.2f}%  (run25=71.91%)')
    for t in ['COLOR', 'COUNT', 'LOCATION', 'OBJECT']:
        diff = test_per.get(t, 0) - run25_test[t]
        print(f'    {t}: {test_per.get(t,0):.2f}%  ({diff:+.2f}pp vs run25)')

    # Save test predictions
    import shutil
    shutil.copy(os.path.join(SOUP_DIR, '_tmp_eval_test.csv'),
                os.path.join(SOUP_DIR, 'eval_test_v3.csv'))
    print(f'\n  Saved: {SOUP_DIR}/eval_test_v3.csv')


if __name__ == '__main__':
    main()
