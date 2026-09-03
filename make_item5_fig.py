"""MUC 5 — HINH: MOT ANH, BON CAU HOI, BON BAN DO ALPHA.

Y nghia cua hinh nay: cung MOT anh va cung MOT model, alpha chay tu ~0.42 den ~0.9995 CHI VI cau
hoi doi. Do la thu TCVG that su lam, nhin duoc trong mot khung hinh.

Chay eval.py DUNG MOT LAN tren mot CSV bon dong (cung img_id), nen bon ban do alpha den tu cung
mot lan nap model — khong ghep tu bon lan chay khac nhau.

    python3 make_item5_fig.py                       # anh mac dinh trong test
    python3 make_item5_fig.py --image anh.jpg --questions "cau 1" "cau 2" ...
"""
import argparse, os, subprocess, sys, tempfile, shutil
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

MAIN = os.path.dirname(os.path.abspath(__file__))
PY = '/home/user/workspace/all_env/vivqa/bin/python3'
TYPE_NAME = {0: 'OBJECT', 1: 'COUNT', 2: 'COLOR', 3: 'LOCATION'}
DEFAULT_Q = [
    'điêu khắc của một con chim đang ngồi ở đâu',
    'cái gì đang ngồi trên cửa sổ',
    'có bao nhiêu con chim',
    'màu của con chim là gì',
]

ap = argparse.ArgumentParser()
ap.add_argument('--image', default=f'{MAIN}/archive/data/images/test/100014.jpg')
ap.add_argument('--questions', nargs='+', default=DEFAULT_Q)
ap.add_argument('--checkpoint', default=f'{MAIN}/checkpoints_run87/best_model.pt')
ap.add_argument('--out', default=f'{MAIN}/figs/item5_demo_one_image.png')
a = ap.parse_args()

tmp = tempfile.mkdtemp(prefix='item5_')
try:
    imgdir = os.path.join(tmp, 'images'); os.makedirs(imgdir)
    iid = 999999
    shutil.copy(a.image, os.path.join(imgdir, f'{iid}.jpg'))
    csv = os.path.join(tmp, 'q.csv')
    pd.DataFrame([{'question': q, 'answer': '<x>', 'img_id': iid, 'type': 0}
                  for q in a.questions]).to_csv(csv)
    out_csv = os.path.join(tmp, 'out.csv'); npy = os.path.join(tmp, 'alpha.npy')
    r = subprocess.run([PY, 'src/eval.py', '--checkpoint', a.checkpoint,
                        '--csv_path', csv, '--image_folder', imgdir, '--output_csv', out_csv,
                        '--num_beams', '3', '--repetition_penalty', '1.3', '--max_length', '10',
                        '--use_synonyms', '--use_constrained',
                        '--train_csv_for_trie', 'archive/train_split.csv',
                        '--dump_model_alpha', npy], cwd=MAIN, capture_output=True, text=True)
    if not os.path.isfile(out_csv):
        print(r.stdout[-2500:]); print(r.stderr[-2500:]); sys.exit('eval.py that bai')
    d = pd.read_csv(out_csv)
    al = np.load(npy)
    im = Image.open(a.image).convert('RGB').resize((448, 448))

    n = len(a.questions)
    fig, ax = plt.subplots(2, n, figsize=(4.1 * n, 8.6))
    if n == 1: ax = ax.reshape(2, 1)
    for i in range(n):
        v = np.asarray(al[i], dtype=np.float32).ravel()
        s = int(round(len(v) ** 0.5))
        g = v[-s * s:].reshape(s, s)
        up = np.kron(g, np.ones((448 // s, 448 // s)))
        pt = d['pred_question_type'].iloc[i] if 'pred_question_type' in d.columns else None
        # cot nay co the la CHUOI ('COLOR') hoac SO (2) tuy phien ban eval.py -> nhan ca hai
        ty = '?'
        if pt is not None and not pd.isna(pt):
            if str(pt).strip().upper() in TYPE_NAME.values():
                ty = str(pt).strip().upper()
            else:
                try:
                    ty = TYPE_NAME.get(int(float(pt)), '?')
                except (TypeError, ValueError):
                    ty = '?'
        if ty == '?':
            print(f'   [canh bao] khong doc duoc loai cho cau {i}: pred_question_type = {pt!r}')
        ax[0, i].imshow(im); ax[0, i].imshow(up, cmap='inferno', vmin=0, vmax=1, alpha=0.55)
        ax[0, i].set_title(f'[{ty}]  alpha tb = {g.mean():.3f}', fontsize=12)
        h = ax[1, i].imshow(g, cmap='inferno', vmin=0, vmax=1, interpolation='nearest')
        ax[1, i].set_title(f'"{a.questions[i]}"\n-> {d.prediction.iloc[i]}', fontsize=11)
        plt.colorbar(h, ax=ax[1, i], fraction=0.046)
        for x in (ax[0, i], ax[1, i]): x.set_xticks([]); x.set_yticks([])
    fig.suptitle('MOT anh, MOT model — alpha thay doi CHI vi cau hoi thay doi', fontsize=15)
    plt.tight_layout()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    plt.savefig(a.out, dpi=130, bbox_inches='tight'); plt.close()
    print(f'da luu {a.out}')
    for i in range(n):
        v = np.asarray(al[i], dtype=np.float32).ravel()
        print(f'   "{a.questions[i][:44]:44s}" -> {str(d.prediction.iloc[i]):14s} alpha {v.mean():.4f}')
finally:
    shutil.rmtree(tmp, ignore_errors=True)
