"""Chong ban do alpha 14x14 len anh goc — cho slide."""
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
CASES = [
    (149577, 'mau cua con chim la gi', 'gold: mau do  |  T0/T1: mau nau', 'COLOR'),
    (361778, 'gau teddy ngoi o dau', 'gold: cai ghe  |  T0/T1: van phong', 'LOCATION'),
]
for img, q, ans, ty in CASES:
    a = np.load(f'analysis/alphamap_{img}.npy')
    im = Image.open(f'archive/data/images/test/{img}.jpg').convert('RGB').resize((448, 448))
    up = np.kron(a, np.ones((32, 32)))                      # 14x14 -> 448x448
    fig, ax = plt.subplots(1, 3, figsize=(15, 5.6))
    ax[0].imshow(im); ax[0].set_title('anh goc', fontsize=13)
    h = ax[1].imshow(a, cmap='inferno', vmin=0, vmax=1, interpolation='nearest')
    ax[1].set_title(f'alpha 14x14  (mean {a.mean():.2f}, std {a.std():.3f})', fontsize=13)
    plt.colorbar(h, ax=ax[1], fraction=0.046)
    ax[2].imshow(im); ax[2].imshow(up, cmap='inferno', vmin=0, vmax=1, alpha=0.55)
    ax[2].set_title('chong len anh', fontsize=13)
    for x in ax: x.set_xticks([]); x.set_yticks([])
    fig.suptitle(f'[{ty}]  "{q}"     {ans}', fontsize=14)
    plt.tight_layout()
    out = f'figs/item5_case_{ty}_{img}.png'
    plt.savefig(out, dpi=130, bbox_inches='tight'); plt.close()
    print(f'{out}   alpha mean={a.mean():.3f} std={a.std():.3f} max={a.max():.3f}')
