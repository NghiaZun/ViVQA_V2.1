"""Alpha toi uu THUC RA la gi? -- mo nhan alpha_oracle ra xem, khong doan.

Dau vao: analysis/oracle_alpha/train_alpha.npz (10800 x 197, sinh boi eval.py --dump_oracle_alpha)
         archive/train_split_original.csv (type, img_id)
         patch_region_map.pkl (chi so vung COCO that cho tung patch, 196)

Ba cau hoi, moi cau quyet dinh mot thu khac nhau ve tuong lai cua TCVG:
  1. PHAN RA PHUONG SAI: bao nhieu phan cua alpha_oracle nam o muc LOAI (type), muc MAU
     (sample), va muc PATCH trong mau? -> tra loi truc tiep "truc dieu kien hoa cua TCVG
     (type) co dung truc khong". Neu phan type ~0 thi TCVG dang dieu kien hoa sai truc.
  2. CAU TRUC KHONG GIAN: alpha_oracle co lien tuc theo khong gian khong (tuong quan lang
     gieng tren luoi 14x14)? -> co the giam sat bang box/vung khong, hay no la nhieu per-patch.
  3. DOI CHIEU VUNG COCO: alpha_oracle co cao hon tren patch thuoc vat the that khong?
     -> neu CO thi box supervision la proxy dung cho oracle (dung cach bai Dual-Pipeline lam);
        neu KHONG thi box khong phai duong di, phai distill truc tiep alpha.
"""
import numpy as np
import pandas as pd
import pickle

Z = np.load('analysis/oracle_alpha/train_alpha.npz')
A = Z['alpha'].astype('float32')                      # [N, 197]
df = pd.read_csv('archive/train_split_original.csv')
assert len(df) == A.shape[0], (len(df), A.shape)

# 197 = 1 token toan cuc (use_siglip_pooler chen o dau) + 196 patch luoi 14x14.
# Phai tach ra: token dau KHONG nam tren luoi, gop vao se lam ban moi phep do khong gian.
g_tok, P = A[:, 0], A[:, 1:]                           # [N], [N, 196]
G = 14
types = df['type'].to_numpy() if 'type' in df.columns else None
TN = {0: 'OBJECT', 1: 'COUNT', 2: 'COLOR', 3: 'LOCATION'}

print('=' * 78)
print('ALPHA TOI UU: alpha_oracle tren TRAIN split (nhan gold, run87)')
print('=' * 78)
print(f'shape={A.shape}  token toan cuc: mean={g_tok.mean():.3f}  '
      f'patch: mean={P.mean():.3f} sd={P.std():.3f}  min={P.min():.3f} max={P.max():.3f}')
print(f'ty le patch bi NEN (alpha<0.5): {(P < 0.5).mean() * 100:.1f}%   '
      f'giu gan nguyen (alpha>0.9): {(P > 0.9).mean() * 100:.1f}%')
print(f'so patch bi nen tren moi mau: trung binh {(P < 0.5).sum(1).mean():.1f}/196 '
      f'(trung vi {np.median((P < 0.5).sum(1)):.0f})')

# ── 1. PHAN RA PHUONG SAI ───────────────────────────────────────────────────
print('\n' + '-' * 78)
print('1. PHAN RA PHUONG SAI cua alpha_oracle (tren toan bo cap (mau, patch))')
print('-' * 78)
tot = P.var()
samp_mean = P.mean(1, keepdims=True)                   # [N,1] muc mau
v_within = (P - samp_mean).var()                       # trong mau, giua cac patch
v_sample = samp_mean.var()                             # giua cac mau
print(f'  tong phuong sai           = {tot:.5f}')
print(f'  giua MAU (muc bien do)    = {v_sample:.5f}  ({v_sample / tot * 100:5.1f}%)')
print(f'  trong MAU (giua patch)    = {v_within:.5f}  ({v_within / tot * 100:5.1f}%)')
if types is not None:
    tmean = np.zeros_like(samp_mean)
    for t in np.unique(types):
        tmean[types == t] = P[types == t].mean()
    v_type = tmean.var()
    print(f'    trong do giua LOAI      = {v_type:.5f}  ({v_type / tot * 100:5.1f}% tong)'
          f'  <-- day la tat ca nhung gi mot gate DIEU KIEN HOA THEO LOAI co the giai thich')
    print(f'    con lai giua mau cung loai = {v_sample - v_type:.5f}  '
          f'({(v_sample - v_type) / tot * 100:5.1f}%)')

# ── 2. CAU TRUC KHONG GIAN ──────────────────────────────────────────────────
print('\n' + '-' * 78)
print('2. CAU TRUC KHONG GIAN tren luoi 14x14')
print('-' * 78)
Gr = P.reshape(-1, G, G)
Gc = Gr - Gr.mean(axis=(1, 2), keepdims=True)          # bo bien do moi mau
num = (Gc[:, :, :-1] * Gc[:, :, 1:]).sum(axis=(1, 2)) + (Gc[:, :-1, :] * Gc[:, 1:, :]).sum(axis=(1, 2))
den = (Gc ** 2).sum(axis=(1, 2)) * 2 * G * (G - 1) / (G * G)
r_nb = np.divide(num / (2 * G * (G - 1)), den / (2 * G * (G - 1)) + 1e-12)
r_nb = np.clip(r_nb, -1, 1)
# doi chung: hoan vi patch trong tung mau -> pha cau truc khong gian, giu nguyen phan phoi
rng = np.random.default_rng(0)
Ps = np.take_along_axis(P, rng.permuted(np.tile(np.arange(196), (P.shape[0], 1)), axis=1), axis=1)
Gs = Ps.reshape(-1, G, G); Gsc = Gs - Gs.mean(axis=(1, 2), keepdims=True)
num_s = (Gsc[:, :, :-1] * Gsc[:, :, 1:]).sum(axis=(1, 2)) + (Gsc[:, :-1, :] * Gsc[:, 1:, :]).sum(axis=(1, 2))
den_s = (Gsc ** 2).sum(axis=(1, 2))
r_s = np.clip(np.divide(num_s, den_s * 2 * G * (G - 1) / (G * G) + 1e-12), -1, 1)
print(f'  tuong quan LANG GIENG (4-neighbour, da bo bien do moi mau):')
print(f'    alpha_oracle that    = {r_nb.mean():+.4f}  (sd {r_nb.std():.3f})')
print(f'    doi chung hoan vi    = {r_s.mean():+.4f}  <- neu that ~= doi chung thi alpha_oracle')
print(f'                                    la NHIEU per-patch, KHONG giam sat bang vung/box duoc')
ctr = np.zeros((G, G), bool); ctr[3:11, 3:11] = True
print(f'  trung tam (8x8 giua) mean={Gr[:, ctr].mean():.3f}   '
      f'vien mean={Gr[:, ~ctr].mean():.3f}   lech={Gr[:, ctr].mean() - Gr[:, ~ctr].mean():+.3f}')

# ── 3. DOI CHIEU VUNG COCO ──────────────────────────────────────────────────
print('\n' + '-' * 78)
print('3. alpha_oracle co bam vao VAT THE THAT (vung COCO) khong?')
print('-' * 78)
rm = pickle.load(open('patch_region_map.pkl', 'rb'))
ids = df['img_id'].astype(int).to_numpy()
d_obj, hit = [], 0
for i, iid in enumerate(ids):
    r = rm.get(int(iid))
    if r is None or r.shape[0] != 196:
        continue
    m = r > 0                                          # >0 = thuoc mot vung vat the COCO
    if m.sum() == 0 or (~m).sum() == 0:
        continue
    d_obj.append(P[i][m].mean() - P[i][~m].mean())
    hit += 1
d_obj = np.array(d_obj)
if hit:
    from math import sqrt
    t = d_obj.mean() / (d_obj.std(ddof=1) / sqrt(len(d_obj)))
    print(f'  n={hit} mau co annotation COCO')
    print(f'  alpha(patch VAT THE) - alpha(patch NEN) = {d_obj.mean():+.4f} '
          f'(sd {d_obj.std(ddof=1):.4f}, t={t:+.1f}, {(d_obj > 0).mean() * 100:.1f}% mau duong)')
    print(f'  -> duong ro: box/vung la proxy DUNG cho alpha_oracle (huong Box-Grounded co ly).')
    print(f'  -> ~0: alpha toi uu KHONG phai "vung vat the", box supervision se khong tai tao duoc.')
else:
    print('  khong khop duoc img_id nao voi patch_region_map.pkl')

# ── 4. THEO LOAI ────────────────────────────────────────────────────────────
if types is not None:
    print('\n' + '-' * 78)
    print('4. alpha_oracle theo LOAI cau hoi (truc ma TCVG dang dieu kien hoa)')
    print('-' * 78)
    print(f'  {"loai":<10} {"n":>5} {"mean":>7} {"sd trong mau":>13} {"%nen(<0.5)":>11} {"r lang gieng":>13}')
    for t in sorted(np.unique(types)):
        m = types == t
        print(f'  {TN.get(int(t), t):<10} {m.sum():>5} {P[m].mean():>7.3f} '
              f'{P[m].std(1).mean():>13.3f} {(P[m] < 0.5).mean() * 100:>10.1f}% {r_nb[m].mean():>+13.4f}')
    print('  Neu cac dong gan nhu giong nhau -> alpha toi uu KHONG phai ham cua loai:')
    print('  dieu kien hoa theo LOAI la truc SAI, phai dieu kien hoa theo NOI DUNG/mau.')
