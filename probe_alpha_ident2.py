"""VONG 2 — sua khiem khuyet cua vong 1 va phu ca BON loai.

VONG 1 cat nhan theo TRUNG VI trong tung loai. Voi OBJECT va LOCATION, alpha oracle BAO HOA o 1.0
cho qua nua so mau -> "> trung vi" cho ra mot lop duy nhat -> hai loai do bi LOAI khoi phep tinh,
va AUC 0.5735 that ra chi do tren COUNT + COLOR.

Vong nay dung cach cat dung voi quyet dinh ma cong thuc su phai ra:
    nhan = 1 neu alpha_oracle < 1 - eps   ("mau nay CAN nen thi giac")
           0 neu alpha_oracle ~ 1         ("de nguyen")
Cat nay khong suy bien o bat ky loai nao va co nghia truc tiep: neu doan duoc nhan nay thi cong
biet luc nao nen dong — dung thu no can biet.

Bao cao them AUC TUNG LOAI de thay ro loai nao co tin hieu, va he so tuong quan Spearman cho
ban hoi quy (khong phu thuoc cach cat nhan).

TIEU CHI GHI TRUOC (giu nguyen vong 1): AUC > 0.65 -> MO LAI. ~0.53 -> DONG.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TM = {0: 'OBJECT', 1: 'COUNT', 2: 'COLOR', 3: 'LOCATION'}
z = np.load('analysis/gate_input_feats_val.npz')
V, Q, TY, AM = z['v'], z['q'], z['ty'], z['am']
oa = np.asarray(np.load('analysis/valoracle/val_alpha.npz')['alpha'], np.float32).reshape(-1)
X = np.concatenate([V, Q], 1).astype(np.float32)
print(f'{X.shape[0]} mau, {X.shape[1]} chieu\n')

print('phan bo alpha oracle theo loai (vi sao cat trung vi hong o OBJECT/LOCATION):')
print(f"{'loai':10s} {'n':>5s} {'trung vi':>9s} {'ty le =1.0':>11s} {'ty le <0.99':>12s}")
for t in TM:
    m = TY == t
    print(f'{TM[t]:10s} {m.sum():5d} {np.median(oa[m]):9.4f} '
          f'{100*(oa[m] > 0.999).mean():10.1f}% {100*(oa[m] < 0.99).mean():11.1f}%')

y = (oa < 0.99).astype(int)          # 1 = mau nay CAN nen thi giac
print('\ncan bang nhan moi:', {TM[t]: round(float(y[TY == t].mean()), 3) for t in TM})


def cv_oof(X, y, groups, kind='lr', C=1.0, seed=0):
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    oof = np.zeros(len(y))
    for tr, va in skf.split(X, y * 10 + groups):
        mdl = (make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))
               if kind == 'lr' else
               make_pipeline(StandardScaler(),
                             MLPClassifier(hidden_layer_sizes=(256,), alpha=C,
                                           max_iter=500, random_state=seed)))
        mdl.fit(X[tr], y[tr])
        oof[va] = mdl.predict_proba(X[va])[:, 1]
    return oof


def within(oof, y, groups):
    a, w = [], []
    for g in np.unique(groups):
        m = groups == g
        if len(np.unique(y[m])) > 1:
            a.append(roc_auc_score(y[m], oof[m])); w.append(m.sum())
    return float(np.average(a, weights=w)), len(a)


print('\n' + '=' * 70)
print('DOI CHUNG DUONG MANH — doan alpha CUA CHINH MODEL, cung cach cat')
ym = (AM < np.median(AM)).astype(int)
best_s = 0.0
for kind, C in (('lr', 0.01), ('mlp', 10.0)):
    o = cv_oof(X, ym, TY, kind, C)
    v, k = within(o, ym, TY)
    print(f'   {kind:4s} AUC trong-loai = {v:.4f} ({k}/4 loai)')
    best_s = max(best_s, v)

print('\n' + '=' * 70)
print('PHEP THU CHINH — doan "mau nay CAN nen thi giac" (alpha_oracle < 0.99)')
best, best_oof = 0.0, None
for kind, Cs in (('lr', (0.003, 0.01, 0.1)), ('mlp', (1.0, 10.0))):
    for C in Cs:
        o = cv_oof(X, y, TY, kind, C)
        v, k = within(o, y, TY)
        g = roc_auc_score(y, o)
        print(f'   {kind:4s} C={C:<7} trong-loai {v:.4f} ({k}/4 loai) | gop chung {g:.4f}')
        if v > best:
            best, best_oof = v, o

print('\n   AUC TUNG LOAI (probe tot nhat):')
for t in TM:
    m = TY == t
    if len(np.unique(y[m])) > 1:
        print(f'      {TM[t]:10s} n={m.sum():4d}  AUC {roc_auc_score(y[m], best_oof[m]):.4f}')
    else:
        print(f'      {TM[t]:10s} n={m.sum():4d}  (mot lop, bo qua)')

rho, p = spearmanr(best_oof, oa)
print(f'\n   Spearman(diem probe, alpha oracle lien tuc) = {rho:+.4f} (p={p:.3g})')

print('\n' + '=' * 70)
rng = np.random.default_rng(0)
ysh = y.copy()
for t in TM:
    m = np.where(TY == t)[0]
    ysh[m] = y[m][rng.permutation(len(m))]
o = cv_oof(X, ysh, TY, 'lr', 0.01)
v_sh, _ = within(o, ysh, TY)
print(f'DOI CHUNG XAO NHAN = {v_sh:.4f}   {"OK" if abs(v_sh-0.5) < 0.06 else "!! RO RI"}')

print('\n' + '=' * 70)
print(f'doi chung duong MANH {best_s:.4f} | ORACLE {best:.4f} | xao {v_sh:.4f}')
print('TIEU CHI GHI TRUOC: > 0.65 -> MO LAI. ~0.53 -> DONG.')
print(f'KET LUAN: {"MO LAI" if best > 0.65 else "DONG"}')
