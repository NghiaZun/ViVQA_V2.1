"""alpha_oracle la "chon patch lien quan" hay la KENH NGAM ma hoa DAP AN?

Kiem truc tiep: tu MOT MINH alpha (197 so) — KHONG anh, KHONG cau hoi — co doan duoc dap an khong?
  doan duoc >> majority  -> alpha CHUA thong tin dap an; +11.27pp cua E0 la tuon dap an qua kenh
                            alpha, khong phai chon bang chung tot hon. Khong ham nao cua (anh,
                            cau hoi) tai tao duoc, vi thong tin do khong nam trong (anh, cau hoi).
  ~ majority             -> alpha khong ma hoa dap an; gia thuyet kenh ngam sai.
Doi chung: xao tron alpha giua cac mau (pha lien ket alpha<->dap an, giu nguyen phan phoi).
"""
import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split

A = np.load('analysis/oracle_alpha/train_alpha.npz')['alpha'].astype('float32')
df = pd.read_csv('archive/train_split_original.csv')
y = pd.factorize(df['answer'].astype(str).str.strip().str.lower())[0]
t = df['type'].to_numpy()
NAME = {0: 'OBJECT', 1: 'COUNT', 2: 'COLOR', 3: 'LOCATION'}


def fit(X, yy, seed=0):
    Xa, Xb, ya, yb = train_test_split(X, yy, test_size=0.25, random_state=seed)
    m = SGDClassifier(loss='log_loss', max_iter=60, tol=None, random_state=0, alpha=1e-4)
    m.fit(Xa, ya)
    return m.score(Xb, yb), np.bincount(yb).max() / len(yb)


acc, maj = fit(A, y)
rng = np.random.default_rng(0)
acc_s, _ = fit(A[rng.permutation(len(A))], y)
print('Doan DAP AN chi tu alpha (197 so; KHONG anh, KHONG cau hoi):')
print(f'  accuracy        = {acc*100:6.2f}%   ({len(np.unique(y))} lop)')
print(f'  majority        = {maj*100:6.2f}%')
print(f'  alpha xao tron  = {acc_s*100:6.2f}%   <- doi chung')
print('\nTheo loai:')
for k in sorted(np.unique(t)):
    m = t == k
    a, mj = fit(A[m], y[m])
    print(f'  {NAME[k]:<9} acc={a*100:6.2f}%   majority={mj*100:6.2f}%   ({len(np.unique(y[m]))} lop)')
