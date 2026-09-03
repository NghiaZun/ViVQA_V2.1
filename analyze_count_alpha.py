"""MAT NA alpha DUA COUNT LEN 90.32 CO CAU TRUC GI?

Bo canh (tach tran oracle theo loai, 2026-08-15, SigLIP1):
    loai       base   oracle-alpha   du dia   phan KHONG GIAN (perpatch - scalar)
    COUNT     66.22       90.32      +24.10          +13.74   <- lon nhat trong 4 loai
    COLOR     72.80       90.72      +17.92          +11.68
    LOCATION  71.09       77.37       +6.28           +5.25
    OBJECT    74.98       81.07       +6.09           +5.05
COUNT vua co du dia lon nhat, vua la loai te nhat (66.22), vua nhan it gradient nhat (.042).

Cau hoi: mat na do co HINH DANG gi. Neu no co cau truc thi co the co NHAN — thu ma gate_distill
thieu (nhan tren train thoai hoa vi model dung 99.77%) va da chet vi thieu.

GIA THUYET GHI TRUOC (de khong nhin so roi moi chon ket luan):
  H1 "dem duoc bang cach lam noi bat": so VUNG RIENG BIET cua alpha cao tuong quan DUONG voi
     dap an gold. Neu dung -> alpha oracle dang ma hoa truc tiep so luong -> co nhan hoc duoc.
  H2 "bam vao box": alpha cao trung voi vung co box COCO (IoU / precision-recall voi region_map).
     Neu dung -> box CO the giam sat alpha, du truoc day dem-tu-box that bai.
  H3 "chi la thu hep bien do": alpha oracle chi khac alpha model o MUC trung binh, khong khac o
     HINH DANG. Neu dung -> khong co cau truc khong gian, va +13.74 kia den tu dau khac.
  DOI CHUNG: so sanh MOI chi so tren voi alpha CUA MODEL (alpha_model.npy) tren cung mau.
     Chi ket luan khi oracle KHAC model, khong phai khi oracle chi co gia tri "hop ly".
"""
import numpy as np, pandas as pd, pickle, os
from scipy import stats
from scipy.ndimage import label as cc_label

M = '/home/user/workspace/nghia.duong/thesis'
d = np.load(f'{M}/analysis/oracle_count/alpha_oracle.npz')
A = d['alpha'].astype('float32')
if A.shape[1] == 197:
    A = A[:, 1:]                      # bo pooler token o dau
te = pd.read_csv(f'{M}/archive/test_count_only.csv')
base = pd.read_csv(f'{M}/analysis/oracle_count/base.csv')
orc = pd.read_csv(f'{M}/analysis/oracle_count/perpatch.csv')
W2N = {'không':0,'một':1,'hai':2,'ba':3,'bốn':4,'năm':5,'sáu':6,'bảy':7,'tám':8,'chín':9,'mười':10}
gold = te.answer.map(lambda a: W2N.get(str(a).strip().lower(), np.nan)).values
print(f'{len(A)} mau COUNT | alpha {A.shape} | base {base.exact_match.mean()*100:.2f} '
      f'-> oracle {orc.exact_match.mean()*100:.2f}')

MA = None
if os.path.exists(f'{M}/analysis/oracle_count/alpha_model.npy'):
    MA = np.load(f'{M}/analysis/oracle_count/alpha_model.npy').astype('float32')
    if MA.ndim == 2 and MA.shape[1] == 197: MA = MA[:, 1:]

def feats(X):
    """cac chi so hinh dang, tinh tren luoi 14x14"""
    g = X.reshape(-1, 14, 14)
    hi = g > (g.mean(axis=(1, 2), keepdims=True) + g.std(axis=(1, 2), keepdims=True))
    ncc = np.array([cc_label(h)[1] for h in hi])        # so vung rieng biet cua alpha cao
    area = hi.reshape(len(g), -1).sum(1)                 # dien tich vung cao
    return dict(so_vung=ncc, dien_tich=area,
                alpha_tb=X.mean(1), alpha_std=X.std(1),
                dai_dong=(g.max(2) - g.min(2)).mean(1))

print(f'\n{"chi so":14s} {"ORACLE vs gold":>18s} {"MODEL vs gold":>18s}')
FO, FM = feats(A), (feats(MA) if MA is not None else None)
ok = ~np.isnan(gold)
for k in FO:
    r1 = stats.spearmanr(FO[k][ok], gold[ok])
    s1 = f'{r1.correlation:+7.3f} (p={r1.pvalue:.1e})'
    s2 = '—'
    if FM is not None and len(FM[k]) == len(gold):
        r2 = stats.spearmanr(FM[k][ok], gold[ok]); s2 = f'{r2.correlation:+7.3f} (p={r2.pvalue:.1e})'
    print(f'{k:14s} {s1:>18s} {s2:>18s}')

print('\nH1 (dem bang cach lam noi bat): "so_vung" tuong quan DUONG voi gold, va MANH HON o oracle.')
print('H3 (chi thu hep bien do): chi "alpha_tb" co tuong quan, cac chi so HINH DANG thi khong.')

# H2: alpha cao co trung vung box COCO khong
rm_path = f'{M}/patch_region_map.pkl'
if os.path.exists(rm_path):
    RM = pickle.load(open(rm_path, 'rb'))
    ious, n_ok = [], 0
    for i, img in enumerate(te.img_id.values):
        r = RM.get(int(img))
        if r is None: continue
        r = np.asarray(r).ravel()[:196]
        if r.size < 196: continue
        obj = r > 0
        hi = A[i] > (A[i].mean() + A[i].std())
        inter = (obj & hi).sum(); union = (obj | hi).sum()
        if union: ious.append(inter / union); n_ok += 1
    if ious:
        print(f'\nH2 (bam vao box): IoU(alpha cao, vung box) = {np.mean(ious):.3f} '
              f'+/- {np.std(ious):.3f}  (n={n_ok})')
        print('   IoU ~ 0.2 la muc ngau nhien khi hai mat na cung chiem ~16% dien tich.')
else:
    print('\nH2: khong co patch_region_map.pkl')
