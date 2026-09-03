"""MO 489 MAU LOI CO DINH — uoc luong TRAN NHIEU NHAN cua ViVQA.

Boi canh (do 2026-08-14 tren 20 he thong doc lap: T2 x10 seed + T0 x10 seed):
    dung co dinh (moi he thong dung)      1683 = 56.1%
    vo khong on dinh                       829 = 27.6%
    LOI CO DINH (khong he thong nao dung)  489 = 16.3%
Mot he thong dat 71.74% = 56.08 + 15.66 diem thang duoc trong vo. Tran neu thang sach vo = 83.71%.

Van de: 40+ co che cua du an nay deu tranh nhau trong CAI VO, noi churn cung-cau-hinh la 229.9 mau
trong khi net cua TCVG chi 8.5. Con 489 mau loi co dinh — phan bo DEU qua 4 loai (14.2-18.1%),
nen khong phai thieu nang luc theo loai — thi chua ai nhin vao.

Script nay phan 489 mau do thanh cac ro DEM DUOC, khong phong doan:
  A. gold NGOAI tap dap an train        -> khong the sinh ra duoc (trie + ghi nho deu chan)
  B. cau KHONG co tu nghi van           -> khong phai cau hoi (dau vet dich may)
  C. nhan type LECH cach doc mat chu    -> nhan khong nhat quan
  D. 20 he thong DONG THUAN mot dap an  -> gold rat dang ngo (20 he thong doc lap cung "sai" giong nhau)
  E. du doan PHAN TAN                   -> that su kho / nhieu nghia
  F. gold la dong nghia cua du doan     -> cham diem sai chu khong phai model sai

Ro D la quan trong nhat: neu 20 he thong doc lap deu cho CUNG mot dap an khac gold thi kha nang
cao la gold sai hoac ton tai mot cach doc hop ly khac, chu khong phai ca 20 cung nham giong nhau.
"""
import pandas as pd, numpy as np, sys, unicodedata as ud, importlib.util as ilu
from collections import Counter

M = '/home/user/workspace/nghia.duong/thesis'
S = [0, 1, 2, 3, 4, 5, 6, 7, 8, 42]
T = {0: 'OBJECT', 1: 'COUNT', 2: 'COLOR', 3: 'LOCATION'}
norm = lambda s: ud.normalize('NFC', str(s)).strip().lower()

sys.path.insert(0, f'{M}/src')
from dataset import detect_question_type
_sp = ilu.spec_from_file_location('_ev', f'{M}/src/eval.py')
_ev = ilu.module_from_spec(_sp); _sp.loader.exec_module(_ev)
SYN = _ev._SYNONYM_MAP

te = pd.read_csv(f'{M}/archive/test.csv')
tr = pd.read_csv(f'{M}/archive/train_split_original.csv')
TRAIN_ANS = {norm(a) for a in tr.answer}

# ---- 20 he thong ----
preds, oks = [], []
for s in S:
    for f in (f'{M}/beam3fixed/seed{s}_ep40.csv', f'{M}/analysis/T0/T0_seed{s}.csv'):
        d = pd.read_csv(f)
        preds.append(d.prediction.map(norm).values)
        oks.append(d.exact_match.values > .5)
P = np.stack(preds); OK = np.stack(oks)
core = np.where(OK.sum(0) == 0)[0]                  # loi co dinh
print(f'{P.shape[0]} he thong | loi co dinh: {len(core)} mau\n')

rows = []
for j in core:
    g = norm(te.answer[j]); q = str(te.question[j])
    c = Counter(P[:, j]); top, ntop = c.most_common(1)[0]
    rows.append(dict(
        idx=j, img=te.img_id[j], loai=T[te.type[j]], q=q, gold=g, top=top, ntop=ntop,
        n_uniq=len(c),
        A_gold_ngoai_train=g not in TRAIN_ANS,
        B_khong_cau_hoi=not any(m in q.lower() for m in
                                ['gì', 'đâu', 'nào', 'bao nhiêu', 'mấy', 'ai', 'sao', 'thế nào']),
        C_nhan_type_lech=detect_question_type(q) != te.type[j],
        D_dong_thuan=ntop >= 18,                    # >=18/20 he thong cung mot dap an
        E_phan_tan=len(c) >= 8,
        F_dong_nghia=SYN.get(top, top) == SYN.get(g, g),
    ))
R = pd.DataFrame(rows)
R.to_csv(f'{M}/analysis/hardcore_489.csv', index=False)

print('=== CAC RO (co the CHONG NHAU) ===')
for k, lab in [('A_gold_ngoai_train', 'gold NGOAI tap dap an train  (khong the sinh ra)'),
               ('B_khong_cau_hoi',    'cau KHONG co tu nghi van     (khong phai cau hoi)'),
               ('C_nhan_type_lech',   'nhan type LECH mat chu       (nhan khong nhat quan)'),
               ('D_dong_thuan',       '>=18/20 he thong DONG THUAN  (gold rat dang ngo)'),
               ('E_phan_tan',         'du doan PHAN TAN >=8 dap an  (kho / nhieu nghia)'),
               ('F_dong_nghia',       'gold la DONG NGHIA cua pred  (cham diem sai)')]:
    print(f'  {lab:48s} {R[k].sum():4d}  ({100*R[k].mean():5.1f}%)')

any_flag = R[['A_gold_ngoai_train', 'B_khong_cau_hoi', 'C_nhan_type_lech',
              'D_dong_thuan', 'F_dong_nghia']].any(axis=1)
print(f'\n  CO IT NHAT MOT DAU HIEU van de nhan/du lieu (A,B,C,D,F): {any_flag.sum()} '
      f'({100*any_flag.mean():.1f}% cua 489)')
print(f'  = {100*any_flag.sum()/len(te):.2f} diem phan tram cua toan tap test')
print(f'  KHONG dau hieu nao (co the that su kho)               : {(~any_flag).sum()} '
      f'({100*(~any_flag).mean():.1f}%)')

print(f'\n=== 15 vi du ro D (dong thuan cao nhat) — doc de kiem gold ===')
for _, r in R[R.D_dong_thuan].nlargest(15, 'ntop').iterrows():
    print(f'  [{r.loai:8s}] img {r.img}  {r.ntop}/20 he thong tra loi "{r.top}"  | gold="{r.gold}"')
    print(f'      {r.q[:88]}')
print(f'\n-> luu: {M}/analysis/hardcore_489.csv')
