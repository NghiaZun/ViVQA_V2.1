"""BBOX CO CHUA DUOC DAP AN COUNT KHONG? — chan tren cua ca huong bbox, khong can GPU.

Cau hoi cua tac gia: dem sai thi bbox phai giup chu, sao 4 duong bbox deu am
    (mat na alpha -0.64 | token GCA -0.97 | nhan lop -1.47 | logit alpha -0.03 n=3)

Truoc khi do co che, phai do THONG TIN: annotation box co chua dap an COUNT khong.
Do bang chan tren ORACLE — gia su model biet HOAN HAO cau hoi dang dem lop COCO nao:

  oracle_any  : co TON TAI mot lop COCO nao trong anh co so luong dung bang gold khong?
                Day la chan tren TUYET DOI. Neu thap thi bbox khong the tra loi COUNT,
                va moi co che deu vo ich — khong phai loi kien truc.
  tong so box : gold co bang TONG so box khong (truong hop don gian nhat)

Neu oracle_any cao ma cac lan train deu am -> loi la o CO CHE (cach nhet thong tin vao).
Neu oracle_any thap                        -> loi la o THONG TIN, dong ca huong.
"""
import pandas as pd, numpy as np, pickle, unicodedata as ud, re
from collections import Counter

norm = lambda s: ud.normalize('NFC', str(s)).strip().lower()
W2N = {'không': 0, 'một': 1, 'hai': 2, 'ba': 3, 'bốn': 4, 'năm': 5, 'sáu': 6,
       'bảy': 7, 'tám': 8, 'chín': 9, 'mười': 10, 'mười một': 11, 'mười hai': 12}

def to_num(a):
    a = norm(a)
    if a in W2N:
        return W2N[a]
    m = re.fullmatch(r'\d+', a)
    return int(m.group()) if m else None

ann = pickle.load(open('coco_annotations_matched.pkl', 'rb'))
te = pd.read_csv('archive/test.csv')
cnt = te[te.type == 1].copy()
cnt['gold_n'] = cnt.answer.map(to_num)
print(f'{len(cnt)} cau COUNT trong test | doc duoc thanh so: {cnt.gold_n.notna().sum()}')
print(f'phan bo gold: {Counter(cnt.gold_n.dropna().astype(int)).most_common()}')

have = cnt[cnt.gold_n.notna() & cnt.img_id.isin(ann)].copy()
print(f'co ca annotation va gold so: {len(have)}')

oracle_any, total_eq, best_err, n_cls = [], [], [], []
for _, r in have.iterrows():
    cats = np.asarray(ann[int(r.img_id)]['category'])
    per = Counter(cats.tolist())
    g = int(r.gold_n)
    oracle_any.append(int(g in per.values()))
    total_eq.append(int(len(cats) == g))
    best_err.append(min(abs(v - g) for v in per.values()) if per else g)
    n_cls.append(len(per))

have['oracle_any'] = oracle_any
print(f'\n=== CHAN TREN ===')
print(f'  co MOT lop COCO dem dung bang gold : {np.mean(oracle_any)*100:5.2f}%   <-- chan tren tuyet doi')
print(f'  TONG so box bang gold              : {np.mean(total_eq)*100:5.2f}%')
print(f'  sai lech nho nhat qua cac lop      : {np.mean(best_err):5.2f} (trung binh)')
print(f'  so lop COCO trung binh moi anh     : {np.mean(n_cls):5.2f}')

# doi chung: neu chon NGAU NHIEN mot so trong 1..12 thi trung bao nhieu
g = have.gold_n.astype(int).values
print(f'\n=== DOI CHUNG ===')
print(f'  luon doan "2" (dap an pho bien nhat): {(g==2).mean()*100:5.2f}%')
print(f'  doan ngau nhien 1..12               : {np.mean([np.mean(g==k) for k in range(1,13)])*100:5.2f}%')

# model hien tai lam duoc bao nhieu tren chinh tap nay
ev = pd.read_csv('checkpoints_s2_T2/eval_last.csv')
if len(ev) == len(te):
    m = (te.type == 1).values & te.img_id.isin(ann).values & cnt.gold_n.notna().reindex(te.index, fill_value=False).values
    print(f'\n  model hien tai tren dung tap nay    : {ev.exact_match.values[m].mean()*100:5.2f}%')

print('\n=> Neu chan tren oracle THAP hon nhieu so voi model hien tai thi bbox KHONG CHUA')
print('   dap an COUNT, va 4 ket qua am kia la do THONG TIN chu khong phai co che.')
