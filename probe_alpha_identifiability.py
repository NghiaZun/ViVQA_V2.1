"""PHEP THU QUYET DINH: alpha ORACLE per-sample co DOAN DUOC tu dau vao ma CONG NHIN THAY khong?

Tran oracle la THAT (scalar +3.50, per-patch +11.27). Cau hoi khong phai "co du dia khong" ma
"du dia do co DIA CHI khong" — tu (anh, cau hoi) co suy ra duoc alpha toi uu khong.
  CO     -> huong nay MO LAI, voi co che ro rang: chung cat probe nay vao cong.
  KHONG  -> gioi han duoc xac nhan bang phep thu manh nhat co the.

DU LIEU (da sua sau mot lan chon nham file):
  nhan  = analysis/valoracle/val_alpha.npz — oracle SCALAR, sinh boi CHINH run87, tren VAL 1199 mau
          (canh bao: analysis/*alpha_best*.npz KHONG phai oracle, do la alpha HANG tot nhat —
           s1_alpha_best la hang so 0.75 o moi o. Ten file danh lua.)
  dac trung = DUNG tensor dau vao cua gate_net, hook truc tiep tren cung model, cung tap val

RONG RAI NHAT CO THE cho gia thuyet "doan duoc": dac trung that, probe quet nhieu do phuc tap,
danh gia bang CV ngay tren chinh tap do (in-distribution = CAN TREN cua tinh nhan dang duoc).

BON DOI CHUNG:
  1. DOI CHUNG DUONG YEU: doan LOAI CAU HOI. Phai ~1.0 vi type_emb nam THANG trong dau vao cong.
     Chi kiem duong ong, khong noi len dieu gi ve do giau cua dac trung.
  2. DOI CHUNG DUONG MANH (quyet dinh): doan ALPHA CUA CHINH MO HINH.
     Alpha do duoc tinh TU DUNG NHUNG DAC TRUNG NAY boi mot MLP 2 lop. Probe BAT BUOC phai
     phuc hoi duoc no. Neu phuc hoi duoc alpha-mo-hinh gan hoan hao MA khong doan duoc
     alpha-oracle -> khong phai dac trung ngheo, khong phai probe yeu. Thong tin KHONG CO O DO.
  3. XAO NHAN trong tung loai -> phai ve 0.50.
  4. NHI PHAN HOA THEO TRUNG VI CUA CHINH LOAI -> bo sach phan per-type (da biet vo gia tri,
     ptnet xac nhan), chi giu phan PER-SAMPLE — dung phan mang gia tri.

TIEU CHI GHI TRUOC: AUC > 0.65 -> MO LAI. ~0.53 -> DONG.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, 'src')
from model import DeterministicVQA          # noqa: E402
from dataset import VQAGenDataset           # noqa: E402
from transformers import AutoProcessor      # noqa: E402

CKPT = 'checkpoints_run87/best_model.pt'
ORACLE = 'analysis/valoracle/val_alpha.npz'
CSV, IMG = 'archive/val_split.csv', 'archive/data/images/train'
CACHE = 'analysis/gate_input_feats_val.npz'
DEV = 'cuda'
TM = {0: 'OBJECT', 1: 'COUNT', 2: 'COLOR', 3: 'LOCATION'}

if os.path.exists(CACHE):
    z = np.load(CACHE)
    V, Q, TY, AM = z['v'], z['q'], z['ty'], z['am']
    print(f'dung cache {CACHE}: v{V.shape}')
else:
    ck = torch.load(CKPT, map_location='cpu', weights_only=False)
    sa = ck.get('args', {})
    model = DeterministicVQA(
        vision_model_name=sa.get('vision_model', 'google/siglip-base-patch16-224'),
        bartpho_model_name='vinai/bartpho-syllable',
        num_fusion_layers=sa.get('num_fusion_layers', 2), fusion_type='text2vision',
        use_text_lora=True, text_lora_r=16, text_lora_alpha=32,
        use_vision_gate=True, vision_gate_init=sa.get('vision_gate_init', 1.0),
        vision_gate_min_alpha=0.0, use_type_task=sa.get('use_type_loss', True),
        use_siglip_pooler=sa.get('use_siglip_pooler', True)).to(DEV).eval()
    model.load_state_dict(ck['model_state_dict'], strict=False)
    for p in model.parameters():
        p.requires_grad_(False)

    grab = {}
    h = model.vision_gating.gate_net.register_forward_pre_hook(
        lambda _m, inp: grab.__setitem__('x', inp[0].detach().float().cpu()))

    vp = AutoProcessor.from_pretrained(sa.get('vision_model', 'google/siglip-base-patch16-224'))
    ds = VQAGenDataset(csv_path=CSV, image_folder=IMG, vision_processor=vp,
                       tokenizer_name='vinai/bartpho-syllable', max_q_len=32, max_a_len=10,
                       include_question_type=True, auto_detect_type=False)
    dl = DataLoader(ds, batch_size=16, shuffle=False, num_workers=2)
    vs, qs, tys, ams = [], [], [], []
    t0 = time.time()
    for i, b in enumerate(dl):
        pv = b['pixel_values'].to(DEV); ii = b['input_ids'].to(DEV)
        am_ = b['attention_mask'].to(DEV); qt = b['question_type'].to(DEV).long()
        with torch.no_grad():
            model(pixel_values=pv, input_ids=ii, attention_mask=am_, labels=None,
                  question_types=qt)
        x = grab['x']
        D = x.size(-1) // 2
        v, q = x[..., :D], x[..., D:]
        vs.append(torch.cat([v.mean(1), v.std(1), v.amax(1)], -1).numpy())
        qs.append(q[:, 0, :].numpy())                       # query dong nhat moi patch
        _a = model.vision_gating.last_alpha.detach().float().cpu()
        ams.append((_a.mean(-1) if _a.dim() == 3 else _a).mean(-1).numpy())   # alpha cua CHINH model
        tys.append(b['question_type'].numpy())
        if (i + 1) % 25 == 0:
            print(f'  {(i+1)*16}/{len(ds)} ({time.time()-t0:.0f}s)', flush=True)
    h.remove()
    V = np.concatenate(vs); Q = np.concatenate(qs)
    TY = np.concatenate(tys); AM = np.concatenate(ams)
    np.savez_compressed(CACHE, v=V, q=Q, ty=TY, am=AM)
    print(f'da luu {CACHE}')

z = np.load(ORACLE)
oa = np.asarray(z['alpha'], dtype=np.float32).reshape(-1)
print(f'\noracle scalar: n={len(oa)} min {oa.min():.4f} max {oa.max():.4f} mean {oa.mean():.4f} '
      f'| so gia tri khac nhau {len(np.unique(oa))}')
assert len(np.unique(oa)) > 100, 'nhan gan nhu hang so -> KHONG phai oracle, dung lai'

vs_csv = pd.read_csv(CSV)
assert len(vs_csv) == len(oa) == len(V), f'lech do dai: {len(vs_csv)} {len(oa)} {len(V)}'
if 'img_id' in z.files:
    assert (np.asarray(z['img_id']) == vs_csv.img_id.values).all(), 'thu tu hang KHONG khop csv'
    print('kiem thu tu hang theo img_id: KHOP')

X = np.concatenate([V, Q], 1).astype(np.float32)
print(f'dac trung: {X.shape[1]} chieu tren {X.shape[0]} mau')


def binz(vals):
    """Nhi phan hoa TRONG TUNG LOAI -> bo phan per-type, chi giu per-sample."""
    y = np.zeros(len(vals), dtype=int)
    for t in TM:
        m = TY == t
        y[m] = (vals[m] > np.median(vals[m])).astype(int)
    return y


y_or, y_md = binz(oa), binz(AM)
print('can bang nhan oracle trong tung loai:',
      {TM[t]: round(float(y_or[TY == t].mean()), 3) for t in TM})

from sklearn.linear_model import LogisticRegression      # noqa: E402
from sklearn.neural_network import MLPClassifier         # noqa: E402
from sklearn.pipeline import make_pipeline               # noqa: E402
from sklearn.preprocessing import StandardScaler         # noqa: E402
from sklearn.model_selection import StratifiedKFold      # noqa: E402
from sklearn.metrics import roc_auc_score                # noqa: E402


def cv_auc(X, y, groups=None, kind='lr', C=1.0, seed=0):
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    strat = y if groups is None else y * 10 + groups
    oof = np.zeros(len(y))
    for tr, va in skf.split(X, strat):
        mdl = (make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))
               if kind == 'lr' else
               make_pipeline(StandardScaler(),
                             MLPClassifier(hidden_layer_sizes=(256,), alpha=C,
                                           max_iter=500, random_state=seed)))
        mdl.fit(X[tr], y[tr])
        oof[va] = mdl.predict_proba(X[va])[:, 1]
    if groups is None:
        return roc_auc_score(y, oof)
    a, w = [], []
    for g in np.unique(groups):
        m = groups == g
        if len(np.unique(y[m])) > 1:
            a.append(roc_auc_score(y[m], oof[m])); w.append(m.sum())
    return float(np.average(a, weights=w))


def sweep(y, groups, tag):
    best, arg = 0.0, ''
    for kind, Cs in (('lr', (0.003, 0.01, 0.1, 1.0)), ('mlp', (1.0, 10.0))):
        for C in Cs:
            v = cv_auc(X, y, groups=groups, kind=kind, C=C)
            print(f'   {kind:4s} C={C:<7} AUC = {v:.4f}')
            if v > best:
                best, arg = v, f'{kind} C={C}'
    print(f'   >> {tag}: {best:.4f} ({arg})')
    return best


print('\n' + '=' * 72)
print('DOI CHUNG DUONG YEU — doan LOAI (type_emb nam THANG trong dau vao cong, phai ~1.0)')
p_weak = cv_auc(X, (TY == 2).astype(int), kind='lr', C=0.01)
print(f'   AUC(COLOR vs con lai) = {p_weak:.4f}')

print('\n' + '=' * 72)
print('DOI CHUNG DUONG MANH — doan ALPHA CUA CHINH MO HINH (trong tung loai)')
print('(alpha do duoc tinh TU DUNG dac trung nay boi MLP cua cong -> probe PHAI phuc hoi duoc)')
p_strong = sweep(y_md, TY, 'doi chung duong manh')

print('\n' + '=' * 72)
print('PHEP THU CHINH — doan ALPHA ORACLE per-sample (trong tung loai)')
best = sweep(y_or, TY, 'PROBE TOT NHAT')

print('\n' + '=' * 72)
print('DOI CHUNG XAO NHAN — phai ve ~0.50')
rng = np.random.default_rng(0)
ysh = y_or.copy()
for t in TM:
    m = np.where(TY == t)[0]
    ysh[m] = y_or[m][rng.permutation(len(m))]
a_sh = cv_auc(X, ysh, groups=TY, kind='lr', C=0.01)
print(f'   AUC nhan xao = {a_sh:.4f}   {"OK" if abs(a_sh-0.5) < 0.06 else "!! RO RI"}')

print('\n' + '=' * 72)
print(f'duong-yeu {p_weak:.4f} | duong-MANH {p_strong:.4f} | ORACLE {best:.4f} | xao {a_sh:.4f}')
if p_strong < 0.80:
    print('!! DOI CHUNG DUONG MANH THAT BAI — probe khong phuc hoi noi ca alpha cua chinh model,')
    print('   von la mot ham cua DUNG nhung dac trung nay. DUONG ONG YEU, khong duoc ket luan.')
    sys.exit(1)
print('TIEU CHI GHI TRUOC: AUC > 0.65 -> MO LAI. ~0.53 -> DONG.')
print(f'KET LUAN: {"MO LAI — co tin hieu" if best > 0.65 else "DONG"}')
if best <= 0.65:
    print(f'  Cung bo dac trung, cung probe: alpha-CUA-MODEL {p_strong:.4f} vs alpha-ORACLE {best:.4f}.')
    print('  -> khong phai dac trung ngheo, khong phai probe yeu.')
    print('  Tran oracle la THAT nhung duoc danh chi muc bang DAP AN, khong bang dau vao.')
