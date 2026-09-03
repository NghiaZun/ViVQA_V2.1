"""KHONG GIAN DAU RA CO VOI TOI DUOC KHONG? — tra loi cau "neu visual on thi language se kiem duong chu".

QUAN SAT DAN TOI DAY (thi nghiem OOV): bo 5 dap an khoi train, mo hinh KHONG BAO GIO noi ra chung
(0/3001) DU trie cho phep. Nhung no thay the bang HANG XOM NGU NGHIA, moi lan:
    hươu cao cổ -> ngựa vằn (43/46) | màu cam -> màu đỏ, màu vàng | sáu -> năm, bốn
    phòng bếp -> phòng, quán ăn, lò vi sóng, chảo
=> NHIN thi dung. GOI TEN thi khong.

GIA THUYET: duong dan thi giac -> ngon ngu (vision_proj + fusion) chi duoc huan luyen tren tap dap
an cua train. No anh xa dac trung anh vao dung vung khong gian BARTpho ma cac dap an do chiem.
BARTpho BIET tu "màu hồng", nhung KHONG CO GI ANH XA TOI DO. Anh cua phep anh xa bi gioi han trong
bao cua cac dap an da thay -> moi thu ngoai bao la KHONG VOI TOI DUOC, du ngon ngu co biet tu do.

PHEP THU: cham diem log-likelihood cua 314 dap an DA HUAN LUYEN cong 19 tu tieng Viet PHO THONG
NHUNG CHUA TUNG duoc huan luyen (màu hồng, con hổ, phòng khách, mười hai, ...), tren cung mot anh.
Neu gia thuyet dung:
  - tu chua huan luyen phai xep gan CUOI, DEU DEU, khong phu thuoc noi dung anh
  - khoang cach diem giua chung va dap an da huan luyen phai LON va ON DINH
Neu gia thuyet SAI (ngon ngu "kiem duong"):
  - it nhat mot so anh phai day mot tu chua huan luyen len top, dung luc no phu hop

DOI CHUNG: so thu hang trung binh cua tu chua huan luyen voi thu hang trung binh cua nhung dap an
DA huan luyen nhung HIEM (xuat hien < 5 lan trong train). Neu hai nhom nay ngang nhau -> van de la
TAN SUAT, khong phai "chua tung thay". Neu tu chua thay te hon HAN ca nhom hiem -> co mot RANH GIOI
that su giua "da thay" va "chua thay", khong phai mot doc tan suat lien tuc.
"""
import sys
import time
import unicodedata as ud

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, 'src')
from model import DeterministicVQA          # noqa: E402
from dataset import VQAGenDataset           # noqa: E402
from transformers import AutoProcessor      # noqa: E402

CKPT = 'checkpoints_run87/best_model.pt'
N_SAMPLE = 400          # du cho mot phat bieu ve phan bo; 3001 khong can thiet
DEV = 'cuda'
CHUNK = 48
norm = lambda s: ud.normalize('NFC', str(s)).strip().lower()

NEVER = ['màu hồng', 'màu bạc', 'màu be', 'con hổ', 'con cá', 'con rắn', 'xe cứu hoả',
         'phòng khách', 'sân thượng', 'cầu thang', 'ban công', 'mười hai', 'mười lăm',
         'hai mươi', 'màu hồng nhạt', 'con sư tử', 'nhà thờ', 'bệnh viện', 'siêu thị']

tr = pd.read_csv('archive/train_split.csv')
tr['g'] = [norm(a) for a in tr.answer]
freq = tr.g.value_counts()
TRAINED = sorted(freq.index)
RARE = [a for a in TRAINED if freq[a] < 5]
assert not (set(NEVER) & set(TRAINED)), 'mot tu "chua huan luyen" that ra CO trong train'
CAND = TRAINED + [norm(x) for x in NEVER]
print(f'{len(TRAINED)} dap an da huan luyen (trong do {len(RARE)} hiem, <5 lan) '
      f'+ {len(NEVER)} tu chua tung huan luyen = {len(CAND)} ung vien')

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
tok = model.tokenizer

enc = tok(CAND, return_tensors='pt', padding='max_length', truncation=True, max_length=10)
LB = enc.input_ids.to(DEV).clone(); LB[LB == tok.pad_token_id] = -100
NTOK = (LB != -100).sum(1).float().clamp(min=1)

# CHOT: tu chua huan luyen phai duoc TOKEN HOA duoc, neu khong thi phep do vo nghia
# (mo hinh khong the sinh cai ma tokenizer khong bieu dien noi).
for i, w in enumerate(NEVER):
    ids = LB[len(TRAINED) + i]
    assert (ids != -100).sum() >= 2, f'"{w}" token hoa qua ngan/hong'
print('chot token hoa: 19/19 tu chua huan luyen deu bieu dien duoc\n')

vp = AutoProcessor.from_pretrained(sa.get('vision_model', 'google/siglip-base-patch16-224'))
ds = VQAGenDataset(csv_path='archive/test.csv', image_folder='archive/data/images/test',
                   vision_processor=vp, tokenizer_name='vinai/bartpho-syllable',
                   max_q_len=32, max_a_len=10, include_question_type=True, auto_detect_type=False)
rng = np.random.default_rng(0)
idx = sorted(rng.choice(len(ds), size=min(N_SAMPLE, len(ds)), replace=False).tolist())

nT, nN = len(TRAINED), len(NEVER)
rank_never, rank_rare, best_never_rank, gap = [], [], [], []
t0 = time.time()
for k, j in enumerate(idx):
    b = next(iter(DataLoader(Subset(ds, [j]), batch_size=1)))
    pv = b['pixel_values'].to(DEV); ii = b['input_ids'].to(DEV)
    am = b['attention_mask'].to(DEV)
    qt = b['question_type'].to(DEV).long() if b.get('question_type') is not None else None
    s = torch.empty(len(CAND), device=DEV)
    with torch.no_grad():
        for a in range(0, len(CAND), CHUNK):
            lb = LB[a:a + CHUNK]; n = lb.size(0)
            out = model(pixel_values=pv.expand(n, -1, -1, -1), input_ids=ii.expand(n, -1),
                        attention_mask=am.expand(n, -1), labels=lb,
                        question_types=None if qt is None else qt.expand(n))
            lg = out.answer_logits.float()
            lp = -F.cross_entropy(lg.reshape(-1, lg.size(-1)), lb.reshape(-1),
                                  ignore_index=-100, reduction='none').view(lb.shape)
            s[a:a + n] = lp.sum(1)
    sN = (s / NTOK).cpu().numpy()          # chia do dai: tu dai khong bi phat oan
    order = np.argsort(-sN)
    rank = np.empty(len(CAND), int); rank[order] = np.arange(1, len(CAND) + 1)
    rn = rank[nT:]
    rank_never.append(rn.mean()); best_never_rank.append(rn.min())
    rare_i = [TRAINED.index(a) for a in RARE]
    rank_rare.append(rank[rare_i].mean())
    gap.append(sN[:nT].max() - sN[nT:].max())
    if (k + 1) % 100 == 0:
        print(f'  {k+1}/{len(idx)} ({time.time()-t0:.0f}s)', flush=True)

rn = np.array(rank_never); rr = np.array(rank_rare); bn = np.array(best_never_rank)
print(f'\n{len(idx)} anh, {len(CAND)} ung vien\n')
print(f'thu hang trung binh, tu CHUA TUNG huan luyen : {rn.mean():7.1f} / {len(CAND)}')
print(f'thu hang trung binh, dap an DA hoc nhung HIEM: {rr.mean():7.1f} / {len(CAND)}   ({len(RARE)} tu, <5 lan trong train)')
print(f'thu hang TOT NHAT ma mot tu chua-thay dat duoc, trung binh tren cac anh: {bn.mean():.1f}')
print(f'   so anh co it nhat mot tu chua-thay lot TOP 10 : {int((bn <= 10).sum())}/{len(idx)}')
print(f'   so anh co it nhat mot tu chua-thay dung DAU   : {int((bn == 1).sum())}/{len(idx)}')
print(f'\nkhoang cach diem (dap an da hoc tot nhat - tu chua thay tot nhat): '
      f'{np.mean(gap):.3f} +/- {np.std(gap):.3f}')
print('\nDOC:')
print('  Neu tu chua-thay xep NGANG nhom hiem -> van de la TAN SUAT, co the vá bang du lieu.')
print('  Neu tu chua-thay te hon HAN nhom hiem -> co RANH GIOI that giua "da thay" va "chua thay":')
print('    ngon ngu biet tu do, nhung KHONG CO GI ANH XA TOI DO. Do la gioi han kien truc,')
print('    khong phai gioi han du lieu, va khong vá duoc bang them mau.')
