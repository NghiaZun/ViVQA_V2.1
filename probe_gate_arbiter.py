"""TRONG TAI CONG: khi cong BAT va cong TAT cho hai dap an khac nhau, chon duoc cai dung khong?

SU THAT DA DO (offline, 9 seed, khong ton GPU):
    T0 va T2 bat dong 393 cau/seed (13.1% test)
    T2 sua 122, pha 114  ->  net +0.24  (khop dung ablation co kiem soat)
    ORACLE chon dung tren tap bat dong: T2 + 3.81 EM
Tuc la +3.81 diem nam san trong HAI dap an mo hinh DA sinh ra, chi la chon nham.

CHUA AI HOI: tin hieu tin cay co phan thang bai giua HAI BIEN THE khong.
Da biet margin (top1 - top2) tach DUNG khoi SAI voi AUROC 0.756. Day la cau hoi KHAC:
khong phai "cau nay dung khong" ma "trong hai cau tra loi nay cai nao dung".

VI SAO KHAC voi ttselect DA THAT BAI (chon 1 trong 5 muc alpha bang log-prob, -0.07):
  1. NHI PHAN, khong phai 5 lua chon
  2. dung MARGIN chu khong phai log-prob tho — log-prob giua cac muc alpha khac nhau khong
     duoc hieu chuan cung thang do, con margin la phep so sanh TRONG cung mot phan bo
  3. hai bien the deu la ham hop le cua CHINH mo hinh do, khong phai 5 muc do cuong do

RANH GIOI CAN NEU RO: day la MOT bo trong so, chay hai cau hinh alpha roi chon — mot chinh sach
giai ma, khong phai ensemble hai mo hinh. Nhung nguoi phan bien co the doc khac; phai neu thang.

CAN BAO TRUOC ve che do do: script nay xep hang TOAN BO tap ung vien bang log-likelihood,
khong dung beam+trie nhu he da trien khai. Neu margin KHONG phan thang bai duoc o che do sach
nay thi cang khong the o che do trien khai -> day la phep thu SANG LOC dung huong.
"""
import sys, os, time, torch, pandas as pd, numpy as np, unicodedata as ud
import torch.nn.functional as F
sys.path.insert(0, 'src')
from model import DeterministicVQA
from dataset import VQAGenDataset
from torch.utils.data import DataLoader, Subset
from transformers import AutoProcessor

CKPT = 'checkpoints_run87/best_model.pt'          # SigLIP1, he da cong bo 72.34
OUT = 'analysis/gate_arbiter.csv'
DEV = 'cuda'; CHUNK = 32; NEED_MIB = 11000
norm = lambda s: ud.normalize('NFC', str(s)).strip().lower()


def free_mib():
    import subprocess
    o = subprocess.run(['nvidia-smi', '--query-gpu=memory.total,memory.used',
                        '--format=csv,noheader,nounits'], capture_output=True, text=True).stdout
    t, u = [int(x) for x in o.strip().split('\n')[0].split(', ')]
    return t - u


while free_mib() < NEED_MIB:
    print(f'GPU con {free_mib()}MiB, doi 5 phut', flush=True)
    time.sleep(300)

tr = pd.read_csv('archive/train_split_original.csv')
CAND = sorted({norm(a) for a in tr.answer})
print(f'{len(CAND)} ung vien', flush=True)

ck = torch.load(CKPT, map_location='cpu', weights_only=False); sa = ck.get('args', {})
model = DeterministicVQA(
    vision_model_name=sa.get('vision_model', 'google/siglip-base-patch16-224'),
    bartpho_model_name='vinai/bartpho-syllable',
    num_fusion_layers=sa.get('num_fusion_layers', 2), fusion_type='text2vision',
    use_text_lora=True, text_lora_r=16, text_lora_alpha=32,
    use_vision_gate=True, vision_gate_init=sa.get('vision_gate_init', 1.0),
    vision_gate_min_alpha=0.0, use_type_task=sa.get('use_type_loss', True),
    use_siglip_pooler=sa.get('use_siglip_pooler', True)).to(DEV).eval()
model.load_state_dict(ck['model_state_dict'], strict=False)
for p in model.parameters(): p.requires_grad_(False)
tok = model.tokenizer

# KIEM CHUNG alpha_override truoc khi do 3001 mau: ep alpha=1 PHAI doi dau ra, va phai
# tra lai nguyen trang. Neu no tro thi ca phep do nay do "T0" giong het T2 va ket luan vo nghia.
_vg = model.vision_gating
assert _vg is not None, 'khong tim thay vision_gating'

enc = tok(CAND, return_tensors='pt', padding='max_length', truncation=True, max_length=10)
LB = enc.input_ids.to(DEV).clone(); LB[LB == tok.pad_token_id] = -100
NTOK = (LB != -100).sum(1).float()

te = pd.read_csv('archive/test.csv')
vp = AutoProcessor.from_pretrained(sa.get('vision_model', 'google/siglip-base-patch16-224'))
ds = VQAGenDataset(csv_path='archive/test.csv', image_folder='archive/data/images/test',
                   vision_processor=vp, tokenizer_name='vinai/bartpho-syllable',
                   max_q_len=32, max_a_len=10, include_question_type=True, auto_detect_type=False)


def score_all(pv, ii, am, qt, force_alpha_one):
    """Log-likelihood cua moi ung vien. force_alpha_one=True -> TCVG thanh anh xa dong nhat (T0)."""
    prev = getattr(_vg, 'alpha_override', None)
    try:
        _vg.alpha_override = (torch.ones(1, 1, device=DEV) if force_alpha_one else None)
        sums = torch.empty(len(CAND), device=DEV)
        with torch.no_grad():
            for s in range(0, len(CAND), CHUNK):
                lb = LB[s:s + CHUNK]; k = lb.size(0)
                out = model(pixel_values=pv.expand(k, -1, -1, -1),
                            input_ids=ii.expand(k, -1), attention_mask=am.expand(k, -1),
                            labels=lb, question_types=None if qt is None else qt.expand(k))
                lg = out.answer_logits.float()
                lp = -F.cross_entropy(lg.reshape(-1, lg.size(-1)), lb.reshape(-1),
                                      ignore_index=-100, reduction='none').view(lb.shape)
                sums[s:s + k] = lp.sum(1)
        return sums
    finally:
        _vg.alpha_override = prev


# CHOT BAT BUOC: alpha_override phai THUC SU doi dau ra. Neu no tro thi "T0" do duoc se giong
# het T2, tap bat dong ve 0, va ket luan "khong co tin hieu" se la hien vat cai dat chu khong
# phai ket qua. Da tung mat mot luot GPU vi mot co khong duoc noi day.
_b = next(iter(DataLoader(Subset(ds, [0]), batch_size=1)))
_pv = _b['pixel_values'].to(DEV); _ii = _b['input_ids'].to(DEV); _am = _b['attention_mask'].to(DEV)
_qt = _b['question_type'].to(DEV).long() if _b.get('question_type') is not None else None
_s2 = score_all(_pv, _ii, _am, _qt, False); _s0 = score_all(_pv, _ii, _am, _qt, True)
_dif = (_s2 - _s0).abs().max().item()
print(f'chot alpha_override: |chenh lech log-lik| lon nhat = {_dif:.6f}')
assert _dif > 1e-4, 'alpha_override KHONG doi dau ra -> phep do vo nghia, dung lai'
_s2b = score_all(_pv, _ii, _am, _qt, False)
assert (_s2 - _s2b).abs().max().item() < 1e-4, 'trang thai khong duoc tra lai sau probe'
print('chot alpha_override: QUA (co tac dung, va tra lai nguyen trang)\n', flush=True)

rows = []
t0 = time.time()
for j in range(len(ds)):
    b = next(iter(DataLoader(Subset(ds, [j]), batch_size=1)))
    pv = b['pixel_values'].to(DEV); ii = b['input_ids'].to(DEV); am = b['attention_mask'].to(DEV)
    qt = b['question_type'].to(DEV).long() if b.get('question_type') is not None else None
    g = norm(te.answer.iloc[j])
    rec = dict(idx=j, type=te.type.iloc[j], gold=g)
    for tag, force in (('t2', False), ('t0', True)):
        s = score_all(pv, ii, am, qt, force)
        o = torch.argsort(s, descending=True)
        rec[f'{tag}_pick'] = CAND[int(o[0])]
        rec[f'{tag}_hit'] = int(CAND[int(o[0])] == g)
        rec[f'{tag}_top1'] = float(s[o[0]])
        rec[f'{tag}_margin'] = float(s[o[0]] - s[o[1]])
        rec[f'{tag}_pgold'] = float(s[CAND.index(g)]) if g in CAND else float('nan')
    rows.append(rec)
    if (j + 1) % 250 == 0:
        el = time.time() - t0
        print(f'  {j+1}/{len(ds)}  ({el/60:.1f} phut, con ~{el/(j+1)*(len(ds)-j-1)/60:.0f} phut)', flush=True)

d = pd.DataFrame(rows)
os.makedirs('analysis', exist_ok=True)
d.to_csv(OUT, index=False)
print(f'\nda luu {OUT} ({len(d)} dong)\n')

# ── DOC KET QUA ─────────────────────────────────────────────────────────────
dis = d.t2_pick.values != d.t0_pick.values
h2, h0 = d.t2_hit.values.astype(bool), d.t0_hit.values.astype(bool)
n = len(d)
print(f'T0 {100*h0.mean():.2f} | T2 {100*h2.mean():.2f} | bat dong {dis.sum()} ({100*dis.mean():.1f}%)')
print(f'   trong tap bat dong: T2 dung {(dis&h2).sum()}, T0 dung {(dis&h0).sum()}, '
      f'ca hai sai {(dis&~h2&~h0).sum()}')
print(f'   ORACLE (chon dung moi cau bat dong) = {100*(h2|h0).mean():.2f} '
      f'(T2 {100*(h2|h0).mean()-100*h2.mean():+.2f})\n')

# Cau hoi trung tam: tren TAP BAT DONG, tin hieu nao phan thang bai duoc?
sub = d[dis]
y = sub.t2_hit.values.astype(bool)          # 1 = T2 dung, 0 = T0 dung (bo cap ca hai sai)
keep = y | sub.t0_hit.values.astype(bool)
print(f'Tren {keep.sum()} cau BAT DONG co the phan thang bai ({100*keep.sum()/dis.sum():.0f}% cua tap bat dong):')


def auroc(sc, lab):
    o = np.argsort(sc); r = np.empty(len(sc), float); r[o] = np.arange(1, len(sc) + 1)
    p, q = lab.sum(), (~lab).sum()
    return (r[lab].sum() - p * (p + 1) / 2) / (p * q) if p and q else float('nan')


yk = y[keep]
for nm, sc in (('margin(T2) - margin(T0)', sub.t2_margin.values - sub.t0_margin.values),
               ('top1(T2) - top1(T0)', sub.t2_top1.values - sub.t0_top1.values),
               ('margin(T2) don le', sub.t2_margin.values)):
    a = auroc(sc[keep], yk)
    print(f'   AUROC {nm:26s} = {a:.4f}   ({"co tin hieu" if abs(a-.5)>.05 else "= ngau nhien"})')

print('\nCac quy tac trong tai (do tren TOAN BO 3001 cau):')
for nm, pick_t2 in (
        ('luon T2 (moc)', np.ones(n, bool)),
        ('margin cao hon thang', d.t2_margin.values >= d.t0_margin.values),
        ('top1 cao hon thang', d.t2_top1.values >= d.t0_top1.values),
        ('T2 tru khi margin(T0) hon 1.0', ~((d.t0_margin.values - d.t2_margin.values) > 1.0)),
        ('ORACLE', h2 | ~h0)):
    acc = np.where(pick_t2, h2, h0).mean()
    print(f'   {nm:32s} {100*acc:6.2f}  ({100*acc - 100*h2.mean():+.2f} so voi luon T2)')
print('\nTIEU CHI: mot quy tac chi dang theo neu vuot "luon T2" > +0.5 VA AUROC > 0.60.')
