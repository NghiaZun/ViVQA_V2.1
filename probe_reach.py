"""XEP HANG BANG "GIAI MA CO SINH RA UNG VIEN KHONG" — thay vi bang CE.

CHO KET TRUOC DAY (va vi sao no la sai lam cua toi):
  candrank xep hang ung vien bang CE teacher-forced sau khi khop alpha. Ket qua ~0 (COLOR +0.50,
  COUNT 67.0 vs 68.0 o moc 100/200). Toi ket luan "huong alpha dong".
  NHUNG hai so do CUA CHINH TOI mau thuan voi ket luan do:
      khop alpha ve GOLD    -> giai ma sinh ra gold : 83.61%   (oracle perpatch)
      khop alpha ve DAP AN SAI -> giai ma sinh ra no:  1.53%   (run_steer.sh)
  Do la mot bo phan biet cuc manh. CE thi ha xuong cho ung vien NAO CUNG DUOC (de), con
  GIAI MA co sinh ra ung vien do khong thi alpha KHONG bia duoc (kho). Hai dai luong khac nhau,
  va toi da dung dai luong de bi thao tung.

PHUONG PHAP (khong dung gold): voi moi ung vien a trong tu vung dong kin cua loai:
    1. khop alpha per-patch de cuc dai likelihood cua a
    2. GIAI MA binh thuong duoi alpha do
    3. cham diem: 1 neu dau ra == a  ("a la DEN DUOC")
  Chon ung vien den duoc. Neu nhieu ung vien den duoc -> lay cai co CE thap nhat lam tiebreak.

DU DOAN GHI TRUOC: neu ti le 1.53% do tren dap an sai NGAU NHIEN cung dung cho 9 ung vien cu the,
  thi so ung vien sai den duoc trung binh la 9 x 0.015 = 0.14 -> hau nhu khong va cham, va do
  chinh xac se tien gan 83.61%.
RUI RO CHINH (phai do, khong duoc gia dinh): ung vien GAN NGHIA (vd "hai" vs "ba" o COUNT) co
  the DEN DUOC de hon nhieu so voi mot dap an ngau nhien. Neu vay, va cham cao va phuong phap
  hong. Day chinh la ly do COUNT sai lech-mot-don-vi. Script in ra so ung vien den duoc moi mau
  de doc truc tiep dieu nay.
"""
import sys, os, csv, torch, pandas as pd, numpy as np, unicodedata as ud
import torch.nn.functional as F
sys.path.insert(0, 'src')
from model import DeterministicVQA
from dataset import VQAGenDataset
from torch.utils.data import DataLoader, Subset
from transformers import AutoProcessor

CKPT = 'checkpoints_run87/best_model.pt'
DEV = 'cuda'; N = 200; STEPS = 40; LR = 0.5
TYPE = int(os.environ.get('RTYPE', '1'))          # 1=COUNT, 2=COLOR
TNAME = {1: 'COUNT', 2: 'COLOR'}[TYPE]
norm = lambda s: ud.normalize('NFC', str(s)).strip().lower()

tr = pd.read_csv('archive/train_split_original.csv')
CAND = sorted({norm(a) for a in tr[tr.type == TYPE].answer})
print(f'{TNAME}: {len(CAND)} ung vien | {CAND}')

ck = torch.load(CKPT, map_location='cpu', weights_only=False); sa = ck.get('args', {})
VM = sa.get('vision_model', 'google/siglip-base-patch16-224')
model = DeterministicVQA(
    vision_model_name=VM, bartpho_model_name='vinai/bartpho-syllable',
    num_fusion_layers=sa.get('num_fusion_layers', 2), fusion_type='text2vision',
    use_text_lora=True, text_lora_r=16, text_lora_alpha=32,
    use_vision_gate=True, vision_gate_init=sa.get('vision_gate_init', 1.0),
    vision_gate_min_alpha=0.0, use_type_task=sa.get('use_type_loss', True),
    use_siglip_pooler=sa.get('use_siglip_pooler', True)).to(DEV).eval()
model.load_state_dict(ck['model_state_dict'], strict=False)
for p in model.parameters(): p.requires_grad_(False)

te = pd.read_csv('archive/test.csv')
idx = te.index[te.type == TYPE].tolist()[:N]
vp = AutoProcessor.from_pretrained(VM)
ds = VQAGenDataset(csv_path='archive/test.csv', image_folder='archive/data/images/test',
                   vision_processor=vp, tokenizer_name='vinai/bartpho-syllable',
                   max_q_len=32, max_a_len=10, include_question_type=True, auto_detect_type=False)
tok = model.tokenizer
base = pd.read_csv('beam3fixed/seed42_ep40.csv')

PART = f'analysis/reach_{TNAME}_partial.csv'
done = {}
if os.path.exists(PART):
    for r in csv.DictReader(open(PART)):
        done[int(r['idx'])] = (r['pick'], int(r['n_reach']))
    print(f'  tiep tuc: da co {len(done)} mau', flush=True)
os.makedirs('analysis', exist_ok=True)
_fh = open(PART, 'a', newline=''); _w = csv.writer(_fh)
if not done: _w.writerow(['idx', 'pick', 'n_reach', 'gold']); _fh.flush()

def cand_labels(text, L):
    t = tok([text], return_tensors='pt', padding='max_length', truncation=True, max_length=L)
    lb = t.input_ids.to(DEV); lb[lb == tok.pad_token_id] = -100
    return lb

def fit_then_generate(pv, ii, am, lb, qt):
    """khop alpha ve lb, roi GIAI MA. tra ve (chuoi sinh ra, CE cuoi)."""
    model.vision_gating.alpha_override = None
    with torch.no_grad():
        model(pixel_values=pv, input_ids=ii, attention_mask=am, labels=lb, question_types=qt)
        a0 = model.vision_gating.last_alpha.detach().float()
    if a0.dim() == 3: a0 = a0.squeeze(-1)
    th = torch.logit(a0.clamp(1e-4, 1 - 1e-4)).requires_grad_(True)
    opt = torch.optim.Adam([th], lr=LR)
    with torch.enable_grad():
        for _ in range(STEPS):
            opt.zero_grad(set_to_none=True)
            model.vision_gating.alpha_override = torch.sigmoid(th)
            out = model(pixel_values=pv, input_ids=ii, attention_mask=am, labels=lb, question_types=qt)
            lg = out.answer_logits
            loss = F.cross_entropy(lg.reshape(-1, lg.size(-1)), lb.reshape(-1), ignore_index=-100)
            loss.backward(); opt.step()
    ce = float(loss.detach())
    with torch.no_grad():   # GIAI MA duoi alpha vua khop — day moi la tin hieu khong bia duoc
        model.vision_gating.alpha_override = torch.sigmoid(th).detach()
        g = model.generate(pixel_values=pv, input_ids=ii, attention_mask=am,
                           max_length=10, num_beams=3, repetition_penalty=1.3)
        gen = g[0] if isinstance(g, (list, tuple)) else g
        gen = gen[0] if isinstance(gen, list) else gen
    model.vision_gating.alpha_override = None
    return norm(gen), ce

hit, hit_base, n, nreach = 0, 0, 0, []
for j in idx:
    g = norm(te.answer[j])
    if j in done:
        pick, nr = done[j]
    else:
        b = next(iter(DataLoader(Subset(ds, [j]), batch_size=1)))
        pv, ii, am = b['pixel_values'].to(DEV), b['input_ids'].to(DEV), b['attention_mask'].to(DEV)
        qt = b['question_type'].to(DEV).long() if b.get('question_type') is not None else None
        L = b['labels'].size(1)
        reach, ces = [], []
        for c in CAND:
            out, ce = fit_then_generate(pv, ii, am, cand_labels(c, L), qt)
            ces.append(ce)
            if out == c: reach.append(c)
        nr = len(reach)
        if nr == 1:   pick = reach[0]
        elif nr > 1:  pick = min(reach, key=lambda c: ces[CAND.index(c)])   # tiebreak: CE thap nhat
        else:         pick = CAND[int(np.argmin(ces))]                      # khong ai den duoc -> CE
        _w.writerow([j, pick, nr, g]); _fh.flush()
    hit += (pick == g); hit_base += (norm(base.prediction[j]) == g); nreach.append(nr); n += 1
    if n % 20 == 0:
        print(f'  {n}/{len(idx)}  DEN-DUOC {100*hit/n:5.1f}%  |  model {100*hit_base/n:5.1f}%'
              f'  |  so ung vien den duoc tb {np.mean(nreach):.2f}', flush=True)

print(f'\n=== {TNAME}, {n} mau ===')
print(f'  chon bang GIAI MA DEN DUOC : {100*hit/n:.2f}%')
print(f'  model hien tai (T2 seed42) : {100*hit_base/n:.2f}%')
print(f'  chenh                      : {100*(hit-hit_base)/n:+.2f} diem')
print(f'  so ung vien den duoc / mau : {np.mean(nreach):.2f}  (0 = khong ai, 1 = duy nhat)')
print(f'    khong ai den duoc: {100*np.mean(np.array(nreach)==0):.1f}%'
      f' | duy nhat mot: {100*np.mean(np.array(nreach)==1):.1f}%'
      f' | nhieu hon mot: {100*np.mean(np.array(nreach)>1):.1f}%')
print('\n  Neu "duy nhat mot" cao va do chinh xac >> model -> tin hieu DEN DUOC la that.')
print('  Neu "nhieu hon mot" cao -> ung vien gan nghia cung den duoc, va phuong phap hong.')
