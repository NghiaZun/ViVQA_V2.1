"""LUOT HAI THEO UNG VIEN — do TRAN cua phuong phap, truoc khi xay bat cu gi.

Y tuong dang kiem: alpha oracle duoc khop BANG GOLD, va bon phep do nhan dang deu that bai vi
chung chi dung dau vao luc suy luan. Bien con thieu la CHINH CAU TRA LOI. Khong gian dap an cua
ViVQA rat nho (COLOR va COUNT moi loai dung 10 tu, roi han khoi cac loai khac), nen liet ke duoc.

Phuong phap se la: voi moi ung vien a, khop alpha de cuc dai likelihood cua a, roi cham a duoi
chinh alpha do; chon ung vien TU NHAT QUAN nhat.

Vi sao co ly do tin no chay duoc: phep thu steer (2026-08-15) cho thay khop alpha ve mot dap an SAI
chi khien model noi ra dap an do 1.53%. Tuc UNG VIEN SAI KHONG THE duoc lam cho tu nhat quan.
Neu alpha chong lung duoc moi dap an thi xep hang nay vo nghia — do khong phai truong hop nay.

Script nay do TRAN: dung alpha ORACLE (khop truc tiep, khong hoc) cho tung ung vien.
  - neu gold xep NHAT o ti le cao hon EM hien tai  -> phuong phap co dat, dang xay
  - neu ~= EM hien tai                             -> khong them gi, phuong phap chet
Chay tren COLOR (10 ung vien, tu vung dong kin) — vua re vua la loai TCVG von manh nhat.
"""
import sys, torch, pandas as pd, numpy as np, unicodedata as ud
import torch.nn.functional as F
sys.path.insert(0, 'src')
from model import DeterministicVQA
from dataset import VQAGenDataset
from torch.utils.data import DataLoader, Subset
from transformers import AutoProcessor

CKPT = 'checkpoints_run87/best_model.pt'          # SigLIP1, base 72.34, oracle perpatch 83.61
DEV = 'cuda'; N = 200; STEPS = 10; LR = 0.5
norm = lambda s: ud.normalize('NFC', str(s)).strip().lower()

tr = pd.read_csv('archive/train_split_original.csv')
CAND = sorted({norm(a) for a in tr[tr.type == 2].answer})     # tu vung COLOR (dong kin)
print(f'{len(CAND)} ung vien COLOR: {CAND}')

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

te = pd.read_csv('archive/test.csv')
idx = te.index[te.type == 2].tolist()[:N]
vp = AutoProcessor.from_pretrained(sa.get('vision_model', 'google/siglip-base-patch16-224'))
ds = VQAGenDataset(csv_path='archive/test.csv', image_folder='archive/data/images/test',
                   vision_processor=vp, tokenizer_name='vinai/bartpho-syllable',
                   max_q_len=32, max_a_len=10, include_question_type=True, auto_detect_type=False)
tok = model.tokenizer

def cand_labels(text, L):
    t = tok([text], return_tensors='pt', padding='max_length', truncation=True, max_length=L)
    lb = t.input_ids.to(DEV)
    lb[lb == tok.pad_token_id] = -100
    return lb

def fit_ce(pv, ii, am, lb, qt):
    """khop alpha per-patch cuc dai likelihood cua lb, tra ve CE cuoi cung (thap = tu nhat quan)"""
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
    model.vision_gating.alpha_override = None
    return float(loss.detach())

hit_rank1, hit_base, n = 0, 0, 0
base = pd.read_csv('beam3fixed/seed42_ep40.csv')
for j in idx:
    b = next(iter(DataLoader(Subset(ds, [j]), batch_size=1)))
    pv, ii, am = b['pixel_values'].to(DEV), b['input_ids'].to(DEV), b['attention_mask'].to(DEV)
    qt = b['question_type'].to(DEV).long() if b.get('question_type') is not None else None
    L = b['labels'].size(1)
    ces = [fit_ce(pv, ii, am, cand_labels(c, L), qt) for c in CAND]
    pick = CAND[int(np.argmin(ces))]
    g = norm(te.answer[j])
    hit_rank1 += (pick == g)
    hit_base += (norm(base.prediction[j]) == g)
    n += 1
    if n % 25 == 0:
        print(f'  {n}/{len(idx)}  xep-hang-oracle {100*hit_rank1/n:.1f}%  |  model hien tai {100*hit_base/n:.1f}%',
              flush=True)

print(f'\n=== KET QUA tren {n} cau COLOR ===')
print(f'  chon bang ALPHA ORACLE theo tung ung vien : {100*hit_rank1/n:.2f}%')
print(f'  model hien tai (T2 seed42)                : {100*hit_base/n:.2f}%')
print(f'  chenh                                     : {100*(hit_rank1-hit_base)/n:+.2f} diem')
print()
print('  >> cao hon RO  -> ung vien dung NOI LEN duoc; lu 2 dang xay')
print('  >> ~= model    -> alpha khong phan biet duoc ung vien; lu 2 CHET, va do la ket luan sach')
