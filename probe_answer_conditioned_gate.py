"""TRAN CUA CONG THUC MOI: alpha DIEU KIEN HOA THEO DAP AN UNG VIEN (khong dung nhan).

Cong thuc hien tai:   alpha_i = sigma(g([v_i ; W_q[t_cls ; e_type]]))   -> MOT alpha cho ca cau
Cong thuc de xuat:    alpha_i^(a) = sigma(g([v_i ; W_q[t_cls ; e_a]]))  -> MOT alpha cho MOI ung vien
                      s(a) = log P(a | GATE(v,q,a), q),  du doan = argmax_a s(a)

Vi sao dung cong thuc nay chu khong phai cong thuc thu 41:
  1. Tran oracle per-patch +11.27 EM la THAT (da qua control steerability: ep alpha theo mot dap
     an SAI chi lam model noi ra dap an do 1.53%).
  2. Nhung alpha oracle KHONG doc duoc tu (v, q, type): AUC 0.52-0.57 trong cung loai, trong khi
     alpha cua CHINH model doc duoc o AUC 0.94. Bien con thieu la DAP AN.
  3. Va do hom nay: do tin cay cua model NGUOC dau tren dung tap mau ma doi alpha co ich
     (11% duong) -> khong the DOAN alpha dung; nhung co the THU tung ung vien.
  4. Hien gate phai phuc vu moi dap an bang mot alpha -> 236 cau bi viet lai, net 8.5, 96% triet
     tieu. Dieu kien hoa theo ung vien la cach bo cai triet tieu do.

PHEP THU NAY khong train gi ca: voi moi ung vien a, fit alpha rieng de cuc dai log P(a).
Khong dung nhan (fit cho MOI ung vien nhu nhau), roi lay argmax.
  - Neu gold thang -> cong thuc co tran that, dang bo mot lan train.
  - Neu moi ung vien deu duoc fit tot nhu nhau (argmax khong doi / xau di) -> cong thuc chet,
    tiet kiem duoc mot lan train.
Doi chung BAT BUOC: cung tap ung vien, alpha cua chinh model (khong fit).
"""
import sys, os, time, argparse, unicodedata as ud
import torch, torch.nn.functional as F, pandas as pd, numpy as np
sys.path.insert(0, 'src')
from model import DeterministicVQA
from dataset import VQAGenDataset
from torch.utils.data import DataLoader, Subset
from transformers import AutoProcessor

p = argparse.ArgumentParser()
p.add_argument('--checkpoint', default='checkpoints_g_s1_s1/best_model.pt')
p.add_argument('--out', default='analysis/acg/acg_g_s1.csv')
p.add_argument('--csv', default='archive/val_split.csv')
p.add_argument('--image_folder', default='archive/data/images/train')
p.add_argument('--types', default='1,2', help='loai cau hoi (1=COUNT, 2=COLOR: vocab dong 10 tu)')
p.add_argument('--n', type=int, default=250)
p.add_argument('--steps', type=int, default=20)
p.add_argument('--lr', type=float, default=0.1)
p.add_argument('--l2', type=float, default=0.0, help='phat |alpha - alpha_model|^2')
p.add_argument('--shuffle_image', action='store_true',
               help='CONTROL PHAN CHUNG: fit alpha tren ANH CUA MAU KHAC. Neu top1 van len thi '
                    'cai loi den tu tu do toi uu (do dai chuoi, so buoc), KHONG phai bang chung '
                    'thi giac -> cong thuc chet.')
a = p.parse_args()

DEV = 'cuda'
norm = lambda s: ud.normalize('NFC', str(s)).strip().lower()
T = {0: 'OBJECT', 1: 'COUNT', 2: 'COLOR', 3: 'LOCATION'}

ck = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
sa = ck.get('args', {}); sa = sa if isinstance(sa, dict) else vars(sa)
sd = ck['model_state_dict']; K = list(sd.keys())
tlr = next((sd[k].shape[0] for k in K
            if k.startswith('encoder.base_model.model.layers.0.self_attn.q_proj.lora_A')), 16)
model = DeterministicVQA(
    vision_model_name=sa.get('vision_model', 'google/siglip-base-patch16-224'),
    bartpho_model_name='vinai/bartpho-syllable',
    num_fusion_layers=sa.get('num_fusion_layers', 2),
    fusion_type=sa.get('fusion_type', 'text2vision'),
    use_text_lora=True, text_lora_r=tlr, text_lora_alpha=sa.get('text_lora_alpha', 32),
    use_vision_lora=any('vision_encoder' in k and 'lora_A' in k for k in K),
    vision_lora_r=sa.get('vision_lora_r', 8),
    use_decoder_lora=sa.get('use_decoder_lora', False),
    decoder_lora_r=sa.get('decoder_lora_r', 16), decoder_lora_alpha=sa.get('decoder_lora_alpha', 32),
    use_vision_gate=True, vision_gate_init=sa.get('vision_gate_init', 1.5),
    vision_gate_min_alpha=sa.get('vision_gate_min_alpha', 0.0),
    vision_gate_max_alpha=sa.get('vision_gate_max_alpha', 1.0),
    use_type_task=any(k.startswith('type_head.') or k.startswith('type_classifier.') for k in K),
    use_siglip_pooler=sa.get('use_siglip_pooler', True)).to(DEV).eval()
res = model.load_state_dict(sd, strict=False)
AUX = ('teacher', 'contrastive_head', 'qgnd_ids')
assert not [k for k in res.missing_keys if 'teacher' not in k], 'thieu key'
assert not [k for k in res.unexpected_keys if not any(t in k for t in AUX)], 'thua key la'
for q in model.parameters():
    q.requires_grad_(False)
tok = model.tokenizer

tr = pd.read_csv(sa.get('train_csv', 'archive/train_split_oov.csv')); tr['an'] = tr.answer.map(norm)
TVOC = {t: sorted(set(g.an)) for t, g in tr.groupby('type')}
te = pd.read_csv(a.csv); gold_n = te.answer.map(norm)
TYPES = [int(x) for x in a.types.split(',')]
pool = [i for i in range(len(te)) if int(te.type.iloc[i]) in TYPES and gold_n.iloc[i] in TVOC[int(te.type.iloc[i])]]
rng = np.random.default_rng(0)
sel = sorted(rng.choice(pool, min(a.n, len(pool)), replace=False).tolist())
print(f'{len(sel)} mau, loai {[T[t] for t in TYPES]}, |vocab| = '
      f'{ {T[t]: len(TVOC[t]) for t in TYPES} }', flush=True)

vp = AutoProcessor.from_pretrained(sa.get('vision_model', 'google/siglip-base-patch16-224'))
ds = VQAGenDataset(csv_path=a.csv, image_folder=a.image_folder, vision_processor=vp,
                   tokenizer_name='vinai/bartpho-syllable', max_q_len=32, max_a_len=10,
                   include_question_type=True, auto_detect_type=False)

LBC = {}
for t in TYPES:
    e = tok(TVOC[t], return_tensors='pt', padding='max_length', truncation=True, max_length=10)
    lb = e.input_ids.clone(); lb[lb == tok.pad_token_id] = -100
    LBC[t] = lb.to(DEV)


def logp(lg, lb):
    return -F.cross_entropy(lg.reshape(-1, lg.size(-1)).float(), lb.reshape(-1),
                            ignore_index=-100, reduction='none').view(lb.shape).sum(1)


rows, t0 = [], time.time()
for c, j in enumerate(sel):
    b = next(iter(DataLoader(Subset(ds, [j]), batch_size=1)))
    t = int(te.type.iloc[j]); LB = LBC[t]; B = LB.size(0)
    if a.shuffle_image:
        # anh cua mau KHAC cung loai — cau hoi va ung vien giu nguyen
        jj = sel[(c + len(sel) // 2) % len(sel)]
        b['pixel_values'] = next(iter(DataLoader(Subset(ds, [jj]), batch_size=1)))['pixel_values']
    pv = b['pixel_values'].to(DEV).expand(B, -1, -1, -1)
    ii = b['input_ids'].to(DEV).expand(B, -1); am = b['attention_mask'].to(DEV).expand(B, -1)
    qt = b['question_type'].to(DEV).long().expand(B) if b.get('question_type') is not None else None
    gi = TVOC[t].index(gold_n.iloc[j])

    model.vision_gating.alpha_override = None
    with torch.no_grad():
        out = model(pixel_values=pv, input_ids=ii, attention_mask=am, labels=LB, question_types=qt)
        base = logp(out.answer_logits, LB)                      # doi chung: alpha cua model
        a0 = model.vision_gating.last_alpha.detach().float()
    if a0.dim() == 3:
        a0 = a0.squeeze(-1)
    theta = torch.logit(a0.clamp(1e-4, 1 - 1e-4)).clone().requires_grad_(True)
    opt = torch.optim.Adam([theta], lr=a.lr)
    with torch.enable_grad():
        for _ in range(a.steps):
            opt.zero_grad(set_to_none=True)
            al = torch.sigmoid(theta)
            model.vision_gating.alpha_override = al
            o = model(pixel_values=pv, input_ids=ii, attention_mask=am, labels=LB, question_types=qt)
            loss = -logp(o.answer_logits, LB).sum()
            if a.l2 > 0:
                loss = loss + a.l2 * ((al - a0) ** 2).sum()
            loss.backward(); opt.step()
    with torch.no_grad():
        model.vision_gating.alpha_override = torch.sigmoid(theta)
        fit = logp(model(pixel_values=pv, input_ids=ii, attention_mask=am,
                         labels=LB, question_types=qt).answer_logits, LB)
        amean = torch.sigmoid(theta).mean(1)
    model.vision_gating.alpha_override = None

    base = base.cpu().numpy(); fit = fit.cpu().numpy(); amean = amean.cpu().numpy()
    rows.append(dict(idx=j, type=t, gold=gold_n.iloc[j], n_cand=B, gi=gi,
                     hit_base=int(int(np.argmax(base)) == gi), hit_fit=int(int(np.argmax(fit)) == gi),
                     gain_gold=float(fit[gi] - base[gi]),
                     gain_other=float(np.mean(np.delete(fit - base, gi))),
                     alpha_gold=float(amean[gi]), alpha_other=float(np.mean(np.delete(amean, gi))),
                     alpha_model=float(a0.mean().item())))
    if (c + 1) % 25 == 0:
        el = time.time() - t0
        print(f'  {c+1}/{len(sel)} ({el/60:.1f}p, con ~{el/(c+1)*(len(sel)-c-1)/60:.0f}p)', flush=True)

d = pd.DataFrame(rows)
os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
d.to_csv(a.out, index=False)
print(f'\nda luu {a.out} ({len(d)} dong) | steps={a.steps} lr={a.lr} l2={a.l2}')
print(f'\n{"":<12}{"doi chung":>12}{"fit theo ung vien":>20}{"delta":>9}')
print(f'{"top1 (%)":<12}{100*d.hit_base.mean():>12.2f}{100*d.hit_fit.mean():>20.2f}'
      f'{100*(d.hit_fit.mean()-d.hit_base.mean()):>+9.2f}')
for t in TYPES:
    s = d[d.type == t]
    print(f'  {T[t]:<10}{100*s.hit_base.mean():>12.2f}{100*s.hit_fit.mean():>20.2f}'
          f'{100*(s.hit_fit.mean()-s.hit_base.mean()):>+9.2f}   (n={len(s)})')
print(f'\nfit nang log-lik len bao nhieu:  gold {d.gain_gold.mean():+.3f} | '
      f'ung vien khac {d.gain_other.mean():+.3f}  -> chenh {d.gain_gold.mean()-d.gain_other.mean():+.3f}')
print(f'alpha sau fit:  gold {d.alpha_gold.mean():.4f} | khac {d.alpha_other.mean():.4f} | '
      f'model {d.alpha_model.mean():.4f}')
print(f'sua dung {int(((d.hit_fit==1)&(d.hit_base==0)).sum())} | '
      f'lam hong {int(((d.hit_fit==0)&(d.hit_base==1)).sum())}')
