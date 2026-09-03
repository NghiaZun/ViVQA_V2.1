"""TINH TONG QUAT — MODEL CO NHAN RA DAP AN CHUA TUNG TRAIN KHONG?

Su that da do: model duoc train tren train_split_oov (323 dap an, giu lai 9) KHONG BAO GIO
phat ra mot dap an bi giu lai (0/3001), du trie cho phep. No thay bang hang xom ngu nghia
(huou cao co -> ngua van, sau -> nam). Ket luan cu: "thi giac con nguyen, NAMING la nut that".

Nhung "khong phat ra" co hai nguyen nhan rat khac nhau, va chua ai tach:
  (A) PRIOR SINH: decoder da sup ve phan bo dap an cua train. Thong tin thi giac VAN CO,
      chi bi prior de bep. -> sua bang DECODING, khong can train lai.
  (B) THIEU THONG TIN: cau noi vision->text khong mang thong tin ve lop chua tung thay.
      -> phai doi cach can chinh thi giac-ngon ngu, dat hon nhieu.

Phep tach: dua 9 dap an bi giu lai VAO danh sach ung vien roi cham diem ca 331 lop.
  Neu gold-OOV xep hang cao  -> (A). Model BIET, chi khong noi ra.
  Neu gold-OOV xep hang thap -> (B).

Va thu ngay cach sua cua (A): PMI (tru prior ngon ngu, uoc luong bang chinh model voi
vision bi zero -> logP(a | q, khong anh)):
      score_PMI(a) = logP(a | v,q) - lam * logP(a | q)
Khong train lai, khong doi kien truc.
"""
import sys, os, time, argparse, unicodedata as ud
import torch, torch.nn.functional as F, pandas as pd, numpy as np
sys.path.insert(0, 'src')
from model import DeterministicVQA
from dataset import VQAGenDataset
from torch.utils.data import DataLoader, Subset
from transformers import AutoProcessor

p = argparse.ArgumentParser()
p.add_argument('--checkpoint', required=True)
p.add_argument('--out', required=True)
p.add_argument('--csv', default='archive/val_split.csv')
p.add_argument('--image_folder', default='archive/data/images/train')
p.add_argument('--full_vocab', default='archive/train_split_original.csv')
p.add_argument('--subset', default='oov', choices=['oov', 'all'])
p.add_argument('--chunk', type=int, default=256)
a = p.parse_args()

DEV = 'cuda'
norm = lambda s: ud.normalize('NFC', str(s)).strip().lower()
LAMS = [0.0, 0.25, 0.5, 0.75, 1.0]

ck = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
sa = ck.get('args', {}); sa = sa if isinstance(sa, dict) else vars(sa)
sd = ck['model_state_dict']; K = list(sd.keys())
has_gate = any(k.startswith('vision_gating.') for k in K)
has_tl = any(k.startswith('encoder.base_model.model.layers.0.self_attn.q_proj.lora_A') for k in K)
has_vl = any('vision_encoder' in k and 'lora_A' in k for k in K)
has_tt = any(k.startswith('type_head.') or k.startswith('type_classifier.') for k in K)
tlr = next((sd[k].shape[0] for k in K
            if k.startswith('encoder.base_model.model.layers.0.self_attn.q_proj.lora_A')), 16)

model = DeterministicVQA(
    vision_model_name=sa.get('vision_model', 'google/siglip-base-patch16-224'),
    bartpho_model_name='vinai/bartpho-syllable',
    num_fusion_layers=sa.get('num_fusion_layers', 2),
    fusion_type=sa.get('fusion_type', 'text2vision'),
    use_text_lora=has_tl, text_lora_r=tlr, text_lora_alpha=sa.get('text_lora_alpha', 32),
    use_vision_lora=has_vl, vision_lora_r=sa.get('vision_lora_r', 8),
    use_decoder_lora=sa.get('use_decoder_lora', False),
    decoder_lora_r=sa.get('decoder_lora_r', 16), decoder_lora_alpha=sa.get('decoder_lora_alpha', 32),
    use_vision_gate=has_gate, vision_gate_init=sa.get('vision_gate_init', 1.5),
    vision_gate_min_alpha=sa.get('vision_gate_min_alpha', 0.0),
    vision_gate_max_alpha=sa.get('vision_gate_max_alpha', 1.0),
    use_type_task=has_tt, use_siglip_pooler=sa.get('use_siglip_pooler', True),
    use_mean_pool_cls=sa.get('use_mean_pool_cls', False),
    use_attn_pool_cls=sa.get('use_attn_pool_cls', False)).to(DEV).eval()
res = model.load_state_dict(sd, strict=False)
AUX = ('teacher', 'contrastive_head', 'qgnd_ids', '_ref_decoder', '_ref_lm_head')
# _ref_* : moc BARTpho pretrained DONG BANG cua --kl_pretrained_lambda. Chi dung LUC TRAIN
# de tinh KL; luc suy luan khong can, nen vut la DUNG. Da kiem: lech so voi BARTpho
# pretrain = 0.000e+00, tuc moc khong he bi train.
unexp = [k for k in res.unexpected_keys if not any(t in k for t in AUX)]
miss = [k for k in res.missing_keys if 'teacher' not in k]
if unexp or miss:
    print(f'!! KHONG KHOP: {len(unexp)} vut {unexp[:6]} | {len(miss)} ngau nhien {miss[:6]}')
    sys.exit(2)
for q in model.parameters():
    q.requires_grad_(False)
tok = model.tokenizer

TRAIN_SEEN = sa.get('train_csv', 'archive/train_split_oov.csv')
SEEN = {norm(x) for x in pd.read_csv(TRAIN_SEEN).answer}
CAND = sorted({norm(x) for x in pd.read_csv(a.full_vocab).answer})
HELD = sorted(set(CAND) - SEEN)
c2i = {c: i for i, c in enumerate(CAND)}
is_held = np.array([c in set(HELD) for c in CAND])
print(f'{len(CAND)} ung vien | {len(SEEN)} da thay | {len(HELD)} GIU LAI: {HELD}', flush=True)

enc = tok(CAND, return_tensors='pt', padding='max_length', truncation=True, max_length=10)
LB = enc.input_ids.to(DEV).clone(); LB[LB == tok.pad_token_id] = -100
NTOK = (LB != -100).sum(1).float()

te = pd.read_csv(a.csv)
gold_n = te.answer.map(norm)
sel = [i for i in range(len(te)) if (a.subset == 'all' or gold_n.iloc[i] in set(HELD))]
print(f'cham {len(sel)}/{len(te)} mau (subset={a.subset})', flush=True)

vp = AutoProcessor.from_pretrained(sa.get('vision_model', 'google/siglip-base-patch16-224'))
ds = VQAGenDataset(csv_path=a.csv, image_folder=a.image_folder, vision_processor=vp,
                   tokenizer_name='vinai/bartpho-syllable', max_q_len=32, max_a_len=10,
                   include_question_type=True, auto_detect_type=False)


def score_all(pv, ii, am, qt):
    s = torch.empty(len(CAND), device=DEV)
    with torch.no_grad():
        for st in range(0, len(CAND), a.chunk):
            lb = LB[st:st + a.chunk]; k = lb.size(0)
            out = model(pixel_values=pv.expand(k, -1, -1, -1), input_ids=ii.expand(k, -1),
                        attention_mask=am.expand(k, -1), labels=lb,
                        question_types=None if qt is None else qt.expand(k))
            lg = out.answer_logits.float()
            lp = -F.cross_entropy(lg.reshape(-1, lg.size(-1)), lb.reshape(-1),
                                  ignore_index=-100, reduction='none').view(lb.shape)
            s[st:st + k] = lp.sum(1)
    return s.cpu().numpy()


rows, t0 = [], time.time()
SV, SP = [], []   # ma tran diem [n_mau, n_ung_vien] — de moi phan tich hep sau nay lam offline
for c, j in enumerate(sel):
    b = next(iter(DataLoader(Subset(ds, [j]), batch_size=1)))
    pv = b['pixel_values'].to(DEV); ii = b['input_ids'].to(DEV); am = b['attention_mask'].to(DEV)
    qt = b['question_type'].to(DEV).long() if b.get('question_type') is not None else None
    model.text_only_mode = False
    s_v = score_all(pv, ii, am, qt)          # logP(a | v, q)
    model.text_only_mode = True
    s_p = score_all(pv, ii, am, qt)          # logP(a | q)   (vision zero)
    model.text_only_mode = False
    g = gold_n.iloc[j]; gi = c2i.get(g, -1)
    r = dict(idx=j, type=te.type.iloc[j], gold=g, gold_held=int(g in set(HELD)),
             gold_in_vocab=int(gi >= 0))
    for lam in LAMS:
        sc = s_v - lam * s_p
        order = np.argsort(-sc)
        rank = int(np.where(order == gi)[0][0]) + 1 if gi >= 0 else -1
        r[f'pick_l{lam}'] = CAND[int(order[0])]
        r[f'rank_l{lam}'] = rank
        r[f'hit_l{lam}'] = int(gi >= 0 and int(order[0]) == gi)
        r[f'pickheld_l{lam}'] = int(is_held[int(order[0])])
    r['gold_logp'] = float(s_v[gi]) if gi >= 0 else float('nan')
    r['gold_prior'] = float(s_p[gi]) if gi >= 0 else float('nan')
    rows.append(r); SV.append(s_v); SP.append(s_p)
    if (c + 1) % 50 == 0:
        el = time.time() - t0
        print(f'  {c+1}/{len(sel)} ({el/60:.1f}p, con ~{el/(c+1)*(len(sel)-c-1)/60:.0f}p)', flush=True)

d = pd.DataFrame(rows)
os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
d.to_csv(a.out, index=False)
np.savez_compressed(a.out.replace('.csv', '_scores.npz'),
                    s_vision=np.array(SV, dtype=np.float32), s_prior=np.array(SP, dtype=np.float32),
                    cand=np.array(CAND, dtype=object), idx=d.idx.values)
print(f'\nda luu {a.out} ({len(d)} dong) + ma tran diem')

sub = d[d.gold_held == 1]
print(f'\n=== {len(sub)} mau co gold BI GIU LAI (chua he train) ===')
print(f'{"lam":>6}{"EM":>8}{"top5":>8}{"rank tb":>10}{"rank med":>10}{"phat ra lop giu lai":>22}')
for lam in LAMS:
    rk = sub[f'rank_l{lam}']
    print(f'{lam:>6}{100*sub[f"hit_l{lam}"].mean():>8.2f}{100*(rk<=5).mean():>8.2f}'
          f'{rk.mean():>10.1f}{rk.median():>10.1f}{100*sub[f"pickheld_l{lam}"].mean():>22.2f}')
if (d.gold_held == 0).any():
    ss = d[d.gold_held == 0]
    print(f'\n=== {len(ss)} mau gold BINH THUONG (kiem tra PMI khong pha) ===')
    for lam in LAMS:
        print(f'  lam={lam}: EM {100*ss[f"hit_l{lam}"].mean():.2f} | '
              f'phat ra lop giu lai {100*ss[f"pickheld_l{lam}"].mean():.2f}%')
