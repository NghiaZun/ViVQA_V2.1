"""CO CHE CUA LUAT CARDINALITY — vi sao gate het headroom khi tu vung dap an lon.

Luat da biet (tran oracle, SigLIP1): COUNT(10 dap an) +23.64 | COLOR(10) +17.92 |
LOCATION(118) +6.28 | OBJECT(290) +6.09. Moi tuong quan tren 4 diem, chua co co che.

Hai co che canh tranh, phan biet duoc bang thu hang:
  (A) SUC VOI cua alpha co han va TUYET DOI (~2.8 dap an). Tu vung lon -> gold hiem khi nam
      trong tam voi -> headroom sut. Du doan: THU HANG cua gold duoi alpha tot nhat cua no
      xau di theo |tu vung|, nhung khoang CAI THIEN thu hang thi khong doi.
  (B) alpha manh nhu nhau, chi la canh tranh dong hon. Du doan: gold van len duoc top-1 o ty le
      tuong tu, chi la co nhieu doi thu hon.

Do: fit alpha per-sample de cuc dai log P(gold) (dung nhan -> day la ORACLE, khong phai phuong
phap), roi cham TOAN BO tu vung cua loai do duoi chinh alpha ay. Ghi thu hang gold truoc/sau.
Neu (A) dung, luat cardinality co cong thuc tien doan duoc cho dataset moi: headroom ~ P(gold
nam trong tam voi), uoc tu |tu vung| va phan bo thu hang co san — truoc khi train bat cu gi.
"""
import sys, time, argparse, unicodedata as ud
import torch, torch.nn.functional as F, numpy as np, pandas as pd
sys.path.insert(0, 'src')
from model import DeterministicVQA
from dataset import VQAGenDataset
from torch.utils.data import DataLoader, Subset
from transformers import AutoProcessor

p = argparse.ArgumentParser()
p.add_argument('--checkpoint', default='checkpoints_s2_T2/best_model.pt')
p.add_argument('--per_type', type=int, default=80)
p.add_argument('--steps', type=int, default=25)
p.add_argument('--lr', type=float, default=0.1)
p.add_argument('--out', default='analysis/reach/reach_s2T2.csv')
p.add_argument('--train_csv', default=None, help='override answer-vocab CSV when the checkpoint stores a stale absolute path')
a = p.parse_args()

DEV = 'cuda'
norm = lambda s: ud.normalize('NFC', str(s)).strip().lower()
T = {0: 'OBJECT', 1: 'COUNT', 2: 'COLOR', 3: 'LOCATION'}

ck = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
sa = ck['args']; sa = sa if isinstance(sa, dict) else vars(sa)
sd = ck['model_state_dict']; K = list(sd.keys())
tlr = next((sd[k].shape[0] for k in K
            if k.startswith('encoder.base_model.model.layers.0.self_attn.q_proj.lora_A')), 16)
m = DeterministicVQA(
    vision_model_name=sa.get('vision_model'), bartpho_model_name='vinai/bartpho-syllable',
    num_fusion_layers=sa.get('num_fusion_layers', 2), fusion_type=sa.get('fusion_type', 'text2vision'),
    use_text_lora=True, text_lora_r=tlr, text_lora_alpha=sa.get('text_lora_alpha', 32),
    use_decoder_lora=sa.get('use_decoder_lora', False), decoder_lora_r=sa.get('decoder_lora_r', 16),
    decoder_lora_alpha=sa.get('decoder_lora_alpha', 32), use_vision_gate=True,
    vision_gate_init=sa.get('vision_gate_init', 1.5),
    vision_gate_min_alpha=sa.get('vision_gate_min_alpha', 0.0),
    vision_gate_max_alpha=sa.get('vision_gate_max_alpha', 1.0),
    use_type_task=any(k.startswith('type_head.') or k.startswith('type_classifier.') for k in K),
    use_siglip_pooler=sa.get('use_siglip_pooler', True)).to(DEV).eval()
r = m.load_state_dict(sd, strict=False)
assert not [k for k in r.missing_keys if 'teacher' not in k], 'thieu key'
for q in m.parameters():
    q.requires_grad_(False)
tok = m.tokenizer

tr = pd.read_csv(a.train_csv or sa.get('train_csv', 'archive/train_split_original.csv')); tr['an'] = tr.answer.map(norm)
TVOC = {t: sorted(set(g.an)) for t, g in tr.groupby('type')}
LB = {}
for t, voc in TVOC.items():
    e = tok(voc, return_tensors='pt', padding='max_length', truncation=True, max_length=10)
    x = e.input_ids.to(DEV).clone(); x[x == tok.pad_token_id] = -100
    LB[t] = x
print({T[t]: len(v) for t, v in TVOC.items()}, flush=True)

te = pd.read_csv('archive/test.csv'); gold = te.answer.map(norm)
vp = AutoProcessor.from_pretrained(sa.get('vision_model'))
ds = VQAGenDataset(csv_path='archive/test.csv', image_folder='archive/data/images/test',
                   vision_processor=vp, tokenizer_name='vinai/bartpho-syllable', max_q_len=32,
                   max_a_len=10, include_question_type=True, auto_detect_type=False)
rng = np.random.default_rng(0)
sel = []
for t in [0, 1, 2, 3]:
    pool = [i for i in range(len(te)) if int(te.type.iloc[i]) == t and gold.iloc[i] in TVOC[t]]
    sel += rng.choice(pool, min(a.per_type, len(pool)), replace=False).tolist()
sel = sorted(sel)


def logp(lg, lb):
    return -F.cross_entropy(lg.reshape(-1, lg.size(-1)).float(), lb.reshape(-1),
                            ignore_index=-100, reduction='none').view(lb.shape).sum(1)


def score_all(pv, ii, am, qt, lb, chunk=256):
    s = torch.empty(lb.size(0), device=DEV)
    with torch.no_grad():
        for st in range(0, lb.size(0), chunk):
            x = lb[st:st + chunk]; k = x.size(0)
            o = m(pixel_values=pv.expand(k, -1, -1, -1), input_ids=ii.expand(k, -1),
                  attention_mask=am.expand(k, -1), labels=x, question_types=qt.expand(k))
            s[st:st + k] = logp(o.answer_logits, x)
    return s


rows, t0 = [], time.time()
for c, j in enumerate(sel):
    b = next(iter(DataLoader(Subset(ds, [j]), batch_size=1)))
    pv = b['pixel_values'].to(DEV); ii = b['input_ids'].to(DEV); am = b['attention_mask'].to(DEV)
    qt = b['question_type'].to(DEV).long()
    t = int(te.type.iloc[j]); voc = TVOC[t]; gi = voc.index(gold.iloc[j])
    gl = LB[t][gi:gi + 1]

    m.vision_gating.alpha_override = None
    base = score_all(pv, ii, am, qt, LB[t])
    a0 = m.vision_gating.last_alpha.detach().float()
    if a0.dim() == 3:
        a0 = a0.squeeze(-1)
    a0 = a0[:1]
    # fit alpha per-sample de cuc dai log P(gold) — ORACLE (dung nhan)
    th = torch.logit(a0.clamp(1e-4, 1 - 1e-4)).clone().requires_grad_(True)
    opt = torch.optim.Adam([th], lr=a.lr)
    with torch.enable_grad():
        for _ in range(a.steps):
            opt.zero_grad(set_to_none=True)
            m.vision_gating.alpha_override = torch.sigmoid(th)
            o = m(pixel_values=pv, input_ids=ii, attention_mask=am, labels=gl, question_types=qt)
            (-logp(o.answer_logits, gl).sum()).backward(); opt.step()
    m.vision_gating.alpha_override = torch.sigmoid(th).detach()
    fit = score_all(pv, ii, am, qt, LB[t])
    m.vision_gating.alpha_override = None

    bn, fn = base.cpu().numpy(), fit.cpu().numpy()
    rows.append(dict(idx=j, type=t, nvoc=len(voc),
                     rank_base=int((bn > bn[gi]).sum()) + 1, rank_fit=int((fn > fn[gi]).sum()) + 1,
                     hit_base=int(bn.argmax() == gi), hit_fit=int(fn.argmax() == gi),
                     alpha_base=float(a0.mean()), alpha_fit=float(torch.sigmoid(th).mean())))
    if (c + 1) % 40 == 0:
        el = time.time() - t0
        print(f'  {c+1}/{len(sel)} ({el/60:.1f}p, con ~{el/(c+1)*(len(sel)-c-1)/60:.0f}p)', flush=True)

d = pd.DataFrame(rows)
import os
os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
d.to_csv(a.out, index=False)
print(f'\nda luu {a.out} ({len(d)} dong)\n')
print(f'{"loai":<10}{"|vocab|":>8}{"n":>5}{"top1 goc":>10}{"top1 oracle":>13}{"headroom":>10}'
      f'{"rank goc":>10}{"rank fit":>10}{"len bac":>9}')
for t in [0, 1, 2, 3]:
    s = d[d.type == t]
    if not len(s):
        continue
    print(f'{T[t]:<10}{s.nvoc.iloc[0]:>8}{len(s):>5}{100*s.hit_base.mean():>10.2f}'
          f'{100*s.hit_fit.mean():>13.2f}{100*(s.hit_fit.mean()-s.hit_base.mean()):>+10.2f}'
          f'{s.rank_base.median():>10.1f}{s.rank_fit.median():>10.1f}'
          f'{(s.rank_base-s.rank_fit).median():>+9.1f}')
print('\nDOC: neu "len bac" (cai thien thu hang) gan NHU NHAU o moi loai trong khi headroom sut')
print('theo |vocab|, thi suc voi cua alpha la TUYET DOI -> luat cardinality co co che, va')
print('headroom tren dataset moi tien doan duoc tu phan bo thu hang + |tu vung|.')
