"""Decoder doc bao nhieu tu THI GIAC? — tran cung cua moi thu TCVG co the lam.

TCVG chi sua 197 token thi giac trong chuoi encoder [vision 197 ; text 32]. Neu cross-attention
cua decoder chi do mot phan nho khoi luong len khoi thi giac, thi anh huong toi da cua BAT KY
phep sua nao tren khoi do cung bi chan cung boi con so ay — bat ke gate hoc tot the nao.

Do khoi luong cross-attn tren vision vs text, theo tung lop decoder va tung loai cau hoi.
Khong train, khong sua trong so.
"""
import sys, argparse
import torch, numpy as np, pandas as pd
sys.path.insert(0, 'src')
from probe_evidence_readout import build_model

p = argparse.ArgumentParser()
p.add_argument('--checkpoint', required=True)
p.add_argument('--test_csv', default='archive/test.csv')
p.add_argument('--image_folder', default='archive/data/images/test')
p.add_argument('--n', type=int, default=400)
a = p.parse_args()

from dataset import VQAGenDataset
from torch.utils.data import DataLoader, Subset
from transformers import AutoProcessor
DEV = 'cuda'
m, sa = build_model(a.checkpoint, DEV); tok = m.tokenizer
vp = AutoProcessor.from_pretrained(sa.get('vision_model'))
te = pd.read_csv(a.test_csv)
ds = VQAGenDataset(csv_path=a.test_csv, image_folder=a.image_folder, vision_processor=vp,
                   tokenizer_name='vinai/bartpho-syllable', max_q_len=32, max_a_len=10,
                   include_question_type=True, auto_detect_type=False)

# bat cross-attn cua decoder: goi lai decoder voi output_attentions=True bang cach hook
CAP = {}
def make_hook(i):
    def h(mod, args, kw, out):
        # BartDecoderLayer tra (hidden, self_attn_w, cross_attn_w, ...) khi output_attentions
        pass
    return h

n_vis = None
rows = []
idx = list(range(min(a.n, len(te))))
for j in idx:
    b = next(iter(DataLoader(Subset(ds, [j]), batch_size=1)))
    pv = b['pixel_values'].to(DEV); ii = b['input_ids'].to(DEV)
    am = b['attention_mask'].to(DEV); qt = b['question_type'].to(DEV).long()
    lb = b['labels'].to(DEV)
    # chay forward, chan lay encoder_hidden_states qua monkeypatch decoder
    grab = {}
    real = m.decoder
    class Wrap(torch.nn.Module):
        def __init__(s, d): super().__init__(); s.d = d
        def forward(s, **kw):
            grab['ehs'] = kw['encoder_hidden_states']
            kw['output_attentions'] = True
            o = s.d(**kw)
            grab['xa'] = o.cross_attentions
            return o
    m.decoder = Wrap(real)
    with torch.no_grad():
        m(pixel_values=pv, input_ids=ii, attention_mask=am, labels=lb, question_types=qt)
    m.decoder = real
    xa = grab.get('xa')
    if xa is None: continue
    P = grab['ehs'].size(1)
    nv = 197 if P >= 197 else P
    if n_vis is None:
        n_vis = nv
        print(f"chuoi encoder = {P} vi tri  ->  thi giac {nv}, van ban {P-nv}")
    tgt = (lb[0] != -100)
    per_layer = []
    for A in xa:                      # [1, heads, T_dec, P]
        w = A[0].mean(0)              # trung binh dau -> [T_dec, P]
        w = w[tgt[:w.size(0)]] if tgt.sum() > 0 else w
        per_layer.append(w[:, :nv].sum(-1).mean().item())
    rows.append(dict(j=j, qtype=str(te.type.iloc[j]), **{f"L{i}": v for i, v in enumerate(per_layer)}))

D = pd.DataFrame(rows)
LC = [c for c in D.columns if c.startswith('L')]
print(f"\nn = {len(D)} mau, {len(LC)} lop decoder")
print(f"\n=== khoi luong cross-attn tren THI GIAC ({n_vis}/{n_vis + (D.shape[0] and 0) or ''} vi tri) ===")
print(f"{'lop':>5} {'ty le thi giac':>16}")
for c in LC:
    print(f"{c:>5} {D[c].mean():15.1%}")
print(f"{'TB':>5} {D[LC].values.mean():15.1%}")
share_uniform = n_vis / (n_vis + 32)
print(f"\n  neu chia deu theo so vi tri thi se la {share_uniform:.1%}")
print(f"\n=== theo loai cau hoi (trung binh moi lop) ===")
for t, g in D.groupby('qtype'):
    print(f"  type {t}: n={len(g):3}  thi giac {g[LC].values.mean():.1%}")
D.to_csv('analysis/decoder_attn.csv', index=False)
print("\nluu analysis/decoder_attn.csv")
