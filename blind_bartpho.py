"""BASELINE MU SACH: BARTpho encode CAU HOI -> decoder ra DAP AN. Khong co module thi giac nao.

VI SAO CAN, du A.2 da co mot con so:
  A.2 dang bao cao 11.46 cho "chi van ban", nhung do la KIEN TRUC DAY DU voi anh bi ZERO HOA:
  decoder van phai attend vao 197 token toan so 0 chen truoc text, cong contrastive loss tren
  vector rong. Ket qua 11.46 THAP HON ca baseline k-lan-can KHONG huan luyen (28.82) va thap hon
  ca phep doan bang tien to cau hoi (18.33%). Mot mo hinh CO huan luyen khong the te hon mot phep
  doan tan suat — do la dau hieu cau hinh do bi nhieu boi kien truc, khong phai phep do sach.

  Day la baseline dung nghia: chi BARTpho, chi cau hoi, chi dap an. Neu no cao hon 11.46 nhieu thi
  con so trong luan van phai duoc thay, va lap luan "he thong that su dung anh" van dung nhung
  phai dua tren so dung.

KHONG DUNG LAI GI cua model.py de tranh keo theo bat ky duong thi giac nao.
"""
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, MBartForConditionalGeneration

sys.path.insert(0, 'src')
import eval as E                                    # noqa: E402  (chi dung ham chuan hoa)
norm = lambda s: E._normalize_vn(str(s), True)      # noqa: E731

DEV = 'cuda'
EPOCHS, BS, LR = 20, 16, 3e-5
tok = AutoTokenizer.from_pretrained('vinai/bartpho-syllable')
model = MBartForConditionalGeneration.from_pretrained('vinai/bartpho-syllable').to(DEV)
print(f"tham so: {sum(p.numel() for p in model.parameters())/1e6:.1f}M (toan bo trainable)")


class QA(Dataset):
    def __init__(self, csv):
        d = pd.read_csv(csv)
        self.q = d.question.astype(str).tolist()
        self.a = d.answer.astype(str).tolist()

    def __len__(self):
        return len(self.q)

    def __getitem__(self, i):
        return self.q[i], self.a[i]


def collate(b):
    q, a = zip(*b)
    x = tok(list(q), return_tensors='pt', padding=True, truncation=True, max_length=32)
    y = tok(list(a), return_tensors='pt', padding=True, truncation=True, max_length=10)
    lab = y.input_ids.clone()
    lab[lab == tok.pad_token_id] = -100
    return x.input_ids, x.attention_mask, lab


tr = DataLoader(QA('archive/train_split.csv'), batch_size=BS, shuffle=True, collate_fn=collate)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS * len(tr))

model.train()
for ep in range(EPOCHS):
    tot = n = 0
    for ii, am, lb in tr:
        ii, am, lb = ii.to(DEV), am.to(DEV), lb.to(DEV)
        loss = model(input_ids=ii, attention_mask=am, labels=lb).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sch.step(); opt.zero_grad()
        tot += loss.item(); n += 1
    print(f'  epoch {ep+1}/{EPOCHS} loss {tot/n:.4f}', flush=True)

# ── danh gia: giai ma TU DO va giai ma RANG BUOC (de so cong bang voi 72.34) ──
te = pd.read_csv('archive/test.csv')
gold = [norm(a) for a in te.answer]
CAND = sorted({norm(a) for a in pd.read_csv('archive/train_split.csv').answer})
model.eval()

preds = []
with torch.no_grad():
    for i in range(0, len(te), 64):
        x = tok(te.question.astype(str).tolist()[i:i+64], return_tensors='pt',
                padding=True, truncation=True, max_length=32).to(DEV)
        out = model.generate(**x, max_length=10, num_beams=3)
        preds += [norm(s) for s in tok.batch_decode(out, skip_special_tokens=True)]
free_em = 100 * np.mean([p == g for p, g in zip(preds, gold)])

# rang buoc: cham diem toan bo tap ung vien, chon cai co log-likelihood cao nhat
enc = tok(CAND, return_tensors='pt', padding=True, truncation=True, max_length=10)
LB = enc.input_ids.to(DEV).clone(); LB[LB == tok.pad_token_id] = -100
ntok = (LB != -100).sum(1).float().clamp(min=1)
hits = []
with torch.no_grad():
    for i in range(len(te)):
        x = tok([str(te.question.iloc[i])], return_tensors='pt', truncation=True,
                max_length=32).to(DEV)
        s = torch.empty(len(CAND), device=DEV)
        for a in range(0, len(CAND), 64):
            lb = LB[a:a+64]; k = lb.size(0)
            lg = model(input_ids=x.input_ids.expand(k, -1),
                       attention_mask=x.attention_mask.expand(k, -1), labels=lb).logits.float()
            lp = -F.cross_entropy(lg.reshape(-1, lg.size(-1)), lb.reshape(-1),
                                  ignore_index=-100, reduction='none').view(lb.shape)
            s[a:a+k] = lp.sum(1)
        hits.append(CAND[int(torch.argmax(s / ntok))] == gold[i])
cons_em = 100 * np.mean(hits)

print('\n' + '=' * 62)
print(f'BASELINE MU SACH (chi BARTpho, chi cau hoi):')
print(f'   giai ma tu do        : {free_em:.2f}%')
print(f'   chon trong tap ung vien: {cons_em:.2f}%')
print('\nDoi chieu:')
print('   mo hinh day du (co anh)                : 72.34%')
print('   "chi van ban" dang bao cao trong A.2   : 11.46%   <- kien truc day du, anh zero hoa')
print('   k-lan-can mu, KHONG huan luyen         : 28.82%')
print('   doan bang tien to cau hoi              : 18.33%')
print('\nDOC: neu baseline nay >> 11.46 thi con so A.2 la hien vat cua cau hinh, phai thay.')
print('     Khoang cach that cua nhanh thi giac = 72.34 - (so lon hon trong hai so cua baseline nay).')
pd.DataFrame({'question': te.question, 'gold': gold, 'pred_free': preds}).to_csv(
    'analysis/blind_bartpho.csv', index=False)
print('\nda luu analysis/blind_bartpho.csv')
