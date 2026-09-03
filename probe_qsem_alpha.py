"""NGU NGHIA CAU HOI co bao hieu mau nao can HA alpha khong?

Lo hong trong probe_predict_alpha.py: o do toi chi thu LOAI cau hoi (4 chieu) va DO DAI.
Chua thu NOI DUNG cau hoi. Gate hien tai CO nhin text_cls, nhung no duoc huan luyen giam tiep
qua answer loss — huan luyen truc tiep bang nhan oracle co the rut ra duoc cai ma loss giam tiep
khong rut duoc. Nen day la mot cau hoi khac han, va no phai duoc do rieng.

Nhan: 135 mau ma oracle alpha vo huong cuu duoc (base 73.31 -> 77.54).
Dac trung: nhung cai LUC SUY LUAN CO SAN, khong dung dap an.
    - loai cau hoi du doan (moc so sanh, AUC 0.630)
    - nhung cai co the du doan tu cau hoi (embedding BARTpho, trung binh token)
    - gop ca hai

Neu embedding cau hoi day AUC len ro rang tren 0.630 -> co nhan hoc duoc -> TCVG co duong di:
    giam sat alpha bang nhan oracle out-of-fold, dieu kien theo ngu nghia cau hoi.
Neu khong -> cung voi ket qua trong-loai (0.53), bat kha nhan dang la day du tren ca hai truc.
"""
import numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

DEV = 'cuda'; TM = 'vinai/bartpho-syllable'
te = pd.read_csv('archive/test.csv')
o = pd.read_csv('analysis/oracle_s2/scalar.csv')
b = pd.read_csv('checkpoints_s2_T2/eval_last.csv')
y = (((b.exact_match.values <= .5) & (o.exact_match.values > .5))).astype(int)
print(f'{len(te)} mau | duoc cuu {y.sum()}')

tok = AutoTokenizer.from_pretrained(TM)
enc = AutoModel.from_pretrained(TM).encoder.to(DEV).eval().half()
Q = []
with torch.no_grad():
    for i in range(0, len(te), 64):
        t = tok(list(te.question[i:i+64]), padding=True, truncation=True,
                max_length=32, return_tensors='pt').to(DEV)
        h = enc(**t).last_hidden_state.float()
        m = t.attention_mask.unsqueeze(-1).float()
        Q.append(((h * m).sum(1) / m.sum(1)).cpu().numpy())
Q = np.concatenate(Q).astype('float32')
del enc; torch.cuda.empty_cache()
print(f'embedding cau hoi: {Q.shape}')

T = pd.get_dummies(o.pred_question_type).values.astype('float32')
rng = np.random.default_rng(0); idx = rng.permutation(len(te)); h = len(te)//2
tr, ho = idx[:h], idx[h:]

def auc(F, C=0.1):
    mu, sd = F[tr].mean(0), F[tr].std(0).clip(1e-6)
    F = (F - mu) / sd
    return roc_auc_score(y[ho], LogisticRegression(max_iter=3000, C=C)
                         .fit(F[tr], y[tr]).predict_proba(F[ho])[:, 1])

print(f'\n{"tap dac trung":32s} {"AUC":>8s}')
print(f'{"chi LOAI cau hoi (moc)":32s} {auc(T, 1.0):8.3f}')
for C in [0.003, 0.01, 0.03, 0.1]:
    print(f'{"embedding CAU HOI (C=%.3f)"%C:32s} {auc(Q, C):8.3f}')
print(f'{"LOAI + embedding":32s} {auc(np.concatenate([T, Q], 1), 0.03):8.3f}')

# doi chung nhan gia: xao nhan -> AUC cua "khong co gi de hoc" o dung so chieu nay
yp = y.copy(); rng.shuffle(yp); ytrue = y; y = yp
print(f'{"DOI CHUNG nhan xao":32s} {auc(np.concatenate([T, Q], 1), 0.03):8.3f}')
y = ytrue
print('\n=> chi ket luan CO tin hieu neu AUC vuot ro moc 0.630 VA vuot han doi chung nhan xao.')
