"""CO HOC DUOC NHAN alpha KHONG? — phep do RE chan cua truoc khi tieu 5 lan train.

Boi canh do duoc hom nay tren SigLIP2 (checkpoints_s2_T2, base 73.31):
    oracle alpha VO HUONG = 77.54  ->  +4.23  (tren SigLIP1 chi +3.50)
    => tran KHONG bi encoder manh nuot. Cai thieu la NHAN, khong phai du dia.

Phan ra: 135 mau duoc cuu (8 mau mat). Oracle cuu bang cach KEO alpha XUONG
    nhom duoc cuu: alpha model 0.611 -> oracle 0.307   (p = 3e-35)
    COLOR 0.068 | COUNT 0.172 | LOCATION 0.534 | OBJECT 0.550
=> "gate ly tuong" la mot CONG TAC THEO MAU: "cau nay dung tin luong patch tho nua".

Cau hoi chan cua: cong tac do co du doan duoc tu thong tin CO SAN LUC SUY LUAN khong?
  Neu CO  -> co nhan -> huong that su moi, dang tieu 5 lan train cho k-fold out-of-fold.
  Neu KHONG -> dong that su, va biet ma khong ton mot lan train nao.

Cach do (fit tren mot nua test, do tren nua kia — day la probe KHA NANG DU DOAN,
khong phai con so bao cao):
  dac trung deu KHONG CAN DAP AN:
    - pho cua ma tran patch (participation ratio, entropy, top1, effective rank, patch std)
    - loai cau hoi DU DOAN (model tu doan duoc luc suy luan)
    - do dai cau hoi
    - alpha cua chinh model (no biet gi ve mau nay khong?)
  muc tieu 1: alpha oracle          -> Spearman tren tap giu ngoai
  muc tieu 2: co thuoc nhom DUOC CUU -> AUC tren tap giu ngoai

Moc so sanh: AUC 0.5 = khong co tin hieu. Ti le nen cua nhom duoc cuu la 135/3001 = 4.5%.
"""
import numpy as np, pandas as pd, torch
from PIL import Image
from transformers import AutoModel, AutoProcessor
from scipy import stats
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score

DEV = 'cuda'; VM = 'google/siglip2-base-patch16-224'

te = pd.read_csv('archive/test.csv')
o = pd.read_csv('analysis/oracle_s2/scalar.csv')
b = pd.read_csv('checkpoints_s2_T2/eval_last.csv')
am = pd.read_csv('checkpoints_s2_T2/eval_alpha.csv')
oa = np.load('analysis/oracle_s2/scalar_alpha.npz')['alpha'].astype('float32').ravel()
assert len(te) == len(o) == len(b) == len(oa) == len(am)

be = b.exact_match.values > .5
oe = o.exact_match.values > .5
rescued = ((~be) & oe).astype(int)
print(f'{len(te)} mau | duoc cuu {rescued.sum()} ({rescued.mean()*100:.2f}%)')

# ---------- dac trung PHO, khong can nhan ----------
m = AutoModel.from_pretrained(VM).vision_model.to(DEV).eval().half()
pr = AutoProcessor.from_pretrained(VM)
rows = []
with torch.no_grad():
    for i in range(0, len(te), 64):
        ims = [Image.open(f'archive/data/images/test/{x}.jpg').convert('RGB')
               for x in te.img_id[i:i+64]]
        px = pr(images=ims, return_tensors='pt')['pixel_values'].to(DEV).half()
        X = m(pixel_values=px).last_hidden_state.float()          # [B,196,768]
        Z = X - X.mean(1, keepdim=True)
        s = torch.linalg.svdvals(Z.double())                       # [B,196]
        p = s**2; p = p / p.sum(1, keepdim=True)
        pr_ratio = 1.0 / (p**2).sum(1)
        ent = -(p * (p + 1e-12).log()).sum(1)
        rows.append(torch.stack([pr_ratio, ent, p[:, 0], p[:, :10].sum(1),
                                 ent.exp(), X.std(1).mean(1).double()], 1).cpu().numpy())
        if i % 640 == 0: print(f'  {i}/{len(te)}', flush=True)
SP = np.concatenate(rows).astype('float32')
del m; torch.cuda.empty_cache()

names = ['participation_ratio', 'entropy_pho', 'top1_energy', 'top10_energy',
         'effective_rank', 'patch_std']

# ---------- tuong quan don le, GHEP DUNG CAP (oracle SigLIP2 <-> dac trung SigLIP2) ----------
print(f'\n{"chi so":22s} {"vs ORACLE alpha":>20s} {"vs DUOC CUU":>14s}')
for j, n in enumerate(names):
    r1 = stats.spearmanr(SP[:, j], oa)
    r2 = stats.spearmanr(SP[:, j], rescued)
    print(f'{n:22s} {r1.correlation:+8.3f} (p={r1.pvalue:.1e}) {r2.correlation:+8.3f} (p={r2.pvalue:.1e})')

# ---------- gop dac trung + fit / do tren tap giu ngoai ----------
qtype = pd.get_dummies(am.pred_question_type if 'pred_question_type' in am else o.pred_question_type).values.astype('float32')
qlen = te.question.str.split().str.len().values.reshape(-1, 1).astype('float32')
malpha = am.alpha_mean.values.reshape(-1, 1).astype('float32')

BLOCKS = {
    'chi PHO':                 [SP],
    'chi LOAI cau hoi':        [qtype],
    'chi alpha cua MODEL':     [malpha],
    'PHO + LOAI + do dai':     [SP, qtype, qlen],
    'TAT CA (+alpha model)':   [SP, qtype, qlen, malpha],
}
rng = np.random.default_rng(0); idx = rng.permutation(len(te)); h = len(te) // 2
tr, ho = idx[:h], idx[h:]

print(f'\n{"tap dac trung":24s} {"rho(alpha) giu ngoai":>22s} {"AUC(duoc cuu)":>15s}')
for k, blocks in BLOCKS.items():
    F = np.concatenate(blocks, 1)
    mu, sd = F[tr].mean(0), F[tr].std(0).clip(1e-6)
    F = (F - mu) / sd
    rho = stats.spearmanr(Ridge(alpha=10).fit(F[tr], oa[tr]).predict(F[ho]), oa[ho]).correlation
    auc = roc_auc_score(rescued[ho], LogisticRegression(max_iter=2000, C=0.5)
                        .fit(F[tr], rescued[tr]).predict_proba(F[ho])[:, 1])
    print(f'{k:24s} {rho:+22.3f} {auc:15.3f}')

print('\nAUC 0.50 = khong co tin hieu. AUC >= 0.60 = co the xay bo phan lop.')
print('rho: alpha oracle co du doan duoc tu dac trung khong can dap an hay khong.')
print('LUU Y: day la probe KHA NANG DU DOAN (fit tren nua test), KHONG phai so bao cao.')
