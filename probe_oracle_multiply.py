"""TRUONG ALPHA TOI UU duoi MULTIPLY co THUA va CO CAU TRUC khong? (blend: khuech tan + truc giao)

Cau hoi quyet dinh: co nen cai yeu cau (4) "alpha gan nhi phan" khong.
Suy tu co che decoder: alpha MEM luon kha nghich — decoder chi can hoc lai thang do W_k/W_q.
Chi khi patch that su bi XOA thi chon loc moi rang buoc duoc. Nhung FINDINGS_tcvg_mechanism do
duoc truong oracle duoi BLEND la KHUECH TAN (top-10% share 0.161 khi cuu duoc vs 0.281 khi that
bai) va TRUC GIAO giua cac mau (cos +0.009) -> ket luan "gate thua la sai doi tuong".
Nhung phep do do lam tren toan tu BLEND. Chua ai khop oracle duoi MULTIPLY.

CANH BAO PHAI GHI: run87 duoc TRAIN voi blend. Cham no duoi multiply la LECH PHAN BO
(covariate shift) — dung de so TRAN tuyet doi giua hai toan tu. Cai doc duoc la CAU TRUC
cua truong (thua/khuech tan, truc giao/chung huong), von it nhay cam hon voi lech nay.
Ban SACH se chay lai tren checkpoint da train BANG multiply (dang train, seed 42).
"""
import sys, argparse, unicodedata as ud
import torch, torch.nn.functional as F, numpy as np, pandas as pd
sys.path.insert(0,'src')
from probe_evidence_readout import build_model
norm=lambda s: ud.normalize('NFC',str(s)).strip().lower()
p=argparse.ArgumentParser()
p.add_argument('--checkpoint',default='checkpoints_run87/best_model.pt')
p.add_argument('--per_type',type=int,default=60); p.add_argument('--steps',type=int,default=25)
p.add_argument('--lr',type=float,default=0.1); p.add_argument('--out',default='analysis/mech/oracle_mul.csv')
a=p.parse_args()
from dataset import VQAGenDataset
from torch.utils.data import DataLoader, Subset
from transformers import AutoProcessor
DEV='cuda'
tr=pd.read_csv('archive/train_split_original.csv'); tr['an']=tr.answer.map(norm)
TVOC={int(t):sorted(set(g.an)) for t,g in tr.groupby('type')}
te=pd.read_csv('archive/test.csv'); gold=te.answer.map(norm)
rng=np.random.RandomState(0); idx=[]
for t,g in te.groupby('type'):
    ii=[i for i in g.index if gold.iloc[i] in TVOC[int(t)]]
    idx+=list(rng.choice(ii,min(a.per_type,len(ii)),replace=False))
idx=sorted(int(i) for i in idx)

RES={}
for MODE in ('blend','multiply'):
    m,sa=build_model(a.checkpoint,DEV)
    m.vision_gating.gate_mode=MODE                      # ep toan tu
    tok=m.tokenizer
    vp=AutoProcessor.from_pretrained(sa.get('vision_model'))
    ds=VQAGenDataset(csv_path='archive/test.csv',image_folder='archive/data/images/test',
      vision_processor=vp,tokenizer_name='vinai/bartpho-syllable',max_q_len=32,max_a_len=10,
      include_question_type=True,auto_detect_type=False)
    LB={}
    for t,voc in TVOC.items():
        e=tok(voc,return_tensors='pt',padding='max_length',truncation=True,max_length=10)
        x=e.input_ids.to(DEV).clone(); x[x==tok.pad_token_id]=-100; LB[t]=x
    rows=[]; FIELDS=[]
    for n_,j in enumerate(idx):
        t=int(te.type.iloc[j]); voc=TVOC[t]; gi=voc.index(gold.iloc[j]); L=LB[t]
        b=next(iter(DataLoader(Subset(ds,[j]),batch_size=1)))
        kw=dict(pixel_values=b['pixel_values'].to(DEV),input_ids=b['input_ids'].to(DEV),
                attention_mask=b['attention_mask'].to(DEV),
                question_types=b['question_type'].to(DEV).long())
        def score_all():
            s=torch.empty(len(voc),device=DEV)
            with torch.no_grad():
                for st in range(0,len(voc),48):
                    x=L[st:st+48]; k=x.size(0)
                    ov=(None if OV is None else OV.detach().expand(k,-1))
                    m.vision_gating.alpha_override=ov
                    o=m(**{kk:(vv.expand(k,*vv.shape[1:]) if vv.dim()>1 else vv.expand(k))
                           for kk,vv in kw.items()},labels=x)
                    s[st:st+k]=-F.cross_entropy(o.answer_logits.reshape(-1,o.answer_logits.size(-1)).float(),
                        x.reshape(-1),ignore_index=-100,reduction='none').view(x.shape).sum(1)
            return s
        OV=None; base=score_all(); hit_base=int(int(base.argmax())==gi)
        # khop alpha per-patch de cuc dai log-lik cua GOLD
        m.vision_gating.alpha_override=None
        with torch.no_grad():
            _=m(**kw,labels=L[gi:gi+1])
            a0=m.vision_gating.last_alpha.detach().float()
        if a0.dim()==3: a0=a0.squeeze(-1)
        z=torch.logit(a0.clamp(1e-4,1-1e-4)).clone().requires_grad_(True)
        opt=torch.optim.Adam([z],lr=a.lr)
        lab=L[gi:gi+1]
        for _ in range(a.steps):
            opt.zero_grad()
            m.vision_gating.alpha_override=torch.sigmoid(z)
            o=m(**kw,labels=lab)
            loss=F.cross_entropy(o.answer_logits.reshape(-1,o.answer_logits.size(-1)).float(),
                                 lab.reshape(-1),ignore_index=-100)
            loss.backward(); opt.step()
        OV=torch.sigmoid(z).detach()
        fit=score_all(); hit_fit=int(int(fit.argmax())==gi)
        dev=(OV[0]-a0[0]).abs().cpu().numpy()
        FIELDS.append((OV[0]-a0[0]).cpu().numpy())
        sh=np.sort(dev)[::-1]; k10=max(1,len(dev)//10)
        rows.append(dict(idx=j,type=t,hit_base=hit_base,hit_fit=hit_fit,
                         top10_share=float(sh[:k10].sum()/(sh.sum()+1e-9)),
                         dev_mean=float(dev.mean()),dev_max=float(dev.max())))
        m.vision_gating.alpha_override=None
        if (n_+1)%40==0: print(f'  [{MODE}] {n_+1}/{len(idx)}',flush=True)
    D=pd.DataFrame(rows); Fm=np.stack(FIELDS)
    Fc=Fm-Fm.mean(1,keepdims=True); Fn=Fc/(np.linalg.norm(Fc,axis=1,keepdims=True)+1e-9)
    C=Fn@Fn.T; iu=np.triu_indices(len(Fn),1)
    RES[MODE]=dict(D=D,cos=float(C[iu].mean()))
    del m; torch.cuda.empty_cache()

print(f'\n=== n={len(idx)} mau | run87 (TRAIN bang blend — multiply la LECH PHAN BO) ===')
print(f'{"toan tu":<10} {"base":>7} {"sau khop":>9} {"tran":>7} {"cuu duoc":>9} '
      f'{"top10% share":>13} {"cos giua mau":>13}')
for M in ('blend','multiply'):
    D=RES[M]['D']; err=D[D.hit_base==0]
    resc=err.hit_fit.mean() if len(err) else float('nan')
    sub=D[(D.hit_base==0)&(D.hit_fit==1)]
    print(f'{M:<10} {100*D.hit_base.mean():>7.2f} {100*D.hit_fit.mean():>9.2f} '
          f'{100*(D.hit_fit.mean()-D.hit_base.mean()):>+7.2f} {100*resc:>8.1f}% '
          f'{(sub.top10_share.mean() if len(sub) else float("nan")):>13.3f} {RES[M]["cos"]:>+13.4f}')
pd.concat([RES[M]['D'].assign(mode=M) for M in RES]).to_csv(a.out,index=False)
print(f'\nMOC blend da co (FINDINGS_tcvg_mechanism): top10% share 0.161 (cuu duoc) | cos +0.009')
print('DOC: multiply co top10% share CAO HON RO va cos CAO HON RO -> truong THUA va CO CHUNG HUONG')
print('     -> chon loc LA khung dung, va yeu cau (4) alpha gan nhi phan dang cai.')
print('     Neu van khuech tan + truc giao -> chon loc khong phai thu tin hieu nay can, bat ke toan tu.')
