"""ALPHA NHU BO SINH GIA THUYET, khong phai bo chon.

Be tac da do: "WHEN khong co HOW = 0" — selector biet mau nao de lat (AUROC 0.80) nhung khong biet
lat huong nao (AUROC 0.50), nen moi luat nguong chi cat deu ca rescue lan damage.

Loi thoat kha di: KHONG doan alpha nao dung. Cham diem o NHIEU alpha roi TONG HOP.
Tong hop khong can biet alpha nao dung -> thoat duoc dung cho be tac.

Bang chung mo duong (analysis/mech/mech_s2T2.csv, n=600):
  alpha don tot nhat 71.33% | HOP cua cac alpha 75.83% -> du dia +4.50pp
  va 410/454 mau duoc cuu boi >=2 alpha (chi 44 mau phu thuoc DUNG mot alpha)
  -> neu dap an DUNG duoc nhieu view ung ho trong khi dap an SAI tan mat, bo phieu se lay duoc.

CANH BAO: +4.50 la HOP ORACLE (biet nhan moi chon duoc). Script nay do phan LAY DUOC KHONG NHAN.
DOI CHUNG BAT BUOC: cung so view nhung alpha NGAU NHIEN -> neu bang nhau thi loi ich chi den tu
viec trung binh hoa nhieu lan cham, khong phai tu cau truc cua alpha.
"""
import sys, argparse, unicodedata as ud
import torch, torch.nn.functional as F, numpy as np, pandas as pd
from collections import Counter
sys.path.insert(0,'src')
from probe_evidence_readout import build_model
norm=lambda s: ud.normalize('NFC',str(s)).strip().lower()

p=argparse.ArgumentParser()
p.add_argument('--checkpoint',required=True); p.add_argument('--out',required=True)
p.add_argument('--train_csv',default='archive/train_split_original.csv')
p.add_argument('--test_csv',default='archive/test.csv')
p.add_argument('--image_folder',default='archive/data/images/test')
p.add_argument('--alphas',default='0.9,0.8,0.7,0.6')   # cong them alpha CUA MODEL
p.add_argument('--per_type',type=int,default=200)
p.add_argument('--chunk',type=int,default=48)
a=p.parse_args()
AL=[float(x) for x in a.alphas.split(',')]

from dataset import VQAGenDataset
from torch.utils.data import DataLoader, Subset
from transformers import AutoProcessor
DEV='cuda'; m,sa=build_model(a.checkpoint,DEV); tok=m.tokenizer
tr=pd.read_csv(a.train_csv); tr['an']=tr.answer.map(norm)
TVOC={int(t):sorted(set(g.an)) for t,g in tr.groupby('type')}
LB={}
for t,voc in TVOC.items():
    e=tok(voc,return_tensors='pt',padding='max_length',truncation=True,max_length=10)
    x=e.input_ids.to(DEV).clone(); x[x==tok.pad_token_id]=-100; LB[t]=x
te=pd.read_csv(a.test_csv); gold=te.answer.map(norm)
vp=AutoProcessor.from_pretrained(sa.get('vision_model'))
ds=VQAGenDataset(csv_path=a.test_csv,image_folder=a.image_folder,vision_processor=vp,
  tokenizer_name='vinai/bartpho-syllable',max_q_len=32,max_a_len=10,
  include_question_type=True,auto_detect_type=False)
def logp(lg,lb): return -F.cross_entropy(lg.reshape(-1,lg.size(-1)).float(),lb.reshape(-1),
                       ignore_index=-100,reduction='none').view(lb.shape).sum(1)
rng=np.random.RandomState(0)
idx=[]
for t,g in te.groupby('type'):
    ii=list(g.index)
    if a.per_type and len(ii)>a.per_type: ii=list(rng.choice(ii,a.per_type,replace=False))
    idx+=ii
idx=sorted(int(i) for i in idx)
print(f'{len(idx)} mau | views: alpha model + {AL} | doi chung: {len(AL)} alpha ngau nhien',flush=True)

R=[]
for c_,j in enumerate(idx):
    t=int(te.type.iloc[j])
    if gold.iloc[j] not in TVOC[t]: continue
    L=LB[t]; voc=TVOC[t]; gi=voc.index(gold.iloc[j])
    b=next(iter(DataLoader(Subset(ds,[j]),batch_size=1)))
    pv=b['pixel_values'].to(DEV); ii=b['input_ids'].to(DEV)
    am=b['attention_mask'].to(DEV); qt=b['question_type'].to(DEV).long()
    RND=list(rng.uniform(0.55,0.95,len(AL)))
    S={}
    for tag,av in [('own',None)]+[(f'a{v}',v) for v in AL]+[(f'r{k}',v) for k,v in enumerate(RND)]:
        s=torch.empty(L.size(0),device=DEV)
        with torch.no_grad():
            for st in range(0,L.size(0),a.chunk):
                x=L[st:st+a.chunk]; k=x.size(0)
                m.vision_gating.alpha_override=(None if av is None else torch.full((k,1),av,device=DEV))
                o=m(pixel_values=pv.expand(k,-1,-1,-1),input_ids=ii.expand(k,-1),
                    attention_mask=am.expand(k,-1),labels=x,question_types=qt.expand(k))
                s[st:st+k]=logp(o.answer_logits,x)
        S[tag]=s.float().cpu().numpy()
    m.vision_gating.alpha_override=None
    struct=['own']+[f'a{v}' for v in AL]; rnd=['own']+[f'r{k}' for k in range(len(RND))]
    def agg(keys):
        M=np.stack([S[k] for k in keys])                      # [V, C]
        Z=(M-M.mean(1,keepdims=True))/(M.std(1,keepdims=True)+1e-9)
        votes=Counter(int(np.argmax(M[v])) for v in range(len(keys)))
        top=max(votes.items(), key=lambda kv:(kv[1], Z.mean(0)[kv[0]]))[0]
        return dict(mean=int(np.argmax(Z.mean(0))), vote=top, mx=int(np.argmax(Z.max(0))))
    gs=agg(struct); gr=agg(rnd)
    uni=int(any(int(np.argmax(S[k]))==gi for k in struct))
    R.append(dict(idx=j,type=t,gold_i=gi,
                  base=int(int(np.argmax(S['own']))==gi),
                  s_mean=int(gs['mean']==gi), s_vote=int(gs['vote']==gi), s_max=int(gs['mx']==gi),
                  r_mean=int(gr['mean']==gi), r_vote=int(gr['vote']==gi), r_max=int(gr['mx']==gi),
                  oracle_union=uni))
    if (c_+1)%100==0:
        x=pd.DataFrame(R)
        print(f'  {c_+1}/{len(idx)} base {100*x.base.mean():.2f} | mean {100*x.s_mean.mean():.2f} '
              f'| vote {100*x.s_vote.mean():.2f} | oracle-union {100*x.oracle_union.mean():.2f}',flush=True)

D=pd.DataFrame(R); D.to_csv(a.out,index=False)
from scipy import stats
print(f'\n=== n={len(D)} ===')
print(f'{"luat":<28} {"EM":>7} {"vs base":>9} {"McNemar p":>11}')
print(f'{"base (alpha cua model)":<28} {100*D.base.mean():>7.2f} {"—":>9} {"—":>11}')
for k,lab in (('s_mean','TONG HOP: trung binh z'),('s_vote','TONG HOP: bo phieu'),
              ('s_max','TONG HOP: max'),('r_mean','DOI CHUNG ngau nhien: t.binh'),
              ('r_vote','DOI CHUNG ngau nhien: phieu'),('r_max','DOI CHUNG ngau nhien: max')):
    f=int(((D.base==0)&(D[k]==1)).sum()); br=int(((D.base==1)&(D[k]==0)).sum())
    pp=stats.binomtest(f,f+br,0.5).pvalue if f+br>0 else float('nan')
    print(f'{lab:<28} {100*D[k].mean():>7.2f} {100*(D[k].mean()-D.base.mean()):>+9.2f} {pp:>11.4f}   (sua {f}, hong {br})')
print(f'{"HOP ORACLE (tran, dung nhan)":<28} {100*D.oracle_union.mean():>7.2f} '
      f'{100*(D.oracle_union.mean()-D.base.mean()):>+9.2f}')
print('\nDOC: tong hop co cau truc phai vuot CA base LAN doi chung ngau nhien.')
print('     Neu ~= doi chung -> loi ich chi la trung binh hoa nhieu, alpha khong mang cau truc.')
