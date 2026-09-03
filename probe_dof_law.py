"""NGUYEN LY: tran oracle do DO NHAY CUC BO cua mang voi lop bi can thiep, KHONG do thong tin.

Bay hien tuong da do khong phai bay that bai roi rac — chung la MOT hien tuong:
  tran lon (+7.08) | tran TANG theo tu do bo toi uu (60->80 buoc, blend->multiply, 16->197 dof)
  truong khop TRUC GIAO (cos ~0.005) | luc day HANG SO (~2-4 nat) | cuu duoc <=> push>deficit (98.9%)
  KHONG du doan duoc o moi do phan giai (R2 -0.042 / -0.092 vs control +0.995)
  doi chung XAO tro lien tuc ngang bang

Bat ky nhieu loan NHIEU CHIEU nao len mot bieu dien trung gian, khi toi uu nguoc tu mot muc tieu
VO HUONG, deu dich duoc muc tieu do. Luong dich quyet dinh boi SO BAC TU DO va DO NHAY cua dau ra
voi lop do — khong lien quan den viec nhieu loan ay co ma hoa "patch nao quan trong" hay khong.

DU DOAN KIEM CHUNG DUOC: neu dung, tran KHONG dac thu cho thi giac.
Khop he so per-token tren khoi TEXT — cung dof, cung bo toi uu, cung so buoc — phai cho tran
TUONG DUONG. Va tran phai theo DOF chu khong theo VI TRI can thiep.

Nhanh (moi nhanh dung CHUNG mot bo khop, chi khac vi tri va so token duoc phep dieu chinh):
  vision_all   : toan bo token thi giac            (dof = P)
  vision_half  : mot nua token thi giac chon ngau nhien (dof = P/2)
  vision_16    : 16 token thi giac ngau nhien      (dof = 16)
  text_all     : toan bo token text THAT (khong pad) (dof = L_real)
  text_16      : 16 token text ngau nhien          (dof = 16)
DOC: neu tran(text_16) ~ tran(vision_16) va tran theo DOF -> nguyen ly dung.
"""
import sys, argparse, unicodedata as ud
import torch, torch.nn.functional as F, numpy as np, pandas as pd
sys.path.insert(0,'src')
from probe_evidence_readout import build_model
norm=lambda s: ud.normalize('NFC',str(s)).strip().lower()
p=argparse.ArgumentParser()
p.add_argument('--checkpoint',default='checkpoints_run87/best_model.pt')
p.add_argument('--per_type',type=int,default=50); p.add_argument('--steps',type=int,default=25)
p.add_argument('--lr',type=float,default=0.1); p.add_argument('--out',default='analysis/mech/dof_law.csv')
a=p.parse_args()
from dataset import VQAGenDataset
from torch.utils.data import DataLoader, Subset
from transformers import AutoProcessor
DEV='cuda'
m,sa=build_model(a.checkpoint,DEV); tok=m.tokenizer
# hook: nhan he so len encoder_hidden_states tai cac vi tri duoc chi dinh
STATE={'scale':None}
def _pre(mod,args,kwargs):
    s=STATE['scale']
    if s is not None and 'encoder_hidden_states' in kwargs and kwargs['encoder_hidden_states'] is not None:
        kwargs['encoder_hidden_states']=kwargs['encoder_hidden_states']*s
    return args,kwargs
m.decoder.register_forward_pre_hook(_pre,with_kwargs=True)

tr=pd.read_csv('archive/train_split_original.csv'); tr['an']=tr.answer.map(norm)
TVOC={int(t):sorted(set(g.an)) for t,g in tr.groupby('type')}
te=pd.read_csv('archive/test.csv'); gold=te.answer.map(norm)
vp=AutoProcessor.from_pretrained(sa.get('vision_model'))
ds=VQAGenDataset(csv_path='archive/test.csv',image_folder='archive/data/images/test',
  vision_processor=vp,tokenizer_name='vinai/bartpho-syllable',max_q_len=32,max_a_len=10,
  include_question_type=True,auto_detect_type=False)
LB={}
for t,voc in TVOC.items():
    e=tok(voc,return_tensors='pt',padding='max_length',truncation=True,max_length=10)
    x=e.input_ids.to(DEV).clone(); x[x==tok.pad_token_id]=-100; LB[t]=x
rng=np.random.RandomState(0); idx=[]
for t,g in te.groupby('type'):
    ii=[i for i in g.index if gold.iloc[i] in TVOC[int(t)]]
    idx+=list(rng.choice(ii,min(a.per_type,len(ii)),replace=False))
idx=sorted(int(i) for i in idx)
print(f'{len(idx)} mau | nhanh: vision_all/half/16, text_all/16',flush=True)

rows=[]
for n_,j in enumerate(idx):
    t=int(te.type.iloc[j]); voc=TVOC[t]; gi=voc.index(gold.iloc[j]); L=LB[t]
    b=next(iter(DataLoader(Subset(ds,[j]),batch_size=1)))
    kw=dict(pixel_values=b['pixel_values'].to(DEV),input_ids=b['input_ids'].to(DEV),
            attention_mask=b['attention_mask'].to(DEV),
            question_types=b['question_type'].to(DEV).long())
    Lreal=int(b['attention_mask'][0].sum())
    def score():
        s=torch.empty(len(voc),device=DEV)
        with torch.no_grad():
            for st in range(0,len(voc),48):
                x=L[st:st+48]; k=x.size(0)
                o=m(**{kk:(vv.expand(k,*vv.shape[1:]) if vv.dim()>1 else vv.expand(k))
                       for kk,vv in kw.items()},labels=x)
                s[st:st+k]=-F.cross_entropy(o.answer_logits.reshape(-1,o.answer_logits.size(-1)).float(),
                    x.reshape(-1),ignore_index=-100,reduction='none').view(x.shape).sum(1)
        return s
    STATE['scale']=None; base=score(); hit_base=int(int(base.argmax())==gi)
    with torch.no_grad(): o=m(**kw,labels=L[gi:gi+1])
    # so token thi giac = tong - so token text
    P_total=None
    with torch.no_grad():
        _=m(**kw,labels=L[gi:gi+1])
    # suy ra tu kien truc: vision_for_decoder + text
    Pv=197 if sa.get('use_siglip_pooler',True) else 196
    T=Pv+kw['input_ids'].size(1)
    r=dict(idx=j,type=t,hit_base=hit_base,Pv=Pv,Lreal=Lreal)
    rs=np.random.RandomState(1000+j)
    ARMS={'vision_all':np.arange(Pv),
          'vision_half':rs.choice(Pv,Pv//2,replace=False),
          'vision_16':rs.choice(Pv,16,replace=False),
          'text_all':Pv+np.arange(Lreal),
          'text_16':Pv+rs.choice(Lreal,min(16,Lreal),replace=False)}
    for name,pos in ARMS.items():
        pos_t=torch.as_tensor(np.sort(pos),device=DEV,dtype=torch.long)
        z=torch.zeros(len(pos_t),device=DEV,requires_grad=True)
        opt=torch.optim.Adam([z],lr=a.lr); lab=L[gi:gi+1]
        for _ in range(a.steps):
            opt.zero_grad()
            sc=torch.ones(1,T,1,device=DEV)
            sc=sc.index_put((torch.zeros(len(pos_t),dtype=torch.long,device=DEV),pos_t,
                             torch.zeros(len(pos_t),dtype=torch.long,device=DEV)),
                            torch.sigmoid(z)*2.0)          # he so trong [0,2], 1.0 = khong doi
            STATE['scale']=sc
            o=m(**kw,labels=lab)
            F.cross_entropy(o.answer_logits.reshape(-1,o.answer_logits.size(-1)).float(),
                            lab.reshape(-1),ignore_index=-100).backward(); opt.step()
        with torch.no_grad():
            sc=torch.ones(1,T,1,device=DEV)
            sc=sc.index_put((torch.zeros(len(pos_t),dtype=torch.long,device=DEV),pos_t,
                             torch.zeros(len(pos_t),dtype=torch.long,device=DEV)),
                            torch.sigmoid(z)*2.0)
        STATE['scale']=sc; r[f'hit_{name}']=int(int(score().argmax())==gi); r[f'dof_{name}']=len(pos_t)
        STATE['scale']=None
    rows.append(r)
    if (n_+1)%25==0:
        D=pd.DataFrame(rows)
        print(f'  {n_+1}/{len(idx)} base {100*D.hit_base.mean():.1f} | ' +
              ' '.join(f'{k.split("_")[0][:3]}{k.split("_")[1]} {100*D[f"hit_{k}"].mean():.1f}'
                       for k in ARMS),flush=True)

D=pd.DataFrame(rows); D.to_csv(a.out,index=False)
print(f'\n=== n={len(D)} | can thiep = he so per-token tren encoder_hidden_states ===')
print(f'{"nhanh":<14} {"dof/mau":>8} {"EM":>7} {"tran":>7}")' if False else
      f'{"nhanh":<14} {"dof/mau":>8} {"EM":>7} {"tran":>7}')
print(f'{"base":<14} {"—":>8} {100*D.hit_base.mean():>7.2f} {"—":>7}')
for k in ('vision_16','text_16','vision_half','text_all','vision_all'):
    print(f'{k:<14} {D[f"dof_{k}"].mean():>8.1f} {100*D[f"hit_{k}"].mean():>7.2f} '
          f'{100*(D[f"hit_{k}"].mean()-D.hit_base.mean()):>+7.2f}')
print('\nDOC: tran(text_16) ~ tran(vision_16) va tran tang theo DOF chu khong theo VI TRI')
print('     -> tran oracle do TU DO BO TOI UU, khong do thong tin thi giac. Nguyen ly DUNG.')
print('     Neu tran(vision) >> tran(text) o cung dof -> du dia dac thu thi giac la THAT.')
