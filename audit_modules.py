"""KIEM DINH TUNG MODULE bang CAN THIEP, khong doc code.

Repo nay co tien su loi im lang: 5 co ho khoi constructor, bug cham diem beam search (11.4 EM),
R-Drop chi lay vi tri token 0, thu muc anh ViVQA-X (1916 anh trang).
Neu mot trong cac module duoi day sai, no giai thich vi sao chon loc theo patch KHONG BAO GIO chay.
"""
import sys, torch, numpy as np, torch.nn.functional as F
sys.path.insert(0,'src')
from probe_evidence_readout import build_model
from dataset import VQAGenDataset
from torch.utils.data import DataLoader, Subset
from transformers import AutoProcessor
DEV='cuda'
m,sa=build_model('checkpoints_run87/best_model.pt',DEV)
vp=AutoProcessor.from_pretrained(sa.get('vision_model'))
ds=VQAGenDataset(csv_path='archive/test.csv',image_folder='archive/data/images/test',
  vision_processor=vp,tokenizer_name='vinai/bartpho-syllable',max_q_len=32,max_a_len=10,
  include_question_type=True,auto_detect_type=False)
b=next(iter(DataLoader(Subset(ds,list(range(8))),batch_size=8)))
pv=b['pixel_values'].to(DEV); ii=b['input_ids'].to(DEV); am=b['attention_mask'].to(DEV)
lb=b['labels'].to(DEV); qt=b['question_type'].to(DEV).long()
C={}
m.vision_proj.register_forward_hook(lambda mo,i,o: C.__setitem__('vraw',o.detach()))
m.vision_gating.layer_norm.register_forward_hook(lambda mo,i,o: C.__setitem__('post',o.detach()))
_dec={}
def _hook(mod,args,kwargs):
    _dec['ehs']=kwargs.get('encoder_hidden_states'); _dec['mask']=kwargs.get('encoder_attention_mask')
    return args,kwargs
m.decoder.register_forward_pre_hook(_hook,with_kwargs=True)

print("="*70)
print("TEST A — THU TU KHONG GIAN CUA PATCH (neu sai => chon loc theo patch VO NGHIA)")
print("="*70)
with torch.no_grad(): m(pixel_values=pv,input_ids=ii,attention_mask=am,labels=lb,question_types=qt)
v0=C['vraw'].clone()
H=pv.shape[-1]; half=H//2
res={}
for name,sl in (('TREN-TRAI',(slice(0,half),slice(0,half))),('DUOI-PHAI',(slice(half,H),slice(half,H)))):
    pv2=pv.clone(); pv2[:,:,sl[0],sl[1]]=0.0
    with torch.no_grad(): m(pixel_values=pv2,input_ids=ii,attention_mask=am,labels=lb,question_types=qt)
    d=(C['vraw']-v0).norm(dim=-1)[0]                       # [P]
    npad=d.numel()-196
    g=d[npad:].reshape(14,14).cpu().numpy()
    res[name]=g
    q=[g[:7,:7].mean(),g[:7,7:].mean(),g[7:,:7].mean(),g[7:,7:].mean()]
    lab=['tren-trai','tren-phai','duoi-trai','duoi-phai']
    top=lab[int(np.argmax(q))]
    print(f'  che {name:<10}: bien dong theo goc phan tu = ' +
          ' '.join(f'{l}={v:.3f}' for l,v in zip(lab,q)))
    print(f'     -> goc bien dong MANH NHAT: {top}  {"DUNG" if top.upper().replace("-","-")==name.lower().replace("-","-").upper() or top==name.lower() else ""}')
ok = (np.argmax([res['TREN-TRAI'][:7,:7].mean(),res['TREN-TRAI'][:7,7:].mean(),
                 res['TREN-TRAI'][7:,:7].mean(),res['TREN-TRAI'][7:,7:].mean()])==0 and
      np.argmax([res['DUOI-PHAI'][:7,:7].mean(),res['DUOI-PHAI'][:7,7:].mean(),
                 res['DUOI-PHAI'][7:,:7].mean(),res['DUOI-PHAI'][7:,7:].mean()])==3)
print(f'  KET LUAN A: thu tu khong gian {"DUNG" if ok else "!! SAI — day la loi NGHIEM TRONG"}')

print("\n"+"="*70)
print("TEST B — MASK cua decoder co KHOP DO DAI voi encoder_hidden_states khong")
print("="*70)
with torch.no_grad(): m(pixel_values=pv,input_ids=ii,attention_mask=am,labels=lb,question_types=qt)
ehs=_dec['ehs']; mask=_dec['mask']
print(f'  encoder_hidden_states: {tuple(ehs.shape)}')
print(f'  encoder_attention_mask: {tuple(mask.shape)}')
Pv=ehs.shape[1]-ii.shape[1]
print(f'  -> {Pv} token thi giac + {ii.shape[1]} token text')
print(f'  mask phan thi giac toan 1?  {bool((mask[:,:Pv]==1).all())}')
print(f'  mask phan text KHOP attention_mask? {bool((mask[:,Pv:]==am).all())}')
print(f'  KET LUAN B: {"DUNG" if ehs.shape[1]==mask.shape[1] and bool((mask[:,:Pv]==1).all()) and bool((mask[:,Pv:]==am).all()) else "!! SAI"}')

print("\n"+"="*70)
print("TEST C — GCA co THUC SU dung TEXT khong (xao text trong batch)")
print("="*70)
with torch.no_grad(): m(pixel_values=pv,input_ids=ii,attention_mask=am,labels=lb,question_types=qt)
p0=C['post'].clone()
perm=torch.tensor([1,2,3,4,5,6,7,0],device=DEV)
with torch.no_grad(): m(pixel_values=pv,input_ids=ii[perm],attention_mask=am[perm],labels=lb,question_types=qt)
d=(C['post']-p0).norm(dim=-1).mean()
rel=(d/p0.norm(dim=-1).mean()).item()
print(f'  |dam may sau GCA| doi khi xao text: {d:.4f}  (tuong doi {100*rel:.2f}%)')
print(f'  KET LUAN C: {"GCA CO dung text" if rel>0.01 else "!! GCA KHONG dung text — loi nghiem trong"}')

print("\n"+"="*70)
print("TEST D — POSITION EMBEDDING thi giac co duoc cong khong")
print("="*70)
pe=[n for n,_ in m.named_parameters() if 'pos' in n.lower() and 'vision' in n.lower()]
print(f'  tham so vi tri thi giac: {pe if pe else "(nam trong SigLIP dong bang)"}')
sh=v0[0,1:,:] if v0.shape[1]==197 else v0[0]
cs=F.cosine_similarity(sh[:-1],sh[1:],dim=-1).mean()
csr=F.cosine_similarity(sh[torch.randperm(sh.shape[0])],sh,dim=-1).mean()
print(f'  cosine giua patch KE NHAU : {cs:.4f}')
print(f'  cosine giua patch NGAU NHIEN: {csr:.4f}')
print(f'  KET LUAN D: {"co cau truc khong gian" if cs>csr+0.02 else "!! patch ke nhau KHONG giong nhau hon ngau nhien"}')
