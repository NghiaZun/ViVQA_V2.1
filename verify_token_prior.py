"""CHOT: --token_prior_gamma co THUC SU doi hanh vi, va gamma=0 co TRUNG KHOP baseline khong?
Repo nay tung co 5 co bi ho thanh no-op im lang. Khong tin co nao chua qua chot."""
import sys, torch, torch.nn.functional as F
sys.path.insert(0,'src')
import train as T
from model import DeterministicVQA
from dataset import VQAGenDataset
from torch.utils.data import DataLoader, Subset
from transformers import AutoProcessor
DEV='cuda'; VM='google/siglip-base-patch16-224'
torch.manual_seed(0)
m=DeterministicVQA(vision_model_name=VM,bartpho_model_name='vinai/bartpho-syllable',
  num_fusion_layers=2,fusion_type='text2vision',use_text_lora=True,text_lora_r=16,
  text_lora_alpha=32,use_vision_gate=True,vision_gate_init=1.0,vision_gate_min_alpha=0.0,
  use_type_task=True,use_siglip_pooler=True).to(DEV).eval()
vp=AutoProcessor.from_pretrained(VM)
ds=VQAGenDataset(csv_path='archive/train_split.csv',image_folder='archive/data/images/train',
  vision_processor=vp,tokenizer_name='vinai/bartpho-syllable',max_q_len=32,max_a_len=10,
  include_question_type=True,auto_detect_type=False)
b=next(iter(DataLoader(Subset(ds,list(range(12))),batch_size=12)))
lab=b['labels'].to(DEV); qt=b['question_type'].to(DEV).long()
kw=dict(pixel_values=b['pixel_values'].to(DEV),input_ids=b['input_ids'].to(DEV),
        attention_mask=b['attention_mask'].to(DEV),labels=lab,question_types=qt)
tb=T._build_token_prior('archive/train_split.csv',m.tokenizer)

def tw(gamma):
    if gamma<=0: return None
    w=torch.ones_like(lab,dtype=torch.float32)
    q=qt.detach().cpu().tolist()
    for i in range(lab.size(0)):
        for j in range(lab.size(1)):
            if lab[i,j].item()==-100: continue
            p=tb.get((int(q[i]),j))
            if p is not None: w[i,j]=max(1.0-gamma*p,1e-3)
    return w.to(DEV)

print(f'{"gamma":>7} {"answer_loss":>13} {"w trung binh":>13} {"w nho nhat":>11}')
base=None
for g in (0.0,0.5,1.0):
    with torch.no_grad():
        o=m(**kw,token_weights=tw(g))
    al=float(o.answer_loss); w=tw(g)
    if g==0.0: base=al
    wm = 1.0 if w is None else float(w[(lab!=-100)].mean())
    wn = 1.0 if w is None else float(w[(lab!=-100)].min())
    print(f'{g:>7} {al:>13.6f} {wm:>13.4f} {wn:>11.4f}')
with torch.no_grad():
    o1=m(**kw,token_weights=torch.ones_like(lab,dtype=torch.float32).to(DEV))
print(f'\nCHOT TRUNG KHOP: gamma=0 (None) = {base:.8f}')
print(f'                 token_weights=1 deu = {float(o1.answer_loss):.8f}')
print(f'                 lech = {abs(base-float(o1.answer_loss)):.2e}  ->',
      'TRUNG KHOP' if abs(base-float(o1.answer_loss))<1e-5 else '!! LECH — co che tu lam lech baseline')
with torch.no_grad(): o2=m(**kw,token_weights=tw(1.0))
print(f'\nCHOT CO TAC DUNG: |loss(gamma=1) - loss(gamma=0)| = {abs(float(o2.answer_loss)-base):.6f} ->',
      'CO CHAY' if abs(float(o2.answer_loss)-base)>1e-4 else '!! NO-OP')
n=sum(1 for k,v in tb.items() if v>0.95)
print(f'\nbang prior: {len(tb)} (loai,vi tri), {n} vi tri XAC DINH — uoc TU DU LIEU, khong hardcode')
