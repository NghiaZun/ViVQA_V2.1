"""Probe tuyen tinh THEO TUNG LOP cua SigLIP2 dong bang -> du doan DAP AN theo tung loai cau hoi.

Gia thuyet: TCVG chi chon duoc giua nhung gi encoder da bieu dien. Hien chi dung LOP CUOI.
  Mau la thuoc tinh CAP THAP; lop cuoi co the da truu tuong hoa mat no.
  Neu lop som du doan mau tot hon lop cuoi -> co ly do de dua HON HOP LOP CHON THEO LOAI vao decoder.
  Neu lop cuoi tot nhat cho moi loai -> huong nay dong, khong ton mot lan train nao.

Do bang probe TUYEN TINH tren dac trung DONG BANG (khong train encoder, khong train model):
  dac trung = trung binh cac patch cua lop k  ->  Linear  ->  lop dap an (trong tap dap an cua loai do)
Chi so: accuracy tren tap giu ngoai, tach theo loai cau hoi.
"""
import sys, torch, numpy as np, pandas as pd, unicodedata
from PIL import Image
from transformers import AutoModel, AutoProcessor

DEV='cuda'; VM='google/siglip2-base-patch16-224'
LAYERS=[2,4,6,8,10,12]
N_TR, N_TE = 2400, 800
T={0:'OBJECT',1:'COUNT',2:'COLOR',3:'LOCATION'}
norm=lambda s: unicodedata.normalize('NFC',str(s)).strip().lower()

df=pd.read_csv('archive/train_split_original.csv').sample(n=N_TR+N_TE, random_state=0).reset_index(drop=True)
m=AutoModel.from_pretrained(VM).vision_model.to(DEV).eval().half()
pr=AutoProcessor.from_pretrained(VM)

feats={k:[] for k in LAYERS}
with torch.no_grad():
    for i in range(0,len(df),32):
        ims=[Image.open(f'archive/data/images/train/{x}.jpg').convert('RGB') for x in df.img_id[i:i+32]]
        px=pr(images=ims,return_tensors='pt')['pixel_values'].to(DEV).half()
        hs=m(pixel_values=px, output_hidden_states=True).hidden_states   # tuple: 13 phan tu (embed + 12 lop)
        for k in LAYERS:
            feats[k].append(hs[k].float().mean(1).cpu())                # trung binh patch -> [B, D]
X={k:torch.cat(v) for k,v in feats.items()}
del m; torch.cuda.empty_cache()
print(f'{len(df)} anh | dac trung moi lop: {tuple(X[LAYERS[0]].shape)}')

print(f'\n{"loai":10s} {"n_lop":>6s} ' + ' '.join(f'L{k:<5d}' for k in LAYERS))
for t,name in T.items():
    idx=df.index[df.type==t].values
    if len(idx)<200: continue
    ans=[norm(a) for a in df.answer[idx]]
    vocab=sorted(set(ans)); a2i={a:i for i,a in enumerate(vocab)}
    y=torch.tensor([a2i[a] for a in ans])
    ntr=int(len(idx)*0.75)
    row=f'{name:10s} {len(vocab):6d} '
    for k in LAYERS:
        Z=X[k][idx]; Z=(Z-Z[:ntr].mean(0))/Z[:ntr].std(0).clamp(min=1e-3)
        Ztr,Zte=Z[:ntr].to(DEV),Z[ntr:].to(DEV); ytr,yte=y[:ntr].to(DEV),y[ntr:].to(DEV)
        W=torch.zeros(Z.size(1),len(vocab),device=DEV,requires_grad=True)
        b=torch.zeros(len(vocab),device=DEV,requires_grad=True)
        opt=torch.optim.Adam([W,b],lr=0.05,weight_decay=1e-3)
        for _ in range(400):
            opt.zero_grad(); torch.nn.functional.cross_entropy(Ztr@W+b, ytr).backward(); opt.step()
        acc=((Zte@W+b).argmax(1)==yte).float().mean().item()*100
        row+=f'{acc:5.1f} '
    print(row)
print('\n=> neu mot lop SOM cao hon L12 o loai nao do -> lop cuoi da truu tuong hoa mat thong tin do')
print('   => co ly do dua HON HOP LOP CHON THEO LOAI vao decoder')
