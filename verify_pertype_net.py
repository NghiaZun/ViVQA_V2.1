"""XAC MINH --gate_pertype_net: moi loai dung MOT mang gate rieng.

Hai dieu phai dung, neu khong thi co la no-op (da xay ra 5 lan trong du an nay):
  A. Luc KHOI TAO ca 4 ban GIONG HET nhau -> alpha phai Y HET model chung
     (epoch 0 khong duoc lech, moi phan ky ve sau moi la HOC duoc).
  B. Sau khi lam LECH mang cua mot loai -> alpha cua RIENG loai do doi, cac loai khac KHONG doi.
"""
import sys, torch
sys.path.insert(0, 'src')
from model import DeterministicVQA
DEV='cuda'; VM='google/siglip2-base-patch16-224'; B,P,L=8,197,12

def build(pertype):
    torch.manual_seed(0)
    return DeterministicVQA(vision_model_name=VM, bartpho_model_name='vinai/bartpho-syllable',
        num_fusion_layers=2, fusion_type='text2vision', use_vision_gate=True,
        vision_gate_init=1.0, vision_gate_min_alpha=0.0, use_type_task=True,
        use_siglip_pooler=True, gate_pertype_net=pertype).to(DEV).eval()

torch.manual_seed(1)
v=torch.randn(B,P,1024,device=DEV); t=torch.randn(B,L,1024,device=DEV)
y=torch.tensor([0,1,2,3,0,1,2,3],device=DEV)

m0,m1=build(False),build(True)
with torch.no_grad():
    _,a0=m0.vision_gating(v,t,type_ids=y)
    _,a1=m1.vision_gating(v,t,type_ids=y)
print(f'A. |alpha(chung) - alpha(pertype)| luc khoi tao = {(a0-a1).abs().max().item():.8f}   (phai = 0)')
print(f'   so mang gate: {len(m1.vision_gating.gate_nets)}')

with torch.no_grad():   # lam lech RIENG mang cua loai 2 (COLOR)
    for p_ in m1.vision_gating.gate_nets[2].parameters(): p_.add_(torch.randn_like(p_)*0.5)
    _,a2=m1.vision_gating(v,t,type_ids=y)
d=(a1-a2).abs().amax(dim=1)
print('\nB. sau khi lam lech mang cua loai 2 (COLOR):')
for i in range(B):
    print(f'   mau {i} loai={y[i].item()}  |delta alpha| = {d[i].item():.6f}'
          f'{"   <- phai > 0" if y[i].item()==2 else "   <- phai = 0"}')
