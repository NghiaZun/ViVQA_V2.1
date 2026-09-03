"""XAC MINH --gate_blend_vorig, truoc khi tieu 3 lan train.

Hai dieu phai dung:
  A. TAI INIT: dau ra phai TRUNG KHIT gate chuan (blend_gamma = -6 -> sigmoid ~ 0.0025).
     Neu lech, moi so do duoc ve sau la do KHOI TAO khac chu khong phai do hoc.
     (Da suyt sai: zeros(1) -> sigmoid(0) = 0.5, tron nua-nua ngay tu buoc 0.)
  B. SAU KHI day gamma len: doi v_orig PHAI lam doi dau ra. Neu khong, vision_orig khong
     thuc su duoc dung -> co la no-op am tham (da xay ra 5 lan trong du an nay).
"""
import sys, torch
sys.path.insert(0, 'src')
from model import VisionGating

DEV = 'cuda'; B, P, D, L = 6, 197, 1024, 12
torch.manual_seed(0)
v  = torch.randn(B, P, D, device=DEV)
t  = torch.randn(B, L, D, device=DEV)
o1 = torch.randn(B, P, D, device=DEV)
o2 = torch.randn(B, P, D, device=DEV)      # v_orig KHAC
y  = torch.tensor([0, 1, 2, 3, 0, 1], device=DEV)

def build(flag):
    torch.manual_seed(0)
    return VisionGating(hidden_dim=D, num_types=4, init_bias=1.0,
                        gate_blend_vorig=flag).to(DEV).eval()

g0, g1 = build(False), build(True)
with torch.no_grad():
    out0, _ = g0(v, t, type_ids=y)
    out1, _ = g1(v, t, type_ids=y, vision_orig=o1)
print(f'A. |chuan - vorig(init)| max = {(out0 - out1).abs().max().item():.8f}   (phai ~ 0)')
print(f'   blend_gamma = {g1.blend_gamma.item():.4f}  (init 0 -> non-harm tuyet doi)')

with torch.no_grad():
    g1.blend_gamma.fill_(0.5)
    a, _ = g1(v, t, type_ids=y, vision_orig=o1)
    b, _ = g1(v, t, type_ids=y, vision_orig=o2)
print(f'\nB. sau khi day gamma len 0.5:')
print(f'   |dau ra voi v_orig KHAC NHAU| max = {(a - b).abs().max().item():.6f}   (phai > 0)')
print('\nMONG DOI: A ~ 0.00000000 (non-harm tai init) va B > 0 (v_orig that su duoc dung)')
