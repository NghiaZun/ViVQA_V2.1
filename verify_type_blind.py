"""XAC MINH cờ --gate_type_blind that su co tac dung, truoc khi tieu 2 x 4.5 gio train.

Du an nay da co 5 lan cờ im lang khong lam gi (double zero-init, CE long sai if, LoRA bi
unfreeze huy, eval nham checkpoint, EMA khong luu). Nen cờ nao cung phai do bang so.

Phep thu (quyet dinh, khong the giai thich vong vo):
  gate mu loai  <=>  XAO NHAN LOAI KHONG DUOC LAM DOI alpha MOT CHUT NAO.
  - blind=False : xao loai PHAI lam doi alpha  (neu khong doi thi type_ids von da khong duoc dung)
  - blind=True  : xao loai PHAI cho alpha Y HET (sai so 0.0)
"""
import sys, torch
sys.path.insert(0, 'src')
from model import DeterministicVQA

DEV = 'cuda'; VM = 'google/siglip2-base-patch16-224'
B, P, L = 8, 197, 12
torch.manual_seed(0)

def alphas(blind):
    torch.manual_seed(0)
    m = DeterministicVQA(vision_model_name=VM, bartpho_model_name='vinai/bartpho-syllable',
                         num_fusion_layers=2, fusion_type='text2vision',
                         use_vision_gate=True, vision_gate_init=1.0, vision_gate_min_alpha=0.0,
                         use_type_task=False, use_siglip_pooler=True,
                         gate_type_blind=blind).to(DEV).eval()
    # LUU Y: vision_bias la MOT scalar dung chung (tcvg_type_bias=False trong cau hinh paper),
    # nen duong duy nhat loai anh huong alpha la e_type trong query. Cho type_embedding tach xa
    # nhau ro rang, neu khong thi khong phan biet duoc co dung loai hay khong.
    with torch.no_grad():
        m.vision_gating.type_embedding.weight.normal_(0, 3.0)
    g = m.vision_gating
    v = torch.randn(B, P, 1024, device=DEV)
    t = torch.randn(B, L, 1024, device=DEV)
    t1 = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], device=DEV)
    t2 = torch.tensor([3, 2, 1, 0, 3, 2, 1, 0], device=DEV)   # xao loai
    with torch.no_grad():
        _, a1 = g(v, t, type_ids=t1)
        _, a2 = g(v, t, type_ids=t2)
    d = (a1 - a2).abs().max().item()
    del m; torch.cuda.empty_cache()
    return d, a1.mean().item()

for blind in (False, True):
    d, mu = alphas(blind)
    print(f'gate_type_blind={str(blind):5s} | |alpha(loai) - alpha(loai xao)| max = {d:.6f} | alpha tb {mu:.4f}')

print('\nMONG DOI: blind=False -> max > 0 ro rang (gate CO dung loai)')
print('          blind=True  -> max = 0.000000 dung bang 0 (gate KHONG dung loai)')
