"""XAC MINH --patch_self_attn.
A. TAI INIT: tra ve v NGUYEN VEN (out_proj zero-init) -> non-harm tuyet doi.
B. SAU khi lam lech out_proj: doi MOT patch -> cac patch KHAC phai doi (patch da noi voi nhau).
   Gate cu KHONG co tinh chat nay: patch chi attend sang TEXT, khong bao gio sang patch khac.
C. forward va generate deu goi _apply_psa (bai hoc tcvg_topk_random: sua mot ben -> lech han).
"""
import sys, torch
sys.path.insert(0,'src')
from model import DeterministicVQA
DEV='cuda'; VM='google/siglip-base-patch16-224'
torch.manual_seed(0)
m = DeterministicVQA(vision_model_name=VM, bartpho_model_name='vinai/bartpho-syllable',
                     num_fusion_layers=2, fusion_type='text2vision', use_vision_gate=True,
                     vision_gate_init=1.0, use_type_task=True, use_siglip_pooler=True,
                     patch_self_attn=True).to(DEV).eval()
v = torch.randn(4, 197, 1024, device=DEV)
with torch.no_grad():
    out0 = m._apply_psa(v)
print(f'A. |v - psa(v)| tai init = {(v-out0).abs().max().item():.8f}   (phai = 0)')
with torch.no_grad():
    torch.nn.init.normal_(m.psa_out.weight, std=0.05)
    v2 = v.clone(); v2[:, 7, :] += torch.randn(4, 1024, device=DEV) * 3.0  # CONG, khong NHAN: LayerNorm xoa phep nhan
    a, b = m._apply_psa(v), m._apply_psa(v2)
oth = torch.cat([ (a-b)[:, :7], (a-b)[:, 8:] ], 1)
print(f'B. doi patch 7 -> patch 7 doi {(a-b)[:,7].abs().mean().item():.5f} | '
      f'cac patch KHAC doi {oth.abs().mean().item():.5f}   (phai > 0)')
import inspect
src_f = inspect.getsource(DeterministicVQA.forward)
src_g = inspect.getsource(DeterministicVQA.generate)
print(f'C. forward goi _apply_psa: {"_apply_psa" in src_f} | generate goi: {"_apply_psa" in src_g}')
