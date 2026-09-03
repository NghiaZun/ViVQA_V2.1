"""XAC MINH bang so cai bang 2x2 tren query cua gate, truoc khi tieu GPU.

query = query_proj([t_cls ; e_type]).  Bon nhanh:
    T2               : ca hai
    gate_no_type_emb : chi t_cls   -> XAO NHAN LOAI phai KHONG doi alpha
    gate_no_text_cls : chi e_type  -> XAO CAU HOI  phai KHONG doi alpha
    (T0 khong co gate)

Hai phep thu doc lap, moi cai bat mot loai no-op:
  A. xao type_ids   -> alpha doi?   (0 nghia la gate khong dung LOAI)
  B. xao text_cls   -> alpha doi?   (0 nghia la gate khong dung NOI DUNG cau hoi)
Du an nay da co 5 lan co im lang khong lam gi, nen khong tin dong print.
"""
import sys, torch
sys.path.insert(0, 'src')
from model import DeterministicVQA

DEV = 'cuda'; VM = 'google/siglip2-base-patch16-224'
B, P, L = 8, 197, 12

def probe(no_type, no_text):
    torch.manual_seed(0)
    m = DeterministicVQA(vision_model_name=VM, bartpho_model_name='vinai/bartpho-syllable',
                         num_fusion_layers=2, fusion_type='text2vision',
                         use_vision_gate=True, vision_gate_init=1.0, vision_gate_min_alpha=0.0,
                         use_type_task=True, use_siglip_pooler=True,
                         gate_no_type_emb=no_type, gate_no_text_cls=no_text).to(DEV).eval()
    with torch.no_grad():
        m.vision_gating.type_embedding.weight.normal_(0, 3.0)
    g = m.vision_gating
    v = torch.randn(B, P, 1024, device=DEV)
    t1 = torch.randn(B, L, 1024, device=DEV)
    t2 = torch.randn(B, L, 1024, device=DEV)          # cau hoi KHAC
    y1 = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], device=DEV)
    y2 = torch.tensor([3, 2, 1, 0, 3, 2, 1, 0], device=DEV)   # loai XAO
    with torch.no_grad():
        _, a_ref = g(v, t1, type_ids=y1)
        _, a_ty  = g(v, t1, type_ids=y2)      # doi LOAI
        _, a_tx  = g(v, t2, type_ids=y1)      # doi CAU HOI
    d_ty = (a_ref - a_ty).abs().max().item()
    d_tx = (a_ref - a_tx).abs().max().item()
    del m; torch.cuda.empty_cache()
    return d_ty, d_tx

print(f'{"nhanh":22s} {"doi khi XAO LOAI":>18s} {"doi khi XAO CAU HOI":>21s}')
for nm, nt, nx in [('T2 (ca hai)', False, False),
                   ('chi t_cls (no_type_emb)', True, False),
                   ('chi e_type (no_text_cls)', False, True)]:
    a, b = probe(nt, nx)
    print(f'{nm:22s} {a:18.6f} {b:21.6f}')

print('\nMONG DOI:')
print('  T2                : ca hai > 0')
print('  chi t_cls         : xao LOAI = 0.000000, xao CAU HOI > 0')
print('  chi e_type        : xao LOAI > 0, xao CAU HOI = 0.000000')
