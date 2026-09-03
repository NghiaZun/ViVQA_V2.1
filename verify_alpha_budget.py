"""XAC MINH --gate_alpha_budget: alpha co THAT SU canh tranh giua cac patch khong.

Van de cau truc dang sua: hien tai moi alpha_i la mot sigmoid DOC LAP, nen "lam noi" mot vung
KHONG lam mo vung khac. Hau qua do duoc tren SigLIP1: LOCATION alpha = 0.9995 voi SD = 0.0002,
OBJECT 0.9989 — model dat het alpha ~ 1, tuc TU CHOI cong. Va lam phang alpha chi mat 0.03,
vi chua bao gio co phep phan bo nao de ma pha.

Ba dieu phai dung:
  A. GATE CU: day diem cua MOT patch len cao -> cac patch KHAC khong doi (khong canh tranh).
  B. GATE MOI: day diem cua MOT patch len cao -> cac patch khac PHAI GIAM (co canh tranh).
  C. GATE MOI khong the bao hoa: tong alpha bi giu quanh P*m du diem co lech den dau.
"""
import sys, torch
sys.path.insert(0, 'src')
from model import VisionGating

DEV = 'cuda'; B, P, D, L = 4, 197, 1024, 12
torch.manual_seed(0)
v = torch.randn(B, P, D, device=DEV)
t = torch.randn(B, L, D, device=DEV)
y = torch.tensor([0, 1, 2, 3], device=DEV)

def build(budget):
    torch.manual_seed(0)
    return VisionGating(hidden_dim=D, num_types=4, init_bias=1.0,
                        gate_alpha_budget=budget, gate_budget_init=0.72).to(DEV).eval()

for name, budget in [('GATE CU  (sigmoid doc lap)', False), ('GATE MOI (ngan sach)      ', True)]:
    g = build(budget)
    with torch.no_grad():
        _, a0 = g(v, t, type_ids=y)
        v2 = v.clone(); v2[:, 5, :] *= 6.0          # day MANH dac trung cua patch 5
        _, a1 = g(v2, t, type_ids=y)
    d5 = (a1[:, 5] - a0[:, 5]).abs().mean().item()
    others = torch.cat([a1[:, :5], a1[:, 6:]], 1) - torch.cat([a0[:, :5], a0[:, 6:]], 1)
    print(f'{name} | patch 5 doi {d5:.5f} | cac patch KHAC doi {others.abs().mean().item():.6f}'
          f' | tong alpha {a0.sum(1).mean().item():.2f} -> {a1.sum(1).mean().item():.2f}')

print(f'\nMONG DOI: gate CU  -> cac patch khac doi ~ 0        (khong canh tranh)')
print(f'          gate MOI -> cac patch khac doi > 0        (co canh tranh)')
print(f'          gate MOI -> tong alpha giu quanh P*m = {197 * 0.72:.1f}')
