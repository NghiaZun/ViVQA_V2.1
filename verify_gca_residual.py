"""XAC MINH --gate_gca_residual TRUOC KHI DOT GPU.

Y TUONG: hien tai GCA va TCVG GIANH VIEC NHAU — ca hai deu la "dieu tiet patch co dieu kien theo
cau hoi". Do duoc (gca_sweep, SigLIP1 seed42):
    gca=1.0 -> T0 72.48, T2 72.31, Delta -0.17
    gca=0.5 -> T0 72.11, T2 72.08, Delta -0.03
    gca=0.0 -> T0 67.78, T2 70.84, Delta +3.07   (n=3: +1.64 +/- 1.31, duong 3/3)
TCVG CO nang luc that, nhung bi GCA che.

CACH BO TRO THAY VI GIANH: cho TCVG DIEU KHIEN luong GCA theo TUNG PATCH.
    gated = v_proj − γ·(1−α)·(v_proj − v_orig)
  γ=0 -> gated = v_proj = T0 (non-harm TUYET DOI tai init)
  γ=1 -> gated = α·v_proj + (1−α)·v_orig  = "gca_strength theo tung patch, hoc duoc"
Dong co truc tiep tu so lieu: giua gca 0.5 va 1.0 gan nhu PHANG (72.11 vs 72.48) -> loi ich cua GCA
khong dong nhat giua cac patch. He so toan cuc khong bat duoc; he so theo patch thi co.

Phai dung:
  A. gamma=0 -> dau ra DUNG BANG v_proj (T0). Khong sai so.
  B. gamma>0 -> dau ra doi, va doi THEO alpha (patch alpha thap bi tra ve gan v_orig hon)
  C. alpha=1 moi patch -> dau ra = v_proj du gamma bang bao nhieu (GCA giu nguyen)
  D. day noi argparse -> DeterministicVQA -> VisionGating
  E. gradient chay ve gamma
"""
import ast, sys, torch
sys.path.insert(0, 'src')
from model import VisionGating

D, B, P, L = 64, 2, 9, 5
torch.manual_seed(0)
v = torch.randn(B, P, D); t = torch.randn(B, L, D); vo = torch.randn(B, P, D)
y = torch.tensor([0, 2])

def build():
    torch.manual_seed(0)
    return VisionGating(hidden_dim=D, num_types=4, init_bias=1.0,
                        gate_gca_residual=True, ln_mode='none').eval()

g = build()
with torch.no_grad():
    out0, a0 = g(v, t, type_ids=y, vision_orig=vo)
gs = VisionGating(hidden_dim=D, num_types=4, init_bias=1.0, ln_mode='none').eval()
# tham chieu: v_proj cua chinh g (dung projection cua g, khong phai cua module khac)
with torch.no_grad():
    v_proj = g.vision_proj(v) if hasattr(g, 'vision_proj') else None

print('=== A. gamma=0 tai init -> dau ra DUNG BANG TCVG da cong bo (khong phai T0)')
print(f'   gamma khoi tao = {float(g.gcares_gamma):.8f}')
with torch.no_grad():
    g.gcares_gamma.fill_(0.0)
    outA, _ = g(v, t, type_ids=y, vision_orig=vo)
    # so voi chinh no khi ep alpha=1 (tuong duong khong gate)
    g.alpha_override = torch.ones(B, P)
    outRef, _ = g(v, t, type_ids=y, vision_orig=vo)
    g.alpha_override = None
d = (outA - outRef).abs().max().item()
print(f'   |chenh| so voi alpha=1 = {d:.10f}   {"OK" if d < 1e-6 else "SAI"}')
assert d < 1e-6

print('\n=== C. alpha=1 moi patch -> gamma khong con anh huong')
with torch.no_grad():
    g.gcares_gamma.fill_(1.0)
    g.alpha_override = torch.ones(B, P)
    outC, _ = g(v, t, type_ids=y, vision_orig=vo)
    g.alpha_override = None
d = (outC - outRef).abs().max().item()
print(f'   |chenh| = {d:.10f}   {"OK" if d < 1e-6 else "SAI"}')
assert d < 1e-6

print('\n=== B. gamma>0 va alpha<1 -> dau ra keo ve phia v_orig, theo dung alpha')
with torch.no_grad():
    g.gcares_gamma.fill_(1.0)
    for av in (0.0, 0.5, 1.0):
        g.alpha_override = torch.full((B, P), av)
        o, _ = g(v, t, type_ids=y, vision_orig=vo)
        g.alpha_override = None
        print(f'   alpha={av:.1f}: |dau ra − (khong gate)| = {(o - outRef).abs().mean().item():.6f}')
with torch.no_grad():
    g.alpha_override = torch.zeros(B, P)
    o0, _ = g(v, t, type_ids=y, vision_orig=vo)
    g.alpha_override = None
    vo_p = g.gcares_proj(vo)
d = (o0 - vo_p).abs().max().item()
print(f'   alpha=0, gamma=1 -> dau ra phai DUNG BANG proj(v_orig): |chenh| = {d:.8f}  {"OK" if d < 1e-5 else "SAI"}')
assert d < 1e-5

print('\n=== D. day noi argparse -> DeterministicVQA -> VisionGating')
tr = open('src/train.py').read(); md = open('src/model.py').read()
for nm, ok in [("argparse co --gate_gca_residual", "'--gate_gca_residual'" in tr),
               ("train.py truyen vao constructor", 'gate_gca_residual=args.gate_gca_residual' in tr),
               ("DeterministicVQA nhan tham so", 'gate_gca_residual: bool = False' in md),
               ("DeterministicVQA luu lai", 'self.gate_gca_residual = gate_gca_residual' in md),
               ("truyen xuong VisionGating", "gate_gca_residual=getattr(self, 'gate_gca_residual', False)" in md),
               ("vision_orig duoc truyen khi bat co", "or getattr(self, 'gate_gca_residual', False)) else None)" in md),
               ("nhanh tron co dung bien do", "gated_vision = v_proj - self.gcares_gamma" in md)]:
    print(f'   {"OK " if ok else "SAI"} {nm}')
    assert ok, nm

print('\n=== E. gradient chay ve gamma')
g2 = build(); g2.gcares_gamma.data.fill_(0.3)
out, _ = g2(v, t, type_ids=y, vision_orig=vo)
out.sum().backward()
gr = float(g2.gcares_gamma.grad.abs().sum())
print(f'   |grad| tren gamma = {gr:.8f}  {"OK" if gr > 0 else "SAI — gamma khong hoc duoc"}')
assert gr > 0

print('\nTAT CA QUA.')
