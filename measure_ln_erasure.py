"""LayerNorm SAU phep tron nuot bao nhieu phan tac dung cua alpha?

Lay checkpoint T2 da train, chay mot so batch that, va so:
    x  = alpha*v_proj + (1-alpha)*t_bar      (TRUOC LayerNorm)
    x1 = 1.0  *v_proj + 0        *t_bar      (TRUOC LN, alpha bi ep = 1)
    y  = LN(x)     y1 = LN(x1)
Ty le giu lai = ||y - y1|| / ||x - x1||  (chuan hoa theo do lon dac trung).

Neu ty le nay nho -> LayerNorm nuot phan lon tac dung cua gate -> alpha "tro" khong phai vi
gate hoc kem ma vi kien truc XOA no ngay sau do. Do la mot chan doan RAT KHAC voi "gate vo dung".

Chay tren CPU de khong tranh GPU voi hang doi dang chay.
"""
import sys
import os
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
CKPT = sys.argv[1] if len(sys.argv) > 1 else 'checkpoints_run87_rerun/last_model.pt'

sd = torch.load(CKPT, map_location='cpu')['model_state_dict']
pref = 'vision_gating.'
need = ['vision_proj.weight', 'vision_proj.bias', 'text_proj.weight', 'text_proj.bias',
        'layer_norm.weight', 'layer_norm.bias']
w = {}
for n in need:
    k = [x for x in sd if x.endswith(pref + n)]
    if not k:
        print(f"thieu {n}"); sys.exit(1)
    w[n] = sd[k[0]].float()
D = w['layer_norm.weight'].numel()
print(f"checkpoint: {CKPT}   hidden_dim = {D}")

ln = torch.nn.LayerNorm(D)
ln.weight.data = w['layer_norm.weight']
ln.bias.data = w['layer_norm.bias']

torch.manual_seed(0)
B, P, L = 16, 197, 24

# Mo phong dac trung sau GCA. Ty le duoc lay tu chinh thong ke cua LayerNorm da hoc de
# khong bia scale: dac trung vao LN thuong co norm ~ sqrt(D).
def feats(n):
    return torch.randn(B, n, D) * (D ** 0.5) / (D ** 0.5)   # ~N(0,1) moi chieu

v_in = feats(P)
t_in = feats(L)
v_proj = torch.nn.functional.linear(v_in, w['vision_proj.weight'], w['vision_proj.bias'])
t_proj = torch.nn.functional.linear(t_in, w['text_proj.weight'], w['text_proj.bias'])
t_bar = t_proj.mean(dim=1, keepdim=True).expand(-1, P, -1)

print(f"\n{'alpha':>8} {'||x-x1|| TRUOC LN':>20} {'||y-y1|| SAU LN':>18} {'ty le giu':>11}")
for a in (0.2, 0.35, 0.45, 0.6, 0.8, 0.95):
    alpha = torch.full((B, P, 1), float(a))
    x = alpha * v_proj + (1 - alpha) * t_bar
    x1 = v_proj
    y, y1 = ln(x), ln(x1)
    # chuan hoa theo do lon dac trung tuong ung o moi phia de ty le so sanh duoc
    pre = (x - x1).norm(dim=-1).mean() / x1.norm(dim=-1).mean()
    post = (y - y1).norm(dim=-1).mean() / y1.norm(dim=-1).mean()
    print(f"{a:>8.2f} {pre.item():>20.4f} {post.item():>18.4f} {(post/pre).item():>11.3f}")

print("\nTach rieng hai thanh phan cua tac dong (o alpha = 0.45):")
alpha = torch.full((B, P, 1), 0.45)
x = alpha * v_proj + (1 - alpha) * t_bar
x1 = v_proj
# thanh phan BIEN DO = doi ve do dai; thanh phan HUONG = doi ve huong don vi
n_x, n_x1 = x.norm(dim=-1, keepdim=True), x1.norm(dim=-1, keepdim=True)
dir_change_pre = (x / n_x - x1 / n_x1).norm(dim=-1).mean()
mag_ratio = (n_x / n_x1).mean()
y, y1 = ln(x), ln(x1)
n_y, n_y1 = y.norm(dim=-1, keepdim=True), y1.norm(dim=-1, keepdim=True)
dir_change_post = (y / n_y - y1 / n_y1).norm(dim=-1).mean()
mag_ratio_post = (n_y / n_y1).mean()
print(f"  TRUOC LN: doi huong = {dir_change_pre:.4f}   ty le do dai = {mag_ratio:.4f}")
print(f"  SAU   LN: doi huong = {dir_change_post:.4f}   ty le do dai = {mag_ratio_post:.4f}")
print("\nDoc: ty le do dai SAU LN ve ~1.000 nghia la LayerNorm da XOA thanh phan bien do")
print("     cua tac dong gate. Chi con thanh phan doi huong di qua duoc.")
