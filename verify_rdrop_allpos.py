"""XAC MINH --rdrop_all_pos (PHA B3) TRUOC KHI DOT GPU.

Sua duy nhat MOT dieu so voi ban goc: KL tinh tren MOI vi tri token thay vi chi vi tri 0.
Quy uoc detach giu nguyen, de chi co dung mot bien thay doi.

Phai dung:
  A. co TAT -> ra DUNG con so cu, tung chu so (khong lam hong hanh vi cu)
  B. cai bay "mau": hai phan phoi GIONG HET o vi tri 0 nhung KHAC o vi tri 1
       -> KL cu = 0 (mu hoan toan), KL moi > 0
  C. vi tri pad (-100) khong duoc tinh
  D. day noi argparse -> call site -> than vong lap
  E. gradient chay ve logits
"""
import ast, torch
import torch.nn.functional as F

B, T, V = 3, 4, 12
torch.manual_seed(0)

def kl_old(a1, a2):
    l1 = a1[:, 0, :].float(); l2 = a2[:, 0, :].float()
    p1 = F.softmax(l1, -1).clamp(min=1e-8); p2 = F.softmax(l2, -1).clamp(min=1e-8)
    return 0.5 * (F.kl_div(p1.log(), p2.detach(), reduction='batchmean') +
                  F.kl_div(p2.log(), p1.detach(), reduction='batchmean'))

def kl_new(a1, a2, labels):
    v = (labels != -100).float()
    lp1 = F.log_softmax(a1.float(), -1); lp2 = F.log_softmax(a2.float(), -1)
    kl = 0.5 * ((lp1.exp() * (lp1 - lp2.detach())).sum(-1) +
                (lp2.exp() * (lp2 - lp1.detach())).sum(-1))
    return (kl * v).sum() / v.sum().clamp(min=1)

a1 = torch.randn(B, T, V); a2 = torch.randn(B, T, V)
labels = torch.tensor([[3, 7, 1, -100], [2, 5, -100, -100], [4, 4, 9, 2]])

print('=== A. co TAT -> con so cu khong doi')
print(f'   KL cu = {float(kl_old(a1, a2)):.8f}  (nhanh else giu nguyen ma nguon, khong dung toi)')

print('\n=== B. CAI BAY "mau": giong het o vi tri 0, khac o vi tri 1')
b1 = torch.randn(B, T, V); b2 = b1.clone()
b2[:, 1, :] = torch.randn(B, V)            # chi vi tri 1 khac nhau
o = float(kl_old(b1, b2)); n = float(kl_new(b1, b2, labels))
print(f'   KL CU  (chi vi tri 0) = {o:.8f}   <- mu, tuong hai model giong nhau')
print(f'   KL MOI (moi vi tri)   = {n:.8f}   <- thay khac biet')
assert o < 1e-6 and n > 1e-3, 'ban sua khong bat duoc khac biet ngoai vi tri 0'
print('   OK: day chinh la truong hop COLOR — moi dap an mo dau bang "mau".')

print('\n=== C. vi tri pad khong duoc tinh')
c1 = torch.randn(B, T, V); c2 = c1.clone()
# hang 2 CO nhan hop le o vi tri 3 (labels[2] = [4,4,9,2]) — chi pha o hang 0 va 1,
# la hai hang that su co pad tai vi tri 3. (Ban dau toi pha ca ba hang -> test tu sai.)
c2[0:2, 3, :] = torch.randn(2, V) * 50
k_pad = float(kl_new(c1, c2, labels))
c3 = c1.clone()  # khong doi gi
k_ref = float(kl_new(c1, c3, labels))
lab_all = torch.full((B, T), 1)             # coi moi vi tri deu hop le
k_all = float(kl_new(c1, c2, lab_all))
print(f'   nhieu KHONG LO chi o vi tri pad: KL = {k_pad:.8f} (moc {k_ref:.8f})')
print(f'   cung nhieu do, neu coi pad la hop le: KL = {k_all:.8f}')
assert abs(k_pad - k_ref) < 1e-6 and k_all > 1.0, 'pad dang bi tinh vao KL'
print('   OK: pad bi loai dung.')

print('\n=== D. day noi argparse -> call site -> than vong lap')
src = open('src/train.py').read()
tree = ast.parse(src)
fn = next(n for n in ast.walk(tree)
          if isinstance(n, ast.FunctionDef) and any(a.arg == 'use_rdrop' for a in n.args.args + n.args.kwonlyargs))
sig = [a.arg for a in fn.args.args + fn.args.kwonlyargs]
for nm, okk in [("argparse co --rdrop_all_pos", "'--rdrop_all_pos'" in src),
                (f'than vong lap ({fn.name}) nhan rdrop_all_pos', 'rdrop_all_pos' in sig),
                ('call site truyen args.rdrop_all_pos', 'rdrop_all_pos=args.rdrop_all_pos' in src),
                ('than vong lap CO RE NHANH', 'if rdrop_all_pos:' in src),
                ('nhanh cu VAN CON', "l1 = outputs.answer_logits[:, 0, :].float()" in src)]:
    print(f'   {"OK " if okk else "SAI"} {nm}')
    assert okk, nm

print('\n=== E. gradient chay ve logits')
g1 = torch.randn(B, T, V, requires_grad=True); g2 = torch.randn(B, T, V, requires_grad=True)
kl_new(g1, g2, labels).backward()
print(f'   |grad| = {float(g1.grad.abs().sum() + g2.grad.abs().sum()):.6f}')
assert g1.grad.abs().sum() > 0 and g2.grad.abs().sum() > 0

print('\nTAT CA QUA.')
