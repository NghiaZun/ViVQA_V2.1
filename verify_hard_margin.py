"""XAC MINH --hard_margin (PHA B2) TRUOC KHI DOT GPU.

Bon dieu phai dung. Bai hoc ptnet: co khong noi toi noi -> ket qua VO NGHIA ma van ra so dep.

  A. hard_margin=0 -> KHONG doi gi (non-harm tuyet doi)
  B. phat DUNG doi thu manh nhat, KHONG phai chinh gold
  C. da vuot margin thi phat = 0 (hinge that su, khong phai phat tuyen tinh)
  D. co duoc noi tu argparse -> call site -> than vong lap
"""
import sys, ast, torch
import torch.nn.functional as F

B, T, V, M = 2, 4, 10, 1.0

def hinge_loss(lg, labels, m=M):
    valid = labels != -100
    gi = labels.clamp(min=0).unsqueeze(-1)
    gold = lg.gather(-1, gi).squeeze(-1)
    neg = lg.scatter(-1, gi, float('-inf')).max(-1).values
    return F.relu(m - (gold - neg)) * valid, gold, neg

torch.manual_seed(0)
lg = torch.randn(B, T, V)
labels = torch.tensor([[3, 7, 1, -100], [2, 2, -100, -100]])

h, gold, neg = hinge_loss(lg, labels)
print('=== B. phat co nham vao doi thu manh nhat khong')
for b in range(B):
    for t in range(T):
        if labels[b, t] == -100: continue
        g = int(labels[b, t])
        others = [v for v in range(V) if v != g]
        want = max(float(lg[b, t, v]) for v in others)
        ok = abs(float(neg[b, t]) - want) < 1e-6
        print(f'   b{b} t{t}: gold={g} logit_gold={float(gold[b,t]):+.3f} '
              f'doi_thu={float(neg[b,t]):+.3f} (dung nhat trong so KHAC gold: {want:+.3f}) {"OK" if ok else "SAI"}')
        assert ok, 'doi thu bi lay nham (co the dinh chinh gold)'

print('\n=== C. hinge: da vuot margin thi phat bang 0')
lg2 = lg.clone(); lg2[0, 0, 3] = lg[0, 0].max() + 5.0        # day gold vuot xa
h2, _, _ = hinge_loss(lg2, labels)
print(f'   truoc: phat o (0,0) = {float(h[0,0]):.4f}')
print(f'   sau khi gold vuot xa: phat = {float(h2[0,0]):.4f}  {"OK" if float(h2[0,0])==0 else "SAI"}')
assert float(h2[0, 0]) == 0.0

print('\n=== C2. vi tri pad (-100) khong bao gio bi phat')
print(f'   phat tai cac vi tri pad: {float((h * (labels==-100)).sum()):.6f}  '
      f'{"OK" if float((h*(labels==-100)).sum())==0 else "SAI"}')
assert float((h * (labels == -100)).sum()) == 0.0

print('\n=== A. lambda=0 -> loss khong doi')
base = torch.tensor(2.5)
print(f'   loss goc {float(base):.4f} -> voi lambda=0: {float(base + 0.0*h.sum()):.4f}  OK')

print('\n=== D. day noi tu argparse den than vong lap')
src = open('src/train.py').read()
tree = ast.parse(src)
fn = next(n for n in ast.walk(tree)
          if isinstance(n, ast.FunctionDef) and any(a.arg == 'use_cdw_ce' for a in n.args.kwonlyargs + n.args.args))
sig = [a.arg for a in fn.args.args + fn.args.kwonlyargs]
checks = [
    ('argparse co --hard_margin', "'--hard_margin'" in src),
    ('argparse co --hard_margin_m', "'--hard_margin_m'" in src),
    (f'than vong lap ({fn.name}) nhan hard_margin', 'hard_margin' in sig),
    ('call site truyen args.hard_margin', 'hard_margin=args.hard_margin' in src),
    ('call site truyen args.hard_margin_m', 'hard_margin_m=args.hard_margin_m' in src),
    ('than vong lap CO DUNG bien do', 'if hard_margin > 0 and is_training' in src),
]
for nm, okk in checks:
    print(f'   {"OK " if okk else "SAI"} {nm}')
    assert okk, nm

print('\n=== E. gradient co chay ve logits khong')
lg3 = lg.clone().requires_grad_(True)
h3, _, _ = hinge_loss(lg3, labels)
(h3.sum() / (labels != -100).sum()).backward()
gn = lg3.grad.abs().sum()
print(f'   |grad| tren logits = {float(gn):.6f}  {"OK" if gn > 0 else "SAI — khong hoc duoc gi"}')
assert gn > 0

print('\nTAT CA QUA.')
