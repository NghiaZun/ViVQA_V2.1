"""XAC MINH --ogm_ge TRUOC KHI DOT GPU.

Y TUONG (xem khoi ghi chu dai trong src/train.py):
  GCA va TCVG giành viec nhau. Tat GCA luc suy luan thi TCVG dang gia +1.64 EM (duong 3/3 seed)
  => nang luc CO THAT nhung bi che. Trong literature day la modality competition / greedy learner,
  o cap MODULE thay vi cap modality.

  --gca_dropout da thu roi va AM (-0.35, COLOR -4.48) vi no doi FORWARD PASS.
  OGM-GE giu FORWARD Y NGUYEN, chi ham TOC DO HOC cua nhanh dang thang:
      ratio = score(GCA don doc) / score(TCVG don doc)
      coeff = 1 - tanh(alpha * relu(ratio))            [chi khi ratio > 1]
      grad(flamingo_fusion) *= coeff  + nhieu Gauss

PHAI DUNG:
  A. _ogm_gold_score dung dinh nghia (trung binh P(token gold) tren vi tri hop le)
  B. _ogm_coeff KHOP cong thuc code goc OGM-GE
  C. _ogm_apply_grads chi dong vao flamingo_fusion, KHONG dong param khac
  D. nhieu GE tinh std tren grad GOC (truoc khi ham), khong phai sau
  E. day noi argparse -> main -> run_one_epoch_deterministic -> ogm_state
  F. vi tri ap dung: SAU scaler.unscale_(), TRUOC clip_grad_norm_
     (sau clip thi clip keo norm nguoc len -> can thiep bi triet tieu, null tu gay ra)
  G. BAY GHI DE: forward() set _fl.gca_strength tu self.gca_strength moi lan goi
     -> probe PHAI set o cap MODEL. Set o cap layer se bi xoa ngay trong chinh forward do.
  H. probe TRA LAI nguyen trang (alpha_override, gca_strength)
"""
import ast
import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, 'src')
from train import _ogm_gold_score, _ogm_coeff, _ogm_apply_grads   # noqa: E402

TR = open('src/train.py').read()
MD = open('src/model.py').read()
ok_all = True


def check(name, cond):
    global ok_all
    print(f'   {"OK " if cond else "SAI"} {name}')
    ok_all = ok_all and bool(cond)


print('=== A. _ogm_gold_score = trung binh P(token gold) tren vi tri labels != -100')
torch.manual_seed(0)
B, T, V = 3, 4, 7
logits = torch.randn(B, T, V)
labels = torch.tensor([[1, 2, -100, -100],
                       [3, -100, -100, -100],
                       [0, 5, 6, -100]])
p = F.softmax(logits.float(), dim=-1)
man = [p[b, t, labels[b, t]].item() for b in range(B) for t in range(T) if labels[b, t] != -100]
manual = sum(man) / len(man)
got = _ogm_gold_score(logits, labels)
print(f'   thu cong {manual:.8f} | ham {got:.8f} | so vi tri hop le {len(man)}')
check('khop tay tinh', abs(manual - got) < 1e-6)
check('bo qua -100 (khong lay het 12 vi tri)', len(man) == 6)
check('labels toan -100 -> None (khong chia 0)', _ogm_gold_score(logits, torch.full((B, T), -100)) is None)

print('\n=== B. _ogm_coeff khop cong thuc code goc OGM-GE')
# goc: coeff = 1 - tanh(args.alpha * relu(ratio))
for ratio, alpha in [(1.0, 0.5), (1.3, 0.5), (2.0, 0.1), (1.05, 0.3), (-1.0, 0.5)]:
    ref = 1 - math.tanh(alpha * max(ratio, 0.0))
    got = _ogm_coeff(ratio, alpha)
    print(f'   ratio={ratio:5.2f} alpha={alpha:.2f} -> coeff {got:.6f} (tham chieu {ref:.6f})')
    check(f'ratio={ratio} alpha={alpha}', abs(ref - got) < 1e-9)
print('   LUU Y: tai ratio=1.0 he so DA < 1 (khong lien tuc o 1). Do la dung ban goc.')
print(f'   -> alpha quyet dinh muc ham NEN tai diem hoa: alpha=0.5 ham {1-math.tanh(0.5):.2f}, '
      f'alpha=0.1 ham {1-math.tanh(0.1):.2f}. Chon alpha theo ratio do duoc.')


class Stub(nn.Module):
    """Model gia: co flamingo_fusion va cac param KHAC, de kiem tra pham vi tac dong."""
    def __init__(self):
        super().__init__()
        self.flamingo_fusion = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])
        self.vision_gating = nn.Linear(4, 4)
        self.encoder = nn.Linear(4, 4)


print('\n=== C. _ogm_apply_grads chi dong vao flamingo_fusion')
m = Stub()
m(torch.randn(2, 4)) if False else None
out = sum(p.sum() for p in m.parameters())
out.backward()
before = {n: p.grad.clone() for n, p in m.named_parameters()}
n_touched = _ogm_apply_grads(m, coeff=0.5, noise=0.0)
print(f'   so tensor bi dong: {n_touched}')
fl_scaled = all(torch.allclose(p.grad, before[n] * 0.5)
                for n, p in m.named_parameters() if n.startswith('flamingo_fusion'))
other_same = all(torch.equal(p.grad, before[n])
                 for n, p in m.named_parameters() if not n.startswith('flamingo_fusion'))
check('grad flamingo_fusion bi nhan dung 0.5', fl_scaled)
check('grad cac module KHAC khong doi mot chut nao', other_same)
check('so tensor dong = so param cua flamingo_fusion',
      n_touched == len(list(m.flamingo_fusion.parameters())))

print('\n=== C2. coeff = 1.0 va noise = 0 -> hoan toan tro (khong ton chi phi, khong doi gi)')
m2 = Stub(); sum(p.sum() for p in m2.parameters()).backward()
b2 = {n: p.grad.clone() for n, p in m2.named_parameters()}
check('tra ve 0 = khong lam gi', _ogm_apply_grads(m2, coeff=1.0, noise=0.0) == 0)
check('grad y nguyen', all(torch.equal(p.grad, b2[n]) for n, p in m2.named_parameters()))

print('\n=== D. nhieu GE tinh std tren grad GOC (truoc khi ham), dung thu tu code goc')
torch.manual_seed(0)
m3 = Stub()
(sum((p ** 2).sum() for p in m3.flamingo_fusion.parameters())).backward()
g0 = next(iter(m3.flamingo_fusion.parameters())).grad.clone()
sd_goc = g0.std().item()
torch.manual_seed(1)
_ogm_apply_grads(m3, coeff=0.5, noise=1.0)
g1 = next(iter(m3.flamingo_fusion.parameters())).grad
torch.manual_seed(1)
ref = g0 * 0.5 + torch.randn_like(g0) * (sd_goc * 1.0)
print(f'   std(grad goc) = {sd_goc:.6f} | |chenh so voi tham chieu| = {(g1 - ref).abs().max():.2e}')
check('khop cong thuc goc * coeff + N(0, std_goc * he_so)', torch.allclose(g1, ref, atol=1e-6))

print('\n=== E. day noi argparse -> main -> run_one_epoch -> ogm_state')
for nm, cond in [
    ("argparse co --ogm_ge", "'--ogm_ge'" in TR),
    ("argparse co --ogm_ge_noise", "'--ogm_ge_noise'" in TR),
    ("argparse co --ogm_ge_every", "'--ogm_ge_every'" in TR),
    ("ogm_state tao khi bat co", "ogm_state = {} if (args.ogm_ge > 0 or args.gge_diag > 0 or args.gge > 0) else None" in TR),
    ("truyen ogm_ge vao run_one_epoch", "ogm_ge=args.ogm_ge," in TR),
    ("truyen ogm_state vao run_one_epoch", "ogm_state=ogm_state," in TR),
    ("run_one_epoch nhan tham so", "ogm_ge=0.0, ogm_ge_noise=0.0" in TR),
    ("co in ratio/coeff moi epoch", "OGM-GE -> ratio tb" in TR),
]:
    check(nm, cond)

print('\n=== F. VI TRI ap dung: sau scaler.unscale_(), TRUOC clip_grad_norm_')
lines = TR.split('\n')
idx_apply = [i for i, l in enumerate(lines) if '_apply_ogm_to_grads()' in l and 'def ' not in l]
print(f'   so diem goi _apply_ogm_to_grads: {len(idx_apply)} (mong doi 4: 2 nhanh trong vong lap '
      f'+ 2 nhanh flush cuoi epoch)')
check('goi o ca 4 duong di toi optimizer.step', len(idx_apply) == 4)
good = True
for i in idx_apply:
    nxt = next((l for l in lines[i + 1:i + 3] if l.strip()), '')
    prv = next((l for l in reversed(lines[max(0, i - 3):i]) if l.strip()), '')
    if 'clip_grad_norm_' not in nxt:
        good = False
        print(f'   !! dong {i+1}: dong ngay sau KHONG phai clip_grad_norm_ -> {nxt.strip()[:60]}')
    if 'unscale_' not in prv and 'else:' not in prv:
        good = False
        print(f'   !! dong {i+1}: dong ngay truoc bat thuong -> {prv.strip()[:60]}')
check('moi diem goi deu nam NGAY TRUOC clip_grad_norm_', good)

print('\n=== G. BAY GHI DE: forward set _fl.gca_strength tu self.gca_strength moi lan goi')
check('model.py co ghi de gca_strength trong forward',
      'for _fl in self.flamingo_fusion:' in MD and '_fl.gca_strength = _gsb' in MD)
check('probe set o CAP MODEL (model.gca_strength), khong phai cap layer',
      "model.gca_strength = 0.0" in TR)
check('probe KHONG set truc tiep tren fusion layer (se bi forward xoa)',
      "fl.gca_strength = 0.0" not in TR)

print('\n=== H. probe TRA LAI nguyen trang (dung try/finally, khong ro ri trang thai)')
src = TR[TR.index('def _ogm_branch_scores'):TR.index('def _ogm_apply_grads')]
tree = ast.parse(src).body[0]
n_try = sum(1 for n in ast.walk(tree) if isinstance(n, ast.Try) and n.finalbody)
check('ca hai probe boc trong try/finally', n_try == 2)
check('khoi phuc alpha_override', 'vg.alpha_override = _prev_ov' in src)
check('khoi phuc gca_strength', 'model.gca_strength = _prev_gs' in src)
check('probe chay duoi no_grad', src.count('with torch.no_grad():') == 2)

print('\n=== I. mac dinh TAT -> khong doi hanh vi cu mot chut nao')
check('--ogm_ge mac dinh 0.0', "'--ogm_ge', type=float, default=0.0" in TR)
check("ogm_state = None khi tat", "or args.gge > 0) else None" in TR)
check('nhanh tinh probe co guard ogm_ge > 0', 'is_training and ogm_ge > 0 and ogm_state is not None' in TR)

print('\n' + ('TAT CA QUA.' if ok_all else '>>> CO MUC SAI — DUNG, SUA TRUOC KHI CHAY.'))
sys.exit(0 if ok_all else 1)
