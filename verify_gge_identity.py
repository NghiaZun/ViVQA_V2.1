"""TINH CHAT BAT BUOC CUA NHANH GGE: token_weights = 1 phai TRUNG KHOP TUYET DOI voi baseline.

VI SAO day la phep thu quan trong nhat cua ca huong GGE:
  Neu nhanh token_weights KHONG trung baseline khi trong so deu, thi moi chenh lech EM do duoc
  sau nay se lan ca "doi cach tinh loss" vao "cong cua GGE" — dung cai loi da tao ra con so
  +4.18 gia trong bai bao (so hai cau hinh do o hai ban code khac nhau).
  Trung khop tuyet doi => DOI CHUNG cua GGE chinh la baseline, khong can chay them arm doi chung.

Kiem ca hai che do vi recipe that co dung --answer_weights:
  A. khong answer_weights   B. CO answer_weights (bay: mau so la TONG TRONG SO LOP, khong phai so token)
  C. co label_smoothing (recipe dung 0.1)
  D. trong so KHONG deu thi phai KHAC baseline (neu khong thi co bi bo qua)
"""
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, 'src')
ok_all = True


def check(name, cond, extra=''):
    global ok_all
    print(f'   {"OK " if cond else "SAI"} {name}{extra}')
    ok_all = ok_all and bool(cond)


def baseline(logits, labels, aw, ls):
    return F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1),
                           ignore_index=-100, weight=aw, label_smoothing=ls)


def gge_branch(logits, labels, aw, ls, tw):
    """Ban sao CHINH XAC bieu thuc trong src/model.py (nhanh token_weights)."""
    lpt = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1),
                          ignore_index=-100, weight=aw, label_smoothing=ls,
                          reduction='none').view(labels.size())
    valid = (labels != -100).to(lpt.dtype)
    w = tw.to(lpt.dtype) * valid
    cw = (aw.to(lpt.dtype)[labels.clamp(min=0)] * valid) if aw is not None else valid
    return (lpt * w).sum() / (cw * tw.to(lpt.dtype)).sum().clamp(min=1e-6)


torch.manual_seed(0)
B, T, V = 4, 6, 23
logits = torch.randn(B, T, V, dtype=torch.float64)
labels = torch.randint(0, V, (B, T))
labels[0, 3:] = -100
labels[2, 4:] = -100
labels[3, 1:] = -100
ones = torch.ones(B, T, dtype=torch.float64)

for nm, aw, ls in (('A. khong answer_weights, khong smoothing', None, 0.0),
                   ('B. CO answer_weights', torch.rand(V, dtype=torch.float64) * 3 + 0.2, 0.0),
                   ('C. CO answer_weights + label_smoothing 0.1',
                    torch.rand(V, dtype=torch.float64) * 3 + 0.2, 0.1),
                   ('D. chi label_smoothing 0.1', None, 0.1)):
    b = baseline(logits, labels, aw, ls).item()
    g = gge_branch(logits, labels, aw, ls, ones).item()
    check(nm, abs(b - g) < 1e-12, f'  | baseline {b:.12f} vs GGE {g:.12f} | chenh {abs(b-g):.2e}')

print('\n=== E. trong so KHONG deu thi PHAI khac baseline (chong co bi bo qua am tham)')
aw = torch.rand(V, dtype=torch.float64) * 3 + 0.2
tw = torch.rand(B, T, dtype=torch.float64)
b = baseline(logits, labels, aw, 0.1).item()
g = gge_branch(logits, labels, aw, 0.1, tw).item()
check('trong so ngau nhien -> loss doi', abs(b - g) > 1e-6, f'  | chenh {abs(b-g):.6f}')

print('\n=== F. bat bien theo TY LE: nhan doi moi trong so KHONG duoc doi loss')
g1 = gge_branch(logits, labels, aw, 0.1, tw).item()
g2 = gge_branch(logits, labels, aw, 0.1, tw * 7.3).item()
check('tu chuan hoa (nhan 7.3 -> khong doi)', abs(g1 - g2) < 1e-12, f'  | chenh {abs(g1-g2):.2e}')
print('   -> he qua: che do that bai cua GGE la "khong doi gi", KHONG phai "ha learning rate".')

print('\n=== G. day noi vao model.py va train.py')
MD = open('src/model.py').read()
check('forward nhan token_weights', 'token_weights: Optional[torch.Tensor] = None' in MD)
check('nhanh token_weights ton tai', 'elif token_weights is not None:' in MD)
check('mau so dung TONG TRONG SO LOP', '(_cw * token_weights.to(_lpt.dtype)).sum()' in MD)
check('nhanh CE thuong van con nguyen', 'else:\n                answer_loss = F.cross_entropy(' in MD)

print('\n' + ('TAT CA QUA.' if ok_all else '>>> CO MUC SAI — DUNG, SUA TRUOC KHI CHAY.'))
sys.exit(0 if ok_all else 1)
