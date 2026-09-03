"""XAC MINH --qgnd_lambda TRUOC KHI DOT GPU.

Y TUONG: cau noi thi giac->ngon ngu (vision_proj) chi duoc giam sat boi 314 chuoi dap an, nen anh
cua phep anh xa bi nhot trong bao cua chung. Bang chung:
  - bo 5 dap an khoi train -> mo hinh KHONG BAO GIO noi ra chung (0/3001) DU trie cho phep
  - thay the luon la hang xom ngu nghia: hươu cao cổ -> ngựa vằn 43/46, sáu -> năm, cam -> đỏ/vàng
  - tu tieng Viet chua tung huan luyen xep 311.6/347; dap an DA hoc nhung HIEM xep 182.8; ngau nhien 174
  - "hươu cao cổ" xuat hien 164 LAN trong CAU HOI train ma van khong noi ra duoc lan nao
  - tu vung cau hoi 2336 tu vs tu vung dap an 377 tu

CACH LAM: ep dac trung thi giac TRUOC HOP NHAT du doan cac token noi dung cua cau hoi, cham diem
bang CHINH ma tran embedding cua BARTpho (lm_head.weight, buoc chung voi embedding).

PHAI DUNG:
  A. KHONG THEM THAM SO NAO — so tham so truoc/sau khi bat co phai y het
  B. embedding phai DETACH — neu khong, mat mat duoc thoa man bang cach DI CHUYEN TU thay vi
     di chuyen THI GIAC, tuc nguoc dung huong muc tieu
  C. lay TRUOC hop nhat — sau GCA thi dac trung thi giac da chua thong tin cau hoi, dau phu se
     doc len thay vi phai neo that
  D. nhan KHONG rong va KHONG bao hoa (so token duong moi mau phai > 0 va << K)
  E. lambda = 0 -> hoan toan tro
  F. gradient CO chay ve vision_proj (neu khong thi cau noi khong hoc gi ca)
  G. day noi argparse -> main -> set_qgnd_vocab -> forward
"""
import ast
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, 'src')
TR = open('src/train.py').read()
MD = open('src/model.py').read()
ok_all = True


def check(name, cond, extra=''):
    global ok_all
    print(f'   {"OK " if cond else "SAI"} {name}{extra}')
    ok_all = ok_all and bool(cond)


print('=== A. khong them tham so nao (dung lm_head.weight lam bo phan loai)')
check('khong tao nn.Linear/Parameter moi cho qgnd',
      'qgnd' not in MD.split('def set_qgnd_vocab')[0].split('nn.Linear')[-1][:200]
      and 'self.qgnd_head' not in MD)
check('bo phan loai la lm_head.weight', 'self.lm_head.weight[_qids]' in MD)
check('qgnd_ids dang buffer (di theo device, nam trong state_dict)',
      "self.register_buffer('qgnd_ids'" in MD)

print('\n=== B. embedding phai DETACH (nguoc lai se di chuyen TU thay vi THI GIAC)')
check('co .detach() tren ma tran embedding', 'self.lm_head.weight[_qids].detach()' in MD)

print('\n=== C. lay TRUOC hop nhat (chong doc len thong tin cau hoi)')
seg = MD[MD.index('_ql = getattr(self'):MD.index('_ql = getattr(self') + 1600]
check('dung `vision_features` chu khong phai `fused_vision`',
      'vision_features.float().mean(dim=1)' in seg and 'fused_vision' not in seg)
# vision_features KHONG duoc gan lai thanh ban da fuse o giua
reas = [l for l in MD.split('\n') if 'vision_features =' in l and 'fused' in l]
check('vision_features khong bao gio bi gan = ban da fuse', not reas)

print('\n=== D/E/F. hanh vi so hoc that (mo phong dung bieu thuc trong model.py)')
torch.manual_seed(0)
B, P, D, V, K = 4, 9, 32, 200, 40
lm_w = torch.randn(V, D)
qids = torch.arange(5, 5 + K)
vproj = torch.nn.Linear(D, D)
vis_in = torch.randn(B, P, D)
input_ids = torch.randint(0, V, (B, 12))


def qgnd(vis_in, temp=0.07, detach=True):
    vf = vproj(vis_in)
    E = lm_w[qids]
    E = E.detach() if detach else E
    E = F.normalize(E.float(), dim=-1)
    v = F.normalize(vf.float().mean(dim=1), dim=-1)
    lg = (v @ E.t()) / temp
    tg = (input_ids.unsqueeze(-1) == qids.view(1, 1, -1)).any(dim=1).float()
    npos = tg.sum(dim=1).mean().clamp(min=1.0)
    pw = torch.full((K,), float(K) / float(npos))
    return F.binary_cross_entropy_with_logits(lg, tg, pos_weight=pw), tg


loss, tg = qgnd(vis_in)
pos = tg.sum(1)
print(f'   so token duong moi mau: {pos.tolist()} (K={K})')
check('nhan KHONG rong', pos.sum() > 0, f'  | tong duong {int(pos.sum())}')
check('nhan KHONG bao hoa (khong phai moi token deu duong)', pos.max() < K)
check('mat mat huu han', torch.isfinite(loss).item(), f'  | loss {loss.item():.4f}')

vproj.zero_grad()
loss.backward()
g = vproj.weight.grad.abs().sum().item()
print(f'   |grad| tren vision_proj = {g:.6f}')
check('gradient CHAY VE cau noi (vision_proj)', g > 0)

print('\n=== B2. detach thuc su chan gradient ve embedding')
lm_w2 = lm_w.clone().requires_grad_(True)
E2 = F.normalize(lm_w2[qids].detach().float(), dim=-1)
v2 = F.normalize(vproj(vis_in).float().mean(dim=1), dim=-1)
(v2 @ E2.t()).sum().backward()
check('embedding KHONG nhan gradient', lm_w2.grad is None or lm_w2.grad.abs().sum() == 0)

print('\n=== G. day noi argparse -> main -> set_qgnd_vocab -> forward')
for nm, cond in [
    ("argparse co --qgnd_lambda", "'--qgnd_lambda'" in TR),
    ("main dung tu vung CAU HOI (khong phai dap an)", "read_csv(args.train_csv).question" in TR),
    ("bo N token pho bien nhat", 'args.qgnd_drop_top' in TR),
    ("goi set_qgnd_vocab", 'model.set_qgnd_vocab(_ids' in TR),
    ("chuyen lai .to(device) sau khi them buffer", 'model = model.to(device)' in TR),
    ("in do lon SONG moi epoch", 'QGND -> loss' in TR),
    ("bao dong khi mat mat KHONG chay", 'mat mat KHONG chay' in TR),
    ("mac dinh TAT", "'--qgnd_lambda', type=float, default=0.0" in TR),
]:
    check(nm, cond)

print('\n=== E2. lambda = 0 -> nhanh khong bao gio duoc vao')
check('co guard _ql > 0', '_ql > 0 and _qids is not None' in MD)
check('guard ca vision_features None', 'and vision_features is not None' in MD)

print('\n' + ('TAT CA QUA.' if ok_all else '>>> CO MUC SAI — DUNG, SUA TRUOC KHI CHAY.'))
sys.exit(0 if ok_all else 1)
