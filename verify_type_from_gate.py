"""XAC MINH --type_from_gate_lambda that su doi duong di cua gradient.

Luan diem can chung minh bang so, khong phai bang lap luan:
    type_loss HIEN TAI doc text_cls (model.py ~2944, TRUOC fusion) -> gradient KHONG cham gate.
    type_loss MOI doc gated_vision                                 -> gradient PHAI cham gate.

Phep thu: backward CHI tren type_loss, roi do chuan gradient tren tham so cua vision_gating.
    lambda = 0  ->  gate phai co gradient DUNG BANG 0
    lambda > 0  ->  gate phai co gradient KHAC 0
Neu ca hai deu 0 thi co la no-op am tham (da xay ra 5 lan trong du an nay).
"""
import sys, torch
sys.path.insert(0, 'src')
from model import DeterministicVQA

DEV = 'cuda'; VM = 'google/siglip2-base-patch16-224'
B, L = 4, 12

def grad_on_gate(lam):
    torch.manual_seed(0)
    m = DeterministicVQA(
        vision_model_name=VM, bartpho_model_name='vinai/bartpho-syllable',
        num_fusion_layers=2, fusion_type='text2vision',
        use_vision_gate=True, vision_gate_init=1.0, vision_gate_min_alpha=0.0,
        use_type_task=True, use_siglip_pooler=True,
        type_from_gate_lambda=lam).to(DEV)
    m.train()
    px = torch.randn(B, 3, 224, 224, device=DEV)
    ids = torch.randint(5, 900, (B, L), device=DEV)
    am = torch.ones(B, L, dtype=torch.long, device=DEV)
    lb = torch.randint(5, 900, (B, 6), device=DEV)
    qt = torch.tensor([0, 1, 2, 3], device=DEV)
    out = m(pixel_values=px, input_ids=ids, attention_mask=am, labels=lb, question_types=qt)
    m.zero_grad(set_to_none=True)
    out.type_loss.backward()                      # CHI type_loss, khong phai tong loss

    def norm(prefix):
        t = 0.0
        for n, p in m.named_parameters():
            if n.startswith(prefix) and p.grad is not None:
                t += p.grad.detach().float().pow(2).sum().item()
        return t ** 0.5

    r = dict(gate=norm('vision_gating.'),
             gate_net=norm('vision_gating.gate_net'),
             query_proj=norm('vision_gating.query_proj'),
             flamingo=norm('flamingo'),
             text_head=norm('type_head.'))
    del m; torch.cuda.empty_cache()
    return r

for lam in (0.0, 1.0):
    r = grad_on_gate(lam)
    print(f'\nlambda = {lam}')
    for k, v in r.items():
        print(f'   |grad| {k:12s} = {v:.6e}')

print('\nMONG DOI:')
print('  lambda=0 -> gate / gate_net / query_proj / flamingo deu = 0.000000e+00')
print('  lambda=1 -> ca bon deu > 0  (type_loss gio chay qua gate va qua Flamingo)')
print('  text_head > 0 o ca hai (dau cu tren text_cls van hoat dong)')
