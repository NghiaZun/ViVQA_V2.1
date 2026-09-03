"""IQG CO THUC SU DUNG ANH KHONG? — doi chung dang le phai co TRUOC khi chay.

NGHI VAN: IQG loss di tu 4.6565 xuong 0.0462. Sinh lai mot cau hoi 13 token CHI tu dac trung
thi giac ma dat loss 0.046 la dieu KHONG THE — anh khong quyet dinh duoc cach dien dat cua cau hoi.

NGUYEN NHAN NGHI NGO: dau vao decoder la shift_tokens_right(cau hoi), tuc TEACHER FORCING.
Decoder duoc dua chinh cac token goc da dich phai, nen du doan token t+1 chi can nhin tien to
1..t -> GIAI DUOC MA KHONG CAN ANH. Chot chong gian lan cu (bo nho = thi giac TRUOC hop nhat)
chan duoc kieu "chep tu fused_vision", nhung KHONG kiem duoc rang vision co dong gop gi.

PHEP THU: tren chinh checkpoint da train, tinh IQG loss trong ba che do
    A. bo nho thi giac THAT
    B. bo nho thi giac XAO TRON theo batch (anh cua mau khac)
    C. bo nho thi giac ZERO
Neu A ~ B ~ C thi vision KHONG duoc dung, va ket qua 0/199 khong noi len dieu gi ve co che —
no chi noi rang co che CHUA BAO GIO CHAY.
Neu A << B, C thi vision that su duoc dung va 0/199 la ket qua that.
"""
import sys
import torch
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader

sys.path.insert(0, 'src')
from model import DeterministicVQA, shift_tokens_right   # noqa: E402
from dataset import VQAGenDataset                        # noqa: E402
from transformers import AutoProcessor                   # noqa: E402

CKPT = 'checkpoints_iqg02_s0/last_model.pt'
DEV = 'cuda'
ck = torch.load(CKPT, map_location='cpu', weights_only=False)
sa = ck.get('args', {})
print(f"iqg_lambda trong checkpoint: {sa.get('iqg_lambda')}")
assert sa.get('iqg_lambda', 0) > 0, 'checkpoint nay KHONG train voi IQG'

model = DeterministicVQA(
    vision_model_name=sa.get('vision_model', 'google/siglip-base-patch16-224'),
    bartpho_model_name='vinai/bartpho-syllable',
    num_fusion_layers=sa.get('num_fusion_layers', 2), fusion_type='text2vision',
    use_text_lora=True, text_lora_r=16, text_lora_alpha=32,
    use_vision_gate=True, vision_gate_init=sa.get('vision_gate_init', 1.0),
    vision_gate_min_alpha=0.0, use_type_task=sa.get('use_type_loss', True),
    use_siglip_pooler=sa.get('use_siglip_pooler', True)).to(DEV).eval()
model.load_state_dict(ck['model_state_dict'], strict=False)
for p in model.parameters():
    p.requires_grad_(False)

vp = AutoProcessor.from_pretrained(sa.get('vision_model', 'google/siglip-base-patch16-224'))
ds = VQAGenDataset(csv_path='archive/val_split.csv', image_folder='archive/data/images/train',
                   vision_processor=vp, tokenizer_name='vinai/bartpho-syllable',
                   max_q_len=32, max_a_len=10, include_question_type=True, auto_detect_type=False)
dl = DataLoader(ds, batch_size=16, shuffle=False, num_workers=2)
pad = model.config.pad_token_id


def iqg_loss(mode, nb=30):
    """Tinh lai DUNG bieu thuc IQG trong model.py, chi doi bo nho thi giac."""
    tot = n = 0
    for i, b in enumerate(dl):
        if i >= nb:
            break
        pv = b['pixel_values'].to(DEV); ii = b['input_ids'].to(DEV)
        am = b['attention_mask'].to(DEV)
        with torch.no_grad():
            patch = model._extract_vision(pv) if hasattr(model, '_extract_vision') else None
            if patch is None:
                out = model.vision_encoder(pixel_values=pv)
                patch = out.last_hidden_state
            vf = model.vision_proj(patch)
            if getattr(model, 'use_siglip_pooler', False):
                g = model.global_proj(out.pooler_output) if hasattr(model, 'global_proj') else None
                if g is not None:
                    vf = torch.cat([g.unsqueeze(1), vf], dim=1)
            if mode == 'xao':
                vf = vf[torch.randperm(vf.size(0), device=vf.device)]
            elif mode == 'zero':
                vf = torch.zeros_like(vf)
            qlbl = ii.clone(); qlbl[qlbl == pad] = -100; qlbl[am == 0] = -100
            qdec = shift_tokens_right(qlbl.clone().masked_fill(qlbl == -100, pad),
                                      pad, model.config.decoder_start_token_id)
            mmask = torch.ones(vf.size(0), vf.size(1), device=vf.device, dtype=torch.long)
            do = model.decoder(input_ids=qdec, attention_mask=None,
                               encoder_hidden_states=vf, encoder_attention_mask=mmask)
            lg = model.lm_head(do.last_hidden_state)
            l = F.cross_entropy(lg.reshape(-1, lg.size(-1)), qlbl.reshape(-1), ignore_index=-100)
        tot += float(l); n += 1
    return tot / n


print('\ntinh IQG loss tren val, 3 che do bo nho thi giac:')
a = iqg_loss('that'); b = iqg_loss('xao'); c = iqg_loss('zero')
print(f'   A. thi giac THAT  : {a:.4f}')
print(f'   B. thi giac XAO   : {b:.4f}   (chenh so voi A: {b-a:+.4f})')
print(f'   C. thi giac ZERO  : {c:.4f}   (chenh so voi A: {c-a:+.4f})')
print('\nDOC:')
if max(b, c) - a < 0.05:
    print('   A ~ B ~ C  ->  VISION KHONG DUOC DUNG.')
    print('   Teacher forcing khien nhiem vu giai duoc bang tien to cau hoi. IQG hoc mo hinh')
    print('   NGON NGU cua cau hoi, khong hoc noi thi giac voi ngon ngu.')
    print('   => ket qua 0/199 KHONG bac bo gia thuyet; co che chua bao gio chay.')
else:
    print('   A << B, C  ->  vision THAT SU duoc dung, va 0/199 la ket qua that.')
