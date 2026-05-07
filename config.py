import os
import torch as _torch

def _detect_vram_gb():
    if _torch.cuda.is_available():
        return _torch.cuda.get_device_properties(0).total_memory / 1024**3
    return 0.0

_VRAM_GB = _detect_vram_gb()

# Tự động điều chỉnh theo VRAM
# H100 80GB: batch=32, full GPU, torch.compile
# A100 40GB: batch=16, full GPU, torch.compile
# P100 16GB: batch=4, cpu_offload
if _VRAM_GB >= 60:
    _FLUX_BATCH  = 32
    _CPU_OFFLOAD = False
    _COMPILE     = True
elif _VRAM_GB >= 35:
    _FLUX_BATCH  = 16
    _CPU_OFFLOAD = False
    _COMPILE     = True
else:
    _FLUX_BATCH  = 4
    _CPU_OFFLOAD = True
    _COMPILE     = False

# Root của project (thư mục chứa config.py này)
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
_ARCHIVE      = os.path.join(_PROJECT_ROOT, 'archive')

CFG = {
    # ── Data — đường dẫn relative so với project root ─────────────
    'train_csv'       : os.path.join(_ARCHIVE, 'train_split.csv'),
    'val_csv'         : os.path.join(_ARCHIVE, 'val_split.csv'),
    'image_dir'       : os.path.join(_ARCHIVE, 'data', 'images', 'train'),
    'answer_weights'  : 'answer_weights.json',

    # ── Working dirs ──────────────────────────────────────────────
    'work_dir'        : _PROJECT_ROOT,
    'ckpt_dir'        : os.path.join(_PROJECT_ROOT, 'checkpoints'),
    'aug_root'        : os.path.join(_PROJECT_ROOT, 'aug_images'),

    # ── Fine-tune loop config (main.py) ───────────────────────────
    'initial_epochs'          : 40,
    'finetune_epochs'         : 15,
    'finetune_lr'             : 2e-5,
    'finetune_warmup_epochs'  : 2,
    'finetune_es_patience'    : 8,
    'max_loops'               : 5,

    # ── Training hyperparameters ──────────────────────────────────
    'epochs'                      : 40,
    'lr'                          : 7e-5,
    'weight_decay'                : 0.01,
    'dropout'                     : 0.1,
    'label_smoothing'             : 0.1,
    'gradient_accumulation_steps' : 1,
    'scheduler'                   : 'cosine',
    'warmup_epochs'               : 3,
    'early_stopping'              : True,
    'early_stopping_patience'     : 7,
    'early_stopping_metric'       : 'em',
    'sample_every'                : 5,
    'seed'                        : 42,

    # ── Model / Fusion ────────────────────────────────────────────
    'vision_model'      : 'google/siglip-base-patch16-224',
    'bartpho_model'     : 'vinai/bartpho-syllable',
    'fusion_type'       : 'text2vision',
    'num_fusion_layers' : 2,

    # ── Batch sampling ────────────────────────────────────────────
    # PK sampling tắt: hard-cap K/type sẽ under-train OBJECT (41.6% test)
    # Thay bằng share cap trong compute_generation_plan
    'pk_sampling' : False,
    'pk_p'        : 4,
    'pk_k'        : 8,

    # ── LoRA (text encoder) ───────────────────────────────────────
    'use_text_lora'   : True,
    'text_lora_r'     : 16,
    'text_lora_alpha' : 32,

    # ── Vision gate ───────────────────────────────────────────────
    'use_vision_gate' : True,

    # ── Auxiliary losses ──────────────────────────────────────────
    'use_type_loss'      : True,
    'use_contrastive'    : True,
    'contrastive_lambda' : 0.1,
    'contrastive_temp'   : 0.07,

    # ── Generation — FLUX.1-schnell ───────────────────────────────
    # H100 80GB: FLUX (~23GB) + Qwen (~4GB) = ~27GB, load full lên GPU
    # P100 16GB: cpu_offload để fit 16GB
    'flux_model'      : 'black-forest-labs/FLUX.1-dev',
    'flux_steps'      : 20,  # dev: 20=good, 28=best quality
    'flux_batch_size' : _FLUX_BATCH,
    'flux_cpu_offload': _CPU_OFFLOAD,
    'flux_compile'    : _COMPILE,     # torch.compile ~30% throughput on Ampere+
    'mllm_model'      : 'Qwen/Qwen2-VL-2B-Instruct',
    'yolo_model'      : 'yolov8m.pt',

    # ── Generation budget ─────────────────────────────────────────
    'gen_budget_total': 800,
    'gen_budget_min'  : 80,
    'gen_budget_max'  : 400,
    # Type có EM >= ngưỡng này → bỏ qua generation, giữ performance
    'gen_skip_em'       : 70.0,
    # Chỉ gen cho answer có error_rate >= ngưỡng này
    'gen_min_error_rate': 0.3,
    # Share tối đa của mỗi type sau khi aug (so với total dataset)
    'gen_max_type_share': 0.25,

    # ── Stop condition ────────────────────────────────────────────
    'stop_em_variance': 8.0,
    'stop_em_delta'   : 0.3,
    'stop_gap_max'    : 10.0,
}

TYPES = ['COUNT', 'LOCATION', 'OBJECT', 'COLOR']

TYPE_NAME_MAP = {'COLOR': 'COLOR', 'COUNT': 'COUNT', 'LOCATION': 'LOCATION', 'OBJECT': 'OBJECT'}
TYPE_INT_MAP  = {'OBJECT': 0, 'COUNT': 1, 'COLOR': 2, 'LOCATION': 3}

VN_NUM_MAP = {
    'khong': 0, 'mot': 1, 'hai': 2, 'ba': 3, 'bon': 4, 'nam': 5,
    'sau': 6, 'bay': 7, 'tam': 8, 'chin': 9, 'muoi': 10,
    'không': 0, 'một': 1, 'bốn': 4, 'năm': 5,
    'sáu': 6, 'bảy': 7, 'tám': 8, 'chín': 9, 'mười': 10,
    **{str(i): i for i in range(11)},
}
VN_COLOR_EN = {
    'đỏ': 'red', 'xanh lam': 'blue', 'xanh dương': 'blue',
    'xanh lá': 'green', 'xanh lá cây': 'green', 'xanh': 'blue',
    'vàng': 'yellow', 'trắng': 'white', 'đen': 'black',
    'nâu': 'brown', 'cam': 'orange', 'tím': 'purple',
    'hồng': 'pink', 'xám': 'gray', 'bạc': 'silver',
    'vàng kim': 'gold', 'be': 'beige', 'kem': 'cream',
}
VN_LOC_EN = {
    'bên trái': 'to the left of', 'bên phải': 'to the right of',
    'phía trên': 'above', 'phía dưới': 'below',
    'trên': 'on top of', 'dưới': 'underneath',
    'trong': 'inside', 'ngoài': 'outside',
    'trước': 'in front of', 'sau': 'behind',
    'giữa': 'in the middle of', 'cạnh': 'next to', 'gần': 'near',
}
EN_NUM = {
    0: 'zero', 1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five',
    6: 'six', 7: 'seven', 8: 'eight', 9: 'nine', 10: 'ten',
}
VN_NUM_WORD = {v: k for k, v in VN_NUM_MAP.items() if not k.isdigit()}

COUNT_OBJECTS = [
    'cats', 'dogs', 'birds', 'people', 'chairs', 'cups', 'books', 'cars',
    'bicycles', 'trees', 'flowers', 'balls', 'bottles', 'apples', 'oranges',
    'bags', 'hats', 'shoes', 'laptops', 'phones', 'candles', 'fish', 'ducks',
]
COUNT_SCENES = [
    'on a wooden table', 'in a park', 'on a clean white background',
    'in a kitchen', 'on a sidewalk', 'in a garden', 'on a shelf',
    'on green grass', 'in a living room', 'at the beach', 'on a farm',
]
COLOR_OBJECTS = [
    'umbrella', 'bag', 'shirt', 'car', 'bicycle', 'flower', 'ball', 'hat',
    'chair', 'cup', 'book', 'shoe', 'jacket', 'door', 'wall', 'bench',
    'box', 'bottle', 'bus', 'kite', 'scarf', 'sofa', 'truck',
]
LOC_ANCHORS = [
    ('cat', 'box'), ('dog', 'car'), ('person', 'bicycle'), ('bird', 'tree'),
    ('ball', 'chair'), ('book', 'table'), ('cup', 'laptop'), ('bag', 'bench'),
    ('bottle', 'basket'), ('phone', 'keyboard'), ('vase', 'window'), ('lamp', 'sofa'),
    ('hat', 'coat rack'), ('apple', 'bowl'),
]

os.makedirs(CFG['ckpt_dir'], exist_ok=True)
os.makedirs(CFG['aug_root'], exist_ok=True)
