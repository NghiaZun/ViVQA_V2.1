import gc
import os
import random

import numpy as np
import pandas as pd
import torch

from config import (
    CFG, TYPES, TYPE_INT_MAP,
    VN_NUM_MAP, VN_NUM_WORD, VN_COLOR_EN, VN_LOC_EN, EN_NUM,
    COUNT_OBJECTS, COUNT_SCENES, COLOR_OBJECTS, LOC_ANCHORS,
)

_flux_pipe = None

# ── Vietnamese subject extractor ──────────────────────────────────

# Sorted by length descending so multi-word matches win over single-word
_VN_SUBJECT_EN = {
    'xe đạp': 'bicycle', 'xe máy': 'motorcycle', 'xe buýt': 'bus',
    'xe tải': 'truck', 'xe hơi': 'car', 'máy bay': 'airplane',
    'máy tính': 'laptop', 'điện thoại': 'phone', 'trái cây': 'fruit',
    'con người': 'person', 'đàn ông': 'man', 'phụ nữ': 'woman',
    'trẻ em': 'child', 'em bé': 'baby',
    'chó': 'dog', 'mèo': 'cat', 'chim': 'bird', 'cá': 'fish',
    'vịt': 'duck', 'gà': 'chicken', 'bò': 'cow', 'ngựa': 'horse',
    'lợn': 'pig', 'cừu': 'sheep', 'voi': 'elephant',
    'người': 'person', 'xe': 'car', 'ghế': 'chair', 'bàn': 'table',
    'cốc': 'cup', 'ly': 'cup', 'chai': 'bottle', 'hộp': 'box',
    'sách': 'book', 'túi': 'bag', 'mũ': 'hat', 'áo': 'shirt',
    'váy': 'dress', 'giày': 'shoe', 'bình': 'vase', 'hoa': 'flower',
    'cây': 'tree', 'bóng': 'ball', 'diều': 'kite', 'thuyền': 'boat',
    'tàu': 'train', 'đèn': 'lamp', 'sofa': 'sofa', 'giường': 'bed',
    'bánh': 'cake', 'táo': 'apple', 'cam': 'orange', 'chuối': 'banana',
    'dù': 'umbrella', 'ô': 'umbrella', 'đồng hồ': 'clock',
    'tranh': 'painting', 'ảnh': 'photo', 'gương': 'mirror',
    'tủ': 'cabinet', 'kệ': 'shelf', 'cửa': 'door', 'cửa sổ': 'window',
}
_VN_SUBJECT_EN_SORTED = sorted(_VN_SUBJECT_EN.items(), key=lambda x: -len(x[0]))


def _extract_subject_from_question(question: str) -> str | None:
    """
    Extract the main visual subject from a Vietnamese question and return
    its English translation for use in FLUX prompts.
    e.g. "có bao nhiêu con chó trong ảnh" -> "dog"
         "màu của chiếc áo là gì"          -> "shirt"
         "con mèo đang ngồi ở đâu"         -> "cat"
    """
    q = question.lower().strip()
    for vn, en in _VN_SUBJECT_EN_SORTED:
        if vn in q:
            return en
    return None


# ── Budget + style planning ───────────────────────────────────────

def compute_generation_plan(result_csv, train_csv, loop_idx=0):
    """
    Đọc result.csv (output của eval) để tìm câu hỏi/answer hay sai nhất.
    Budget được phân bổ theo số lần sai thực tế thay vì aggregate EM.

    Với mỗi type:
      - Tính error_count = số lần model sai với ground_truth đó
      - Chỉ target answer có error_rate >= gen_min_error_rate
      - Budget mỗi answer tỉ lệ với error_count
      - Cap tổng budget của type theo gen_max_type_share
    """
    skip_threshold   = CFG.get('gen_skip_em', 70.0)
    gap_budget_scale = CFG.get('gen_gap_budget_scale', 0.0)
    max_share        = CFG.get('gen_max_type_share', 0.25)
    base_min_error_rate = CFG.get('gen_min_error_rate', 0.3)   # chỉ gen nếu sai >= 30%
    min_error_rate_delta = CFG.get('loop_min_error_rate_delta', 0.0)
    min_error_rate = min(0.9, base_min_error_rate + loop_idx * min_error_rate_delta)
    total_budget     = CFG['gen_budget_total']
    budget_min       = CFG['gen_budget_min']
    budget_max       = CFG['gen_budget_max']
    focus_weights    = CFG.get('gen_focus_weights', {})
    dynamic_focus    = CFG.get('gen_focus_dynamic', False)
    focus_min        = CFG.get('gen_focus_min', 0.7)
    focus_max        = CFG.get('gen_focus_max', 1.4)
    loop_budget_decay = CFG.get('loop_budget_decay', 1.0)
    loop_budget_scale = loop_budget_decay ** max(loop_idx - 1, 0)

    res  = pd.read_csv(result_csv)
    tdf  = pd.read_csv(train_csv)
    total_train = len(tdf)
    train_col   = 'type' if 'type' in tdf.columns else 'question_type'

    # Tên cột type trong result.csv
    res_type_col = 'question_type' if 'question_type' in res.columns else 'type'

    # Precompute per-type EM for dynamic focus
    type_em_map = {}
    for t in TYPES:
        sub = res[res[res_type_col] == t]
        if not sub.empty:
            type_em_map[t] = sub['exact_match'].mean() * 100

    # Restrict generation to selected types (optional)
    only_types = CFG.get('gen_only_types')
    top_k = CFG.get('gen_focus_top_k')
    if only_types:
        allowed_types = [t for t in TYPES if t in set(only_types)]
    elif top_k and type_em_map:
        allowed_types = [t for t, _ in sorted(type_em_map.items(), key=lambda x: x[1])][:int(top_k)]
    else:
        allowed_types = TYPES

    if dynamic_focus and type_em_map:
        em_values = list(type_em_map.values())
        em_min = min(em_values)
        em_max = max(em_values)
        em_range = max(1e-6, em_max - em_min)
        dynamic_weights = {
            t: focus_max - (type_em_map.get(t, em_max) - em_min) / em_range * (focus_max - focus_min)
            for t in TYPES
        }
    else:
        dynamic_weights = {}

    plan = {}
    for t in TYPES:
        type_int  = TYPE_INT_MAP[t]
        sub       = res[res[res_type_col] == t]
        train_sub = tdf[tdf[train_col] == type_int]

        if t not in allowed_types:
            type_em = type_em_map.get(t, sub['exact_match'].mean() * 100 if not sub.empty else 0.0)
            plan[t] = {
                'n_images'      : 0,
                'target_answers': [],
                'em'            : type_em,
                'note'          : 'skip (not in focus set)',
            }
            continue

        if sub.empty:
            continue

        type_em = type_em_map.get(t, sub['exact_match'].mean() * 100)

        if type_em >= skip_threshold:
            plan[t] = {
                'n_images'      : 0,
                'target_answers': [],
                'em'            : type_em,
                'note'          : f'EM={type_em:.1f}% >= {skip_threshold} → skip',
            }
            continue

        # Tính error stats per answer
        wrong = sub[sub['exact_match'] == 0]
        ans_stats = []
        for ans, grp in sub.groupby('ground_truth'):
            total = len(grp)
            errors = (grp['exact_match'] == 0).sum()
            error_rate = errors / total
            # Chỉ target answer có trong train (mới có thể gen được câu hỏi)
            in_train = (train_sub['answer'] == ans).any()
            if error_rate >= min_error_rate and in_train:
                ans_stats.append({'answer': ans, 'errors': errors,
                                  'total': total, 'error_rate': error_rate})

        if not ans_stats:
            plan[t] = {
                'n_images'      : 0,
                'target_answers': [],
                'em'            : type_em,
                'note'          : 'không có answer nào đạt ngưỡng lỗi',
            }
            continue

        # Phân bổ budget theo tỉ lệ error_count
        total_errors  = sum(a['errors'] for a in ans_stats)
        current_count = int((tdf[train_col] == type_int).sum())

        # Cap tổng budget của type theo share
        max_budget = int((max_share * total_train - current_count) / (1 - max_share))
        type_budget = max(budget_min, min(budget_max, total_errors * 2))
        type_budget = int(type_budget * loop_budget_scale)
        type_budget = max(0, min(type_budget, max_budget))
        focus_weight = dynamic_weights.get(t, focus_weights.get(t, 1.0))
        type_budget = max(0, int(type_budget * focus_weight))

        # Scale by EM gap vs skip threshold (larger gap → more budget)
        if skip_threshold > 0 and type_em < skip_threshold and gap_budget_scale > 0:
            gap_ratio = (skip_threshold - type_em) / skip_threshold
            gap_scale = 1.0 + gap_budget_scale * gap_ratio
            type_budget = max(0, int(type_budget * gap_scale))

        # Phân budget xuống từng answer
        for a in ans_stats:
            a['budget'] = max(1, int(a['errors'] / total_errors * type_budget))

        plan[t] = {
            'n_images'      : type_budget,
            'target_answers': ans_stats,
            'em'            : type_em,
            'note'          : f'{len(ans_stats)} answers cần gen | {total_errors} lần sai | min_err={min_error_rate:.2f}',
        }

    return plan


def print_plan(plan):
    print(f'\n  {"Type":<12} {"EM":>6} {"Budget":>8}  Note')
    print(f'  {"-"*65}')
    for t, p in plan.items():
        print(f'  {t:<12} {p["em"]:>6.1f} {p["n_images"]:>8}  {p["note"]}')
        for a in p.get('target_answers', [])[:5]:
            print(f'    └ {a["answer"]:<25} sai {a["errors"]}/{a["total"]} '
                  f'({a["error_rate"]*100:.0f}%) → gen {a["budget"]}')
        if len(p.get('target_answers', [])) > 5:
            print(f'    └ ... +{len(p["target_answers"])-5} answers khác')


# ── Prompt builders ───────────────────────────────────────────────

# COCO-style suffix: rút gọn để tránh CLIP truncation
_COCO_SUFFIX = (
    "DSLR 35mm photo, natural lighting, "
    "MS COCO, photojournalism, realistic, "
    "sharp focus, candid, documentary style, "
    "multiple objects, high detail"
)

# Negative prompt: để tách riêng các yếu tố cần tránh
_NEGATIVE_PROMPT = "text, watermark, border, frame"

# Scene templates: mỗi scene có main context + co-occurring objects phổ biến trong COCO
_SCENES = [
    # (context, background objects)
    ('in a park',           'people walking, trees, benches, grass in background'),
    ('on a city street',    'cars parked, buildings, pedestrians, trash cans nearby'),
    ('in a kitchen',        'countertop, cabinets, sink, various kitchen items around'),
    ('in a living room',    'sofa, tv, coffee table, lamps, shelves in background'),
    ('at a dining table',   'plates, cups, chairs, food items scattered around'),
    ('in a backyard',       'fence, patio furniture, plants, shed in background'),
    ('at a market',         'stalls, people shopping, various goods piled nearby'),
    ('on a sidewalk',       'parked bikes, fire hydrant, street signs, people passing'),
    ('in a bedroom',        'bed, dresser, curtains, personal items scattered'),
    ('at a sports field',   'other players, equipment, spectators in background'),
    ('in a zoo enclosure',  'rocks, trees, water feature, fence, other animals nearby'),
    ('near a dining area',  'tables with food, people eating, menus, condiments'),
]


def _scene():
    ctx, bg = random.choice(_SCENES)
    return ctx, bg


def _build_count_sample(answer, question):
    ans = str(answer).strip().lower()
    count = VN_NUM_MAP.get(ans)
    if count is None:
        try:    count = int(ans)
        except: count = random.randint(1, 5)
    count = max(0, min(count, 10))
    obj      = _extract_subject_from_question(question) or random.choice(COUNT_OBJECTS)
    ctx, bg  = _scene()
    prompt = (
        f"a scene {ctx} showing exactly {EN_NUM.get(count, str(count))} {obj}, "
        f"{bg}, each {obj} clearly visible and individually countable, "
        f"{_COCO_SUFFIX}"
    )
    # Generate a simple question that matches the generated image exactly
    vn_obj   = next((vn for vn, en in _VN_SUBJECT_EN_SORTED if en == obj), obj)
    new_q    = f'có bao nhiêu {vn_obj} trong ảnh này'
    return {'prompt': prompt,
            'question': new_q,
            'answer': VN_NUM_WORD.get(count, str(count))}


def _build_color_sample(answer, question):
    ans      = answer.strip().lower()
    color_en = VN_COLOR_EN.get(ans, ans)
    obj      = _extract_subject_from_question(question) or random.choice(COLOR_OBJECTS)
    ctx, bg  = _scene()
    prompt   = (
        f"a scene {ctx}, a {color_en} {obj} prominently visible, "
        f"{bg}, the {color_en} color of the {obj} clearly distinguishable, "
        f"{_COCO_SUFFIX}"
    )
    vn_obj = next((vn for vn, en in _VN_SUBJECT_EN_SORTED if en == obj), obj)
    new_q  = f'màu của {vn_obj} là gì'
    return {'prompt': prompt,
            'question': new_q,
            'answer': answer}


def _build_location_sample(answer, question):
    ans     = answer.strip().lower()
    prep_en = None
    for vn_key, en_val in VN_LOC_EN.items():
        if vn_key in ans:
            prep_en = en_val
            break
    extracted = _extract_subject_from_question(question)
    if extracted:
        # Use extracted subject as the located object, pick random anchor
        subj   = extracted
        anchor = random.choice([a for _, a in LOC_ANCHORS])
    else:
        subj, anchor = random.choice(LOC_ANCHORS)
    ctx, bg = _scene()
    if prep_en is None:
        prompt = (
            f"a scene {ctx} with a {subj} and a {anchor} visible, "
            f"{bg}, their spatial relationship clearly shown, {_COCO_SUFFIX}"
        )
    else:
        prompt = (
            f"a scene {ctx}, a {subj} {prep_en} a {anchor}, "
            f"{bg}, spatial positioning clearly visible, {_COCO_SUFFIX}"
        )
    vn_subj = next((vn for vn, en in _VN_SUBJECT_EN_SORTED if en == subj), subj)
    new_q   = f'{vn_subj} đang ở đâu'
    return {'prompt': prompt,
            'question': new_q,
            'answer': answer}


def _build_object_sample(answer, question):
    ctx, bg = _scene()
    # Try to get English subject: from question extraction, then from answer translation
    subj_en = (
        _extract_subject_from_question(question)
        or _VN_SUBJECT_EN.get(answer.lower().strip())
        or answer
    )
    prompt  = (
        f"a scene {ctx} featuring a {subj_en}, "
        f"{bg}, the {subj_en} is the main subject clearly identifiable, "
        f"{_COCO_SUFFIX}"
    )
    new_q   = 'cái gì là vật chính trong ảnh này'
    return {'prompt': prompt, 'question': new_q, 'answer': answer}


def _build_sample(type_name, answer, question):
    if type_name == 'COUNT':      return _build_count_sample(answer, question)
    elif type_name == 'COLOR':    return _build_color_sample(answer, question)
    elif type_name == 'LOCATION': return _build_location_sample(answer, question)
    else:                         return _build_object_sample(answer, question)


# ── FLUX batch inference ──────────────────────────────────────────

def _load_ref_image(img_id: str) -> 'PIL.Image':
    """
    Load ảnh COCO thật từ image_dir làm reference cho img2img.
    KHÔNG bao giờ trả về None — nếu img_id không load được thì
    fallback sang ảnh random trong image_dir để FLUX luôn chạy img2img.
    """
    from PIL import Image as PILImage

    def _try_load(path):
        try:
            return PILImage.open(path).convert('RGB')
        except Exception:
            return None

    img_dir = CFG['image_dir']

    # 1. Thử load đúng img_id với các extension phổ biến
    for ext in ('.jpg', '.jpeg', '.png', ''):
        path = os.path.join(img_dir, f'{img_id}{ext}')
        if os.path.exists(path):
            img = _try_load(path)
            if img is not None:
                return img
            # File tồn tại nhưng corrupt → thử ext tiếp theo

    # 2. Fallback: chọn random từ image_dir — vẫn đảm bảo COCO style
    #    (không để FLUX chạy text2img vì output sẽ rất khác COCO)
    try:
        all_files = [
            f for f in os.listdir(img_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        if all_files:
            for _ in range(10):  # thử tối đa 10 file random
                candidate = os.path.join(img_dir, random.choice(all_files))
                img = _try_load(candidate)
                if img is not None:
                    return img
    except Exception:
        pass

    # 3. Last resort: tạo ảnh trắng 512x512 còn hơn là None
    #    (FLUX img2img với ảnh trắng sẽ bị dominated bởi prompt — chấp nhận được)
    print(f'  ⚠️  _load_ref_image: không load được ảnh nào từ {img_dir}, dùng blank fallback')
    return PILImage.new('RGB', (512, 512), color=(128, 128, 128))


def _generate_flux_batch(prompts: list, seeds: list,
                         ref_images: list = None) -> list:
    global _flux_pipe
    if _flux_pipe is None:
        raise RuntimeError('FLUX pipe chưa load — gọi load_generation_models() trước')

    generators = [torch.Generator('cpu').manual_seed(s) for s in seeds]
    is_dev     = 'dev' in CFG['flux_model']
    guidance   = 3.5 if is_dev else 0.0
    # use_img2img: chỉ cần flux_use_img2img=True và ref_images có sẵn
    # Bỏ điều kiện is_dev — schnell cũng hỗ trợ img2img qua FluxImg2ImgPipeline
    use_img2img = (
        CFG.get('flux_use_img2img', True)
        and ref_images is not None
        and len(ref_images) > 0
    )

    images = []
    for i, (prompt, gen) in enumerate(zip(prompts, generators)):
        # _load_ref_image đảm bảo không trả None → ref luôn là PIL.Image
        ref = ref_images[i] if use_img2img else None

        if use_img2img and ref is not None:
            # img2img: giữ composition/complexity từ ảnh COCO thật
            # strength=0.85 → 85% noise → content thay đổi theo prompt
            # nhưng scene structure/complexity giữ từ reference
            from PIL import Image as PILImage
            w, h = ref.size
            # Snap to multiples of 64
            w = (w // 64) * 64 or 512
            h = (h // 64) * 64 or 512
            ref_resized = ref.resize((w, h), PILImage.LANCZOS)
            try:
                result = _flux_pipe(
                    prompt=prompt,
                    negative_prompt=_NEGATIVE_PROMPT,
                    image=ref_resized,
                    strength=CFG.get('flux_img2img_strength', 0.85),
                    num_inference_steps=CFG['flux_steps'],
                    guidance_scale=guidance,
                    max_sequence_length=256,
                    generator=gen,
                    output_type='pil',
                )
            except TypeError:
                result = _flux_pipe(
                    prompt=prompt,
                    image=ref_resized,
                    strength=CFG.get('flux_img2img_strength', 0.85),
                    num_inference_steps=CFG['flux_steps'],
                    guidance_scale=guidance,
                    max_sequence_length=256,
                    generator=gen,
                    output_type='pil',
                )
        else:
            # Fallback: text-to-image
            r = random.random()
            w, h = (576, 448) if r < 0.72 else (448, 576) if r < 0.94 else (512, 512)
            try:
                result = _flux_pipe(
                    prompt=prompt,
                    negative_prompt=_NEGATIVE_PROMPT,
                    num_inference_steps=CFG['flux_steps'],
                    guidance_scale=guidance,
                    width=w,
                    height=h,
                    max_sequence_length=256,
                    generator=gen,
                    output_type='pil',
                )
            except TypeError:
                result = _flux_pipe(
                    prompt=prompt,
                    num_inference_steps=CFG['flux_steps'],
                    guidance_scale=guidance,
                    width=w,
                    height=h,
                    max_sequence_length=256,
                    generator=gen,
                    output_type='pil',
                )
        images.append(result.images[0])
    return images


# ── Qwen2-VL verifier ─────────────────────────────────────────────

def _verify_with_qwen(image, question_vi, expected_vi, qwen_model, qwen_proc):
    from qwen_vl_utils import process_vision_info

    prompt = f'Nhìn vào ảnh và trả lời ngắn gọn bằng tiếng Việt: {question_vi}'
    messages = [{'role': 'user', 'content': [
        {'type': 'image', 'image': image},
        {'type': 'text',  'text': prompt},
    ]}]
    text = qwen_proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = qwen_proc(text=[text], images=image_inputs, return_tensors='pt').to('cuda')
    with torch.no_grad():
        out = qwen_model.generate(**inputs, max_new_tokens=32, do_sample=False)
    pred     = qwen_proc.decode(out[0][inputs['input_ids'].shape[1]:],
                                skip_special_tokens=True).strip().lower()
    expected = expected_vi.strip().lower()
    if expected in pred or pred in expected:
        return True
    ta, tb = set(pred.split()), set(expected.split())
    return (len(ta & tb) / max(len(ta), len(tb), 1)) > 0.5


# ── Load / unload ─────────────────────────────────────────────────

def load_generation_models():
    global _flux_pipe
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    from ultralytics import YOLO

    use_img2img = CFG.get('flux_use_img2img', True) and 'dev' in CFG['flux_model']
    if use_img2img:
        from diffusers import FluxImg2ImgPipeline as FluxPipeline
        print(f'Loading FLUX.1-dev img2img (strength={CFG.get("flux_img2img_strength", 0.85)})...')
    else:
        from diffusers import FluxPipeline
        print(f'Loading {CFG["flux_model"]}...')

    cpu_offload = CFG.get('flux_cpu_offload', True)
    do_compile  = CFG.get('flux_compile', False)

    if cpu_offload:
        _flux_pipe = FluxPipeline.from_pretrained(CFG['flux_model'], torch_dtype=torch.bfloat16)
        _flux_pipe.enable_model_cpu_offload()
    else:
        _flux_pipe = FluxPipeline.from_pretrained(
            CFG['flux_model'], torch_dtype=torch.bfloat16,
        ).to('cuda')
        # Flash attention 2 nếu có (giảm VRAM + tăng tốc transformer blocks)
        try:
            _flux_pipe.enable_xformers_memory_efficient_attention()
            print('  xformers enabled')
        except Exception:
            pass

    _flux_pipe.set_progress_bar_config(disable=True)

    # torch.compile tăng throughput ~20-30% trên Ampere+ sau lần warmup đầu
    if do_compile:
        print('  torch.compile FLUX transformer (warmup ~60s lần đầu)...')
        try:
            _flux_pipe.transformer = torch.compile(
                _flux_pipe.transformer,
                mode='reduce-overhead',
                fullgraph=False,
            )
        except Exception as e:
            print(f'  ⚠️  torch.compile failed ({e}), skipping')

    print('Loading Qwen2-VL (verifier)...')
    qwen_proc  = AutoProcessor.from_pretrained(CFG['mllm_model'], trust_remote_code=True)
    qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
        CFG['mllm_model'], torch_dtype=torch.bfloat16,
        device_map='cuda', trust_remote_code=True,
    )

    print('Loading YOLO...')
    yolo = YOLO(CFG['yolo_model'])

    vram = torch.cuda.memory_allocated() / 1024**3
    vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f'✓ Models loaded  |  VRAM: {vram:.1f}/{vram_total:.0f}GB')
    print(f'  batch_size={CFG["flux_batch_size"]}  cpu_offload={cpu_offload}  compile={do_compile}')
    return {'qwen_proc': qwen_proc, 'qwen': qwen_model, 'yolo': yolo}


def unload_generation_models(models):
    global _flux_pipe
    del models['qwen'], models['yolo'], _flux_pipe
    _flux_pipe = None
    gc.collect()
    torch.cuda.empty_cache()
    print('✓ Generation models unloaded')


# ── Main generator ────────────────────────────────────────────────

def generate_images_for_type(type_name, plan_entry, current_train_csv, models, loop_idx):
    """
    Sinh ảnh theo batch flux_batch_size:
      1. Build B samples từ train pool
      2. FLUX sinh B ảnh cùng lúc (batch inference)
      3. Verify từng ảnh bằng YOLO (COUNT) hoặc Qwen (các type còn lại)
    """
    type_int     = TYPE_INT_MAP[type_name]
    n            = plan_entry['n_images']
    loop_img_dir = os.path.join(CFG['aug_root'], f'result_{loop_idx}')
    os.makedirs(loop_img_dir, exist_ok=True)
    B            = CFG['flux_batch_size']

    if plan_entry['n_images'] == 0:
        print(f'  [{type_name}] skip (EM cao, giữ nguyên).')
        return []

    target_answers = plan_entry.get('target_answers', [])
    if not target_answers:
        print(f'  [{type_name}] không có target answers, skip.')
        return []

    df   = pd.read_csv(current_train_csv)
    pool = df[df['type'] == type_int].copy()

    if pool.empty:
        print(f'  [{type_name}] pool rỗng, skip.')
        return []

    # Build weighted sample pool dựa trên target answers + budget từng answer
    # Mỗi answer được chọn theo budget của nó (tỉ lệ errors)
    target_map = {a['answer']: a for a in target_answers}
    pool = pool[pool['answer'].isin(target_map)].copy()
    if pool.empty:
        print(f'  [{type_name}] target answers không có trong train pool, skip.')
        return []

    ans_budgets = pool['answer'].map(lambda a: target_map[a]['budget'])
    ans_weights = ans_budgets.values.astype(float)
    ans_weights = ans_weights / ans_weights.sum()

    new_rows     = []
    generated    = 0
    attempts     = 0
    max_attempts = n * 3

    print(f'  [{type_name}] n={n} | target_ans={len(target_answers)} | pool={len(pool)} | batch={B}')

    ref_same_prob = CFG.get('flux_ref_same_answer_prob', 0.0)

    while generated < n and attempts < max_attempts:

        batch_samples = []
        ref_imgs      = []
        for _ in range(B):
            row      = pool.sample(1, weights=ans_weights).iloc[0]
            base_ans = row['answer']
            base_q   = row['question']
            try:
                sample = _build_sample(type_name, str(base_ans), str(base_q))
                # Sample reference image: ưu tiên cùng answer để giữ style/scene gần COCO
                if random.random() < ref_same_prob:
                    ref_row = row
                else:
                    ref_row = pool.sample(1).iloc[0]
                ref_img = _load_ref_image(str(ref_row['img_id']))
                # Append cả cặp cùng lúc → batch_samples[i] luôn tương ứng ref_imgs[i]
                # _load_ref_image không bao giờ trả None → img2img luôn được dùng
                batch_samples.append(sample)
                ref_imgs.append(ref_img)
            except Exception:
                pass

        if not batch_samples:
            attempts += B
            continue

        prompts = [s['prompt'] for s in batch_samples]
        seeds   = [loop_idx * 100000 + attempts + i for i in range(len(batch_samples))]
        attempts += len(batch_samples)

        try:
            gen_images = _generate_flux_batch(prompts, seeds, ref_images=ref_imgs)
        except Exception as e:
            print(f'    FLUX batch failed: {e}')
            continue

        for sample, gen_img in zip(batch_samples, gen_images):
            if generated >= n:
                break

            verified = False
            if type_name == 'COUNT':
                try:
                    yolo_res   = models['yolo'](np.array(gen_img), verbose=False)
                    n_detected = len(yolo_res[0].boxes)
                    expected   = VN_NUM_MAP.get(sample['answer'].lower().strip())
                    if expected is None:
                        try:    expected = int(sample['answer'])
                        except: expected = -1
                    verified = (expected >= 0 and abs(n_detected - expected) <= 1)
                except Exception:
                    verified = False
            else:
                try:
                    verified = _verify_with_qwen(
                        gen_img, sample['question'], sample['answer'],
                        models['qwen'], models['qwen_proc'],
                    )
                except Exception as e:
                    print(f'    Qwen error: {e}')

            if not verified:
                continue

            new_img_id = f'aug_loop{loop_idx}_{type_name}_{generated:04d}'
            save_path  = os.path.join(loop_img_dir, f'{new_img_id}.jpg')
            gen_img.save(save_path, quality=90)
            new_rows.append({
                'question': sample['question'],
                'answer'  : sample['answer'],
                'img_id'  : new_img_id,
                'type'    : type_int,
            })
            generated += 1

        if generated % 20 == 0 and generated > 0:
            rate = generated / attempts * 100
            print(f'    {generated}/{n} done  (attempts={attempts}, accept={rate:.0f}%)')

    rate = generated / max(attempts, 1) * 100
    print(f'  ✓ [{type_name}] {generated}/{n}  accept={rate:.1f}%  attempts={attempts}')
    return new_rows
