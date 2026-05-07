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


# ── Budget + style planning ───────────────────────────────────────

def compute_generation_plan(result_csv, train_csv):
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
    max_share        = CFG.get('gen_max_type_share', 0.25)
    min_error_rate   = CFG.get('gen_min_error_rate', 0.3)   # chỉ gen nếu sai >= 30%
    total_budget     = CFG['gen_budget_total']
    budget_min       = CFG['gen_budget_min']
    budget_max       = CFG['gen_budget_max']

    res  = pd.read_csv(result_csv)
    tdf  = pd.read_csv(train_csv)
    total_train = len(tdf)
    train_col   = 'type' if 'type' in tdf.columns else 'question_type'

    # Tên cột type trong result.csv
    res_type_col = 'question_type' if 'question_type' in res.columns else 'type'

    plan = {}
    for t in TYPES:
        type_int  = TYPE_INT_MAP[t]
        sub       = res[res[res_type_col] == t]
        train_sub = tdf[tdf[train_col] == type_int]

        if sub.empty:
            continue

        type_em = sub['exact_match'].mean() * 100

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
        type_budget = max(0, min(type_budget, max_budget))

        # Phân budget xuống từng answer
        for a in ans_stats:
            a['budget'] = max(1, int(a['errors'] / total_errors * type_budget))

        plan[t] = {
            'n_images'      : type_budget,
            'target_answers': ans_stats,
            'em'            : type_em,
            'note'          : f'{len(ans_stats)} answers cần gen | {total_errors} lần sai',
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

# COCO-style prompt suffix: ảnh candid ngoài đời, không studio
_COCO_SUFFIX = (
    'candid photograph, real-world scene, natural lighting, '
    'amateur snapshot, slightly cluttered background, '
    'no text, no watermark, no border'
)

_COCO_CONTEXTS = [
    'in a park',
    'on a city street',
    'inside a home',
    'in a kitchen',
    'at a market',
    'in a living room',
    'outdoors on grass',
    'near a road',
    'in a backyard',
    'at a zoo',
]


def _build_count_sample(answer, question):
    ans = str(answer).strip().lower()
    count = VN_NUM_MAP.get(ans)
    if count is None:
        try:    count = int(ans)
        except: count = random.randint(1, 5)
    count = max(0, min(count, 10))
    obj   = random.choice(COUNT_OBJECTS)
    ctx   = random.choice(_COCO_CONTEXTS)
    prompt = (
        f"exactly {EN_NUM.get(count, str(count))} {obj} {ctx}, "
        f"clearly visible and countable, {_COCO_SUFFIX}"
    )
    return {'prompt': prompt,
            'question': question,
            'answer': VN_NUM_WORD.get(count, str(count))}


def _build_color_sample(answer, question):
    ans      = answer.strip().lower()
    color_en = VN_COLOR_EN.get(ans, ans)
    obj      = random.choice(COLOR_OBJECTS)
    ctx      = random.choice(_COCO_CONTEXTS)
    prompt   = (
        f"a {color_en} {obj} {ctx}, "
        f"color clearly visible, {_COCO_SUFFIX}"
    )
    return {'prompt': prompt,
            'question': question,
            'answer': answer}


def _build_location_sample(answer, question):
    ans     = answer.strip().lower()
    prep_en = None
    for vn_key, en_val in VN_LOC_EN.items():
        if vn_key in ans:
            prep_en = en_val
            break
    subj, anchor = random.choice(LOC_ANCHORS)
    if prep_en is None:
        prompt = f'a {subj} and a {anchor} in the same scene, {_COCO_SUFFIX}'
    else:
        prompt = f'a {subj} {prep_en} a {anchor}, {_COCO_SUFFIX}'
    return {'prompt': prompt,
            'question': question,
            'answer': answer}


def _build_object_sample(answer, question):
    ctx = random.choice(_COCO_CONTEXTS)
    prompt = f'{answer} {ctx}, {_COCO_SUFFIX}'
    return {'prompt': prompt, 'question': question, 'answer': answer}


def _build_sample(type_name, answer, question):
    if type_name == 'COUNT':      return _build_count_sample(answer, question)
    elif type_name == 'COLOR':    return _build_color_sample(answer, question)
    elif type_name == 'LOCATION': return _build_location_sample(answer, question)
    else:                         return _build_object_sample(answer, question)


# ── FLUX batch inference ──────────────────────────────────────────

def _generate_flux_batch(prompts: list, seeds: list) -> list:
    global _flux_pipe
    if _flux_pipe is None:
        raise RuntimeError('FLUX pipe chưa load — gọi load_generation_models() trước')

    # COCO: 72.5% landscape (~572x482), 22.5% portrait, 5% square
    # Dùng multiples of 64 gần nhất
    def _pick_wh():
        r = random.random()
        if r < 0.72:  return 576, 448   # landscape
        elif r < 0.94: return 448, 576   # portrait
        else:          return 512, 512   # square

    # Mỗi ảnh dùng seed riêng — tránh batch ra ảnh giống nhau
    generators = [torch.Generator('cpu').manual_seed(s) for s in seeds]

    # schnell: guidance_scale=0.0 (distilled, fixed)
    # dev:     guidance_scale=3.5 (CFG-enabled, prompt adherence tốt hơn)
    is_dev = 'dev' in CFG['flux_model']
    guidance = 3.5 if is_dev else 0.0

    images = []
    for prompt, gen in zip(prompts, generators):
        w, h = _pick_wh()
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
    messages = [{'role': 'user', 'content': [
        {'type': 'image', 'image': image},
        {'type': 'text',  'text': f'Trả lời ngắn gọn bằng tiếng Việt: {question_vi}'},
    ]}]
    text = qwen_proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = qwen_proc(text=[text], images=image_inputs, return_tensors='pt').to('cuda')
    with torch.no_grad():
        out = qwen_model.generate(**inputs, max_new_tokens=20, do_sample=False)
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
    from diffusers import FluxPipeline
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    from ultralytics import YOLO

    cpu_offload = CFG.get('flux_cpu_offload', True)
    do_compile  = CFG.get('flux_compile', False)

    if cpu_offload:
        print('Loading FLUX.1-schnell (cpu_offload mode — low VRAM)...')
        _flux_pipe = FluxPipeline.from_pretrained(CFG['flux_model'], torch_dtype=torch.bfloat16)
        _flux_pipe.enable_model_cpu_offload()
    else:
        print('Loading FLUX.1-schnell (full GPU mode — high VRAM)...')
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

    while generated < n and attempts < max_attempts:

        batch_samples = []
        for _ in range(B):
            row      = pool.sample(1, weights=ans_weights).iloc[0]
            base_ans = row['answer']
            base_q   = row['question']
            try:
                batch_samples.append(_build_sample(type_name, str(base_ans), str(base_q)))
            except Exception:
                pass

        if not batch_samples:
            attempts += B
            continue

        prompts = [s['prompt'] for s in batch_samples]
        seeds   = [loop_idx * 100000 + attempts + i for i in range(len(batch_samples))]
        attempts += len(batch_samples)

        try:
            gen_images = _generate_flux_batch(prompts, seeds)
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
