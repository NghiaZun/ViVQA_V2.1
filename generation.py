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

def compute_generation_plan(per_type):
    """
    Budget tỉ lệ nghịch với EM (type yếu → nhiều ảnh hơn).
    Type có EM >= gen_skip_em → skip generation, giữ performance (tránh forgetting).
    Budget bị skip sẽ được tái phân bổ cho các type còn lại.

    Style theo gap = F1 - EM (chỉ dùng để chẩn đoán loại lỗi, không phải decision):
      gap < 1  → unambiguous   (sai hoàn toàn — cần ảnh rõ ràng)
      gap < 5  → paraphrase    (gần đúng — cần đa dạng cách diễn đạt)
      gap >= 5 → hard_cases    (đoán đúng phần lớn — cần edge cases)
    """
    skip_threshold = CFG.get('gen_skip_em', 70.0)

    active = {t for t in TYPES if t in per_type and per_type[t]['EM'] < skip_threshold}
    deficits = {t: (100.0 - per_type[t]['EM']) for t in active}
    total_deficit = sum(deficits.values()) or 1.0

    plan = {}
    for t in TYPES:
        if t not in per_type:
            continue

        f1  = per_type[t]['F1']
        em  = per_type[t]['EM']
        gap = f1 - em

        if t not in active:
            plan[t] = {
                'n_images'    : 0,
                'style'       : 'skip',
                'augmentation': 'none',
                'f1'          : f1,
                'em'          : em,
                'gap'         : gap,
                'note'        : f'EM={em:.1f} >= {skip_threshold} → skip để giữ performance',
            }
            continue

        raw_budget = int((deficits[t] / total_deficit) * CFG['gen_budget_total'])
        budget = max(CFG['gen_budget_min'], min(CFG['gen_budget_max'], raw_budget))

        if gap < 1.0:
            style, augmentation = 'unambiguous', 'minimal'
            note = 'gap≈0 → sai hoàn toàn, sinh ảnh rõ ràng không mơ hồ'
        elif gap < 5.0:
            style, augmentation = 'paraphrase', 'spatial'
            note = 'gap nhỏ → partial match, sinh đa dạng cách diễn đạt'
        else:
            style, augmentation = 'hard_cases', 'color_jitter'
            note = 'gap lớn → đã khá tốt, sinh edge cases'

        plan[t] = {
            'n_images'    : budget,
            'style'       : style,
            'augmentation': augmentation,
            'f1'          : f1,
            'em'          : em,
            'gap'         : gap,
            'note'        : note,
        }

    return plan


def print_plan(plan):
    print(f'\n  {"Type":<12} {"F1":>6} {"EM":>6} {"Gap":>6} {"Budget":>8}  Style')
    print(f'  {"-"*65}')
    for t, p in plan.items():
        print(
            f'  {t:<12} {p["f1"]:>6.2f} {p["em"]:>6.2f} '
            f'{p["gap"]:>6.2f} {p["n_images"]:>8}  '
            f'[{p["style"]}] {p["note"]}'
        )


# ── Prompt builders ───────────────────────────────────────────────

def _build_count_sample(answer):
    ans = str(answer).strip().lower()
    count = VN_NUM_MAP.get(ans)
    if count is None:
        try:    count = int(ans)
        except: count = random.randint(1, 5)
    count = max(0, min(count, 10))
    obj   = random.choice(COUNT_OBJECTS)
    scene = random.choice(COUNT_SCENES)
    prompt = (
        f"exactly {EN_NUM.get(count, str(count))} {obj} {scene}, "
        f"photorealistic DSLR photo, sharp focus, natural lighting, "
        f"clearly visible and well-separated, no text, no watermark"
    )
    return {'prompt': prompt,
            'question': f'Có bao nhiêu {obj} trong ảnh?',
            'answer': VN_NUM_WORD.get(count, str(count))}


def _build_color_sample(answer):
    ans      = answer.strip().lower()
    color_en = VN_COLOR_EN.get(ans, ans)
    obj      = random.choice(COLOR_OBJECTS)
    scenes   = [
        f'a {color_en} {obj} on a white table, studio lighting, sharp focus',
        f'a bright {color_en} {obj} in a park, natural daylight',
        f'a vivid {color_en} {obj} in an urban setting, clean background',
        f'close-up of a {color_en} {obj}, minimalist background',
    ]
    return {'prompt': random.choice(scenes) + ', photorealistic, no text, no watermark',
            'question': f'Màu sắc của {obj} trong ảnh là gì?',
            'answer': answer}


def _build_location_sample(answer):
    ans     = answer.strip().lower()
    prep_en, matched = None, None
    for vn_key, en_val in VN_LOC_EN.items():
        if vn_key in ans:
            prep_en, matched = en_val, vn_key
            break
    subj, anchor = random.choice(LOC_ANCHORS)
    if prep_en is None:
        prompt  = (f'a {subj} and a {anchor} with clear spatial relationship, '
                   f'photorealistic, natural lighting, uncluttered scene')
        ans_out = answer
    else:
        prompt  = (f'a {subj} {prep_en} a {anchor}, clear spatial composition, '
                   f'photorealistic DSLR photo, natural lighting, no text')
        ans_out = matched
    return {'prompt': prompt,
            'question': f'Con {subj} đang ở đâu so với {anchor}?',
            'answer': ans_out}


def _build_object_sample(answer, question):
    scenes = [
        'in a natural outdoor setting, photorealistic',
        'on a white background, studio photo',
        'in a Vietnamese street market scene',
        'in a home environment, natural lighting',
        'in an urban setting, daytime, sharp focus',
    ]
    return {'prompt': f'{answer}, {random.choice(scenes)}, DSLR photo, no text',
            'question': question, 'answer': answer}


def _build_sample(type_name, answer, question):
    if type_name == 'COUNT':      return _build_count_sample(answer)
    elif type_name == 'COLOR':    return _build_color_sample(answer)
    elif type_name == 'LOCATION': return _build_location_sample(answer)
    else:                         return _build_object_sample(answer, question)


# ── FLUX batch inference ──────────────────────────────────────────

def _generate_flux_batch(prompts: list, seeds: list) -> list:
    global _flux_pipe
    if _flux_pipe is None:
        raise RuntimeError('FLUX pipe chưa load — gọi load_generation_models() trước')

    results = _flux_pipe(
        prompt=prompts,
        num_inference_steps=CFG['flux_steps'],
        guidance_scale=0.0,
        width=512,
        height=512,
        max_sequence_length=256,
        generator=torch.Generator('cpu').manual_seed(seeds[0]),
        output_type='pil',
    )
    return results.images


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

    df   = pd.read_csv(current_train_csv)
    pool = df[df['type'] == type_int].copy()

    if plan_entry['n_images'] == 0:
        print(f'  [{type_name}] skip (EM cao, giữ nguyên).')
        return []

    if pool.empty:
        print(f'  [{type_name}] pool rỗng, skip.')
        return []

    answer_pool  = pool['answer'].dropna().tolist()
    new_rows     = []
    generated    = 0
    attempts     = 0
    max_attempts = n * 3

    print(f'  [{type_name}] n={n} | pool={len(pool)} | batch={B} | model=FLUX.1-schnell')

    while generated < n and attempts < max_attempts:

        batch_samples = []
        for _ in range(B):
            base_ans = random.choice(answer_pool)
            base_q   = (pool[pool['answer'] == base_ans]['question'].iloc[0]
                        if (pool['answer'] == base_ans).any()
                        else pool['question'].iloc[0])
            try:
                batch_samples.append(_build_sample(type_name, base_ans, base_q))
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
