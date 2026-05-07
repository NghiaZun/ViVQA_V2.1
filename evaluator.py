import re
import subprocess

from config import CFG, TYPES, TYPE_NAME_MAP


def run_eval(ckpt_path, result_csv_path):
    """
    Chạy eval.py và parse kết quả.
    Trả về:
        overall  : dict {'F1': float, 'EM': float}  (0-100 scale)
        per_type : dict {type_name: {'F1': float, 'EM': float}}
    """
    cmd = (
        f"python eval.py "
        f"--checkpoint {ckpt_path} "
        f"--csv_path {CFG['val_csv']} "
        f"--image_folder {CFG['image_dir']} "
        f"--output_csv {result_csv_path} "
        f"--num_beams 3 "
        f"--repetition_penalty 1.3"
    )
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    output = result.stdout + result.stderr
    print(output)

    overall = {'F1': 0.0, 'EM': 0.0}
    m = re.search(r'F1 Score:\s+([\d.]+)%', output)
    if m:
        overall['F1'] = float(m.group(1))
    m = re.search(r'Exact Match:\s+([\d.]+)%', output)
    if m:
        overall['EM'] = float(m.group(1))

    per_type = {}
    for raw_name, key in TYPE_NAME_MAP.items():
        m = re.search(rf'\b{raw_name}\s+([\d.]+)\s+([\d.]+)', output)
        if m:
            per_type[key] = {
                'EM': float(m.group(1)),
                'F1': float(m.group(2)),
            }

    if not per_type:
        print('⚠️  per_type rỗng — eval.py có thể đã crash hoặc output format thay đổi.')
        print('── 2000 ký tự cuối output ──')
        print(output[-2000:])
        print('────────────────────────────')

    return overall, per_type


def should_stop(overall, per_type, prev_overall_em):
    """
    Dừng loop khi CẢ BA điều kiện sau đều đúng:
    1. EM variance < stop_em_variance  (các type hội tụ)
    2. EM overall tăng >= stop_em_delta so với loop trước
    3. F1 - EM < stop_gap_max cho mọi type
    """
    em_values  = [per_type[t]['EM'] for t in TYPES if t in per_type]
    gap_values = [per_type[t]['F1'] - per_type[t]['EM'] for t in TYPES if t in per_type]

    if not em_values:
        print('  ⚠️  per_type rỗng — bỏ qua stop check, tiếp tục loop.')
        return False

    em_variance  = max(em_values) - min(em_values)
    em_improving = (overall['EM'] - prev_overall_em) >= CFG['stop_em_delta']
    gaps_ok      = all(g < CFG['stop_gap_max'] for g in gap_values)

    print(f'\n  Stop check:')
    print(f'    EM variance  = {em_variance:.2f}  (cần < {CFG["stop_em_variance"]}) → {"✅" if em_variance < CFG["stop_em_variance"] else "❌"}')
    print(f'    EM delta     = {overall["EM"] - prev_overall_em:+.2f}  (cần >= {CFG["stop_em_delta"]}) → {"✅" if em_improving else "❌"}')
    print(f'    Max EM-F1 gap= {max(gap_values):.2f}  (cần < {CFG["stop_gap_max"]}) → {"✅" if gaps_ok else "❌"}')

    return em_variance < CFG['stop_em_variance'] and em_improving and gaps_ok
