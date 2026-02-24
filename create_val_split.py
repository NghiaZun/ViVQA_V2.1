"""
create_val_split.py — Tách val set cố định từ train CSV cho ViVQA

Dataset: ~12,000 mẫu (ViVQA)
Cột CSV : index, question, answer, img_id, type (0=OBJECT,1=COUNT,2=COLOR,3=LOCATION)

Chiến lược STRATIFIED SPLIT (tốt nhất cho mọi case):
  - Phân tầng theo (type × answer_frequency_bucket)
  - Dùng cột 'type' có sẵn trong CSV, KHÔNG auto-detect
  - Đảm bảo phân phối 4 loại câu hỏi giống nhau ở train lẫn val
  - Đảm bảo cả rare answers lẫn common answers xuất hiện trong val
  - Seed cố định → mọi lần chạy đều ra cùng một split

Tại sao val_ratio=0.10 (10%) phù hợp nhất với ~12K mẫu?
  - 5%  →  ~600 val: metric không ổn định, variance ±3-4% giữa các epoch
  - 10% → ~1200 val: đủ ổn định (variance ±1.5%), không lãng phí train data  ✅
  - 15% → ~1800 val: ổn hơn nhưng lãng phí ~600 mẫu train không cần thiết

Tại sao stratified tốt hơn random_split?
  - random_split: mỗi lần chạy (hoặc mỗi lần khởi động Python) ra split khác
    (dù dùng seed, thứ tự shuffle trong Kaggle notebook đôi khi khác)
  - Stratified CSV: split được lưu ra file, KHÔNG BAO GIỜ thay đổi nữa
  - Phân tầng: val phản ánh đúng phân phối thực → metric đáng tin cậy hơn

Usage:
    # Tách 10% val (khuyến nghị), stratified theo cột type trong CSV
    python create_val_split.py --csv data/train.csv --output_dir data/

    # Xem thống kê nhưng chưa lưu file
    python create_val_split.py --csv data/train.csv --dry_run

    # Tùy chỉnh tỉ lệ (không khuyến nghị thay đổi với ~12K mẫu)
    python create_val_split.py --csv data/train.csv --output_dir data/ --val_ratio 0.15

Output:
    data/train_split.csv   ← dùng thay train.csv khi train
    data/val_split.csv     ← dùng cho --val_csv
    data/split_stats.json  ← thống kê split để kiểm tra
"""

import argparse
import json
import os
import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd


# ============================================================================
# QUESTION TYPE MAPPING
# ============================================================================

# Cột 'type' trong CSV: 0=OBJECT, 1=COUNT, 2=COLOR, 3=LOCATION
TYPE_ID_TO_NAME = {0: 'OBJECT', 1: 'COUNT', 2: 'COLOR', 3: 'LOCATION'}


def detect_question_type(question: str) -> str:
    """
    Fallback: auto-detect từ text nếu CSV không có cột 'type'.
    Ưu tiên dùng cột 'type' trong CSV hơn (xem assign_stratum).
    """
    q = question.lower().strip()
    if re.search(r'màu\s*(gì|sắc|nào)?', q):
        return 'COLOR'
    if re.search(r'(bao nhiêu|mấy|số lượng|có\s*\d+)', q):
        return 'COUNT'
    if re.search(r'(ở\s*(đâu|nào)|phía|vị\s*trí|bên|trên|dưới|trái|phải|trong|ngoài|giữa)', q):
        return 'LOCATION'
    return 'OBJECT'


# ============================================================================
# ANSWER FREQUENCY BUCKET
# ============================================================================

def get_answer_bucket(answer: str, freq: int) -> str:
    """
    Phân loại answer theo tần suất xuất hiện.
    Giúp val set luôn có đủ rare answers để đo generalization.

    Buckets:
        rare    : freq <= 3    (long-tail, khó nhất)
        uncommon: freq 4–15
        common  : freq 16–50
        frequent: freq > 50
    """
    if freq <= 3:
        return 'rare'
    elif freq <= 15:
        return 'uncommon'
    elif freq <= 50:
        return 'common'
    else:
        return 'frequent'


# ============================================================================
# STRATIFIED SPLIT
# ============================================================================

def stratified_split(
    df: pd.DataFrame,
    val_ratio: float = 0.10,
    seed: int = 42,
    min_val_per_stratum: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Tách train/val bằng stratified sampling theo:
        stratum = question_type + '_' + answer_bucket

    Đảm bảo:
        1. Mỗi stratum đều có ít nhất min_val_per_stratum mẫu trong val
        2. Tổng val ≈ val_ratio × toàn bộ data
        3. Reproducible hoàn toàn qua seed

    Args:
        df           : DataFrame với cột 'question', 'answer', 'img_id'
        val_ratio    : Tỉ lệ val (0.0–1.0)
        seed         : Random seed cố định
        min_val_per_stratum: Số mẫu tối thiểu trong val cho mỗi stratum

    Returns:
        train_df, val_df
    """
    rng = np.random.default_rng(seed)

    # ── Bước 1: Tính tần suất answer ──────────────────────────────────────
    answer_freq = Counter(df['answer'].astype(str).str.strip().str.lower())

    # ── Bước 2: Gán stratum cho từng dòng ─────────────────────────────────
    has_type_col = 'type' in df.columns

    def assign_stratum(row) -> str:
        # Ưu tiên cột 'type' có sẵn trong CSV (0=OBJECT,1=COUNT,2=COLOR,3=LOCATION)
        if has_type_col:
            q_type = TYPE_ID_TO_NAME.get(int(row['type']), 'OBJECT')
        else:
            q_type = detect_question_type(str(row['question']))
        ans = str(row['answer']).strip().lower()
        freq = answer_freq[ans]
        bucket = get_answer_bucket(ans, freq)
        return f"{q_type}_{bucket}"

    df = df.copy()
    df['_stratum'] = df.apply(assign_stratum, axis=1)

    # ── Bước 3: Stratified sampling ───────────────────────────────────────
    val_indices = []
    train_indices = []

    strata = df['_stratum'].unique()
    for stratum in sorted(strata):
        stratum_idx = df.index[df['_stratum'] == stratum].tolist()
        n = len(stratum_idx)

        # Số mẫu val cho stratum này
        n_val = max(min_val_per_stratum, round(n * val_ratio))
        n_val = min(n_val, n - 1)  # Luôn giữ ít nhất 1 mẫu cho train

        shuffled = rng.permutation(stratum_idx).tolist()
        val_indices.extend(shuffled[:n_val])
        train_indices.extend(shuffled[n_val:])

    train_df = df.loc[train_indices].drop(columns=['_stratum']).reset_index(drop=True)
    val_df   = df.loc[val_indices].drop(columns=['_stratum']).reset_index(drop=True)

    return train_df, val_df


# ============================================================================
# STATISTICS & REPORTING
# ============================================================================

def compute_split_stats(
    original_df: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> dict:
    """Tổng hợp thống kê để kiểm tra chất lượng split."""

    def type_dist(df: pd.DataFrame) -> dict:
        counts = {}
        if 'type' in df.columns:
            # Dùng cột type trực tiếp — nhanh và chính xác hơn
            for val in df['type']:
                t = TYPE_ID_TO_NAME.get(int(val), 'OBJECT')
                counts[t] = counts.get(t, 0) + 1
        else:
            for _, row in df.iterrows():
                t = detect_question_type(str(row['question']))
                counts[t] = counts.get(t, 0) + 1
        total = len(df)
        return {t: {'count': c, 'pct': round(c / total * 100, 2)}
                for t, c in sorted(counts.items())}

    def answer_stats(df: pd.DataFrame) -> dict:
        answers = df['answer'].astype(str).str.strip().str.lower()
        freq = Counter(answers)
        unique = len(freq)
        top5 = [{'answer': a, 'count': c} for a, c in freq.most_common(5)]
        rare = sum(1 for c in freq.values() if c == 1)
        return {
            'unique_answers': unique,
            'top5': top5,
            'singleton_answers': rare,
            'singleton_pct': round(rare / unique * 100, 2) if unique else 0,
        }

    return {
        'total': len(original_df),
        'train_size': len(train_df),
        'val_size': len(val_df),
        'val_ratio_actual': round(len(val_df) / len(original_df) * 100, 2),
        'question_type_distribution': {
            'original': type_dist(original_df),
            'train': type_dist(train_df),
            'val': type_dist(val_df),
        },
        'answer_stats': {
            'original': answer_stats(original_df),
            'train': answer_stats(train_df),
            'val': answer_stats(val_df),
        },
    }


def print_stats(stats: dict) -> None:
    """In thống kê ra terminal."""
    print("\n" + "="*60)
    print("SPLIT STATISTICS")
    print("="*60)
    print(f"  Total samples  : {stats['total']:,}")
    print(f"  Train          : {stats['train_size']:,}")
    print(f"  Val            : {stats['val_size']:,}  ({stats['val_ratio_actual']}%)")

    print("\n  [Question Type Distribution]")
    header = f"  {'Type':<12}  {'Original':>10}  {'Train':>10}  {'Val':>10}"
    print(header)
    print("  " + "-"*52)
    types = sorted(set(
        list(stats['question_type_distribution']['original'].keys()) +
        list(stats['question_type_distribution']['val'].keys())
    ))
    for t in types:
        orig = stats['question_type_distribution']['original'].get(t, {})
        trn  = stats['question_type_distribution']['train'].get(t, {})
        val  = stats['question_type_distribution']['val'].get(t, {})
        print(f"  {t:<12}  "
              f"{orig.get('pct', 0):>8.1f}%  "
              f"{trn.get('pct', 0):>8.1f}%  "
              f"{val.get('pct', 0):>8.1f}%")

    print("\n  [Answer Stats]")
    for split_name in ('original', 'train', 'val'):
        s = stats['answer_stats'][split_name]
        print(f"  {split_name.capitalize():<10}: {s['unique_answers']:,} unique answers "
              f"| {s['singleton_answers']:,} singletons ({s['singleton_pct']}%)")

    print("\n  [Top-5 Val Answers]")
    for item in stats['answer_stats']['val']['top5']:
        print(f"    '{item['answer']}': {item['count']}")

    print("="*60)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Tạo val split cố định từ train CSV (stratified)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--csv', required=True,
                        help='Đường dẫn tới file train CSV gốc')
    parser.add_argument('--output_dir', default=None,
                        help='Thư mục lưu output. Mặc định = cùng thư mục với --csv')
    parser.add_argument('--val_ratio', type=float, default=0.10,
                        help='Tỉ lệ val (mặc định 0.10 = 10%% — khuyến nghị cho ~12K mẫu)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed cố định (mặc định 42)')
    parser.add_argument('--min_val_per_stratum', type=int, default=1,
                        help='Số mẫu tối thiểu trong val mỗi stratum (mặc định 1)')
    parser.add_argument('--train_out', type=str, default='train_split.csv',
                        help='Tên file train output (mặc định train_split.csv)')
    parser.add_argument('--val_out', type=str, default='val_split.csv',
                        help='Tên file val output (mặc định val_split.csv)')
    parser.add_argument('--dry_run', action='store_true',
                        help='Chỉ in thống kê, không lưu file')
    parser.add_argument('--no_shuffle_train', action='store_true',
                        help='Không shuffle lại train_split.csv sau khi tách')
    args = parser.parse_args()

    # ── Validate ──────────────────────────────────────────────────────────
    if not os.path.isfile(args.csv):
        raise FileNotFoundError(f"Không tìm thấy file CSV: {args.csv}")
    if not (0.0 < args.val_ratio < 1.0):
        raise ValueError(f"val_ratio phải trong khoảng (0, 1), nhận được: {args.val_ratio}")

    output_dir = args.output_dir or os.path.dirname(os.path.abspath(args.csv))
    os.makedirs(output_dir, exist_ok=True)

    # ── Load ──────────────────────────────────────────────────────────────
    print(f"[Load] Đọc file: {args.csv}")
    df = pd.read_csv(args.csv)

    # Kiểm tra cột bắt buộc
    required_cols = {'question', 'answer', 'img_id'}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV thiếu cột: {missing}. Các cột hiện có: {list(df.columns)}")

    has_type = 'type' in df.columns
    print(f"[Load] Tổng {len(df):,} mẫu | Cột: {list(df.columns)}")
    print(f"[Load] Cột 'type': {'✅ có sẵn (dùng trực tiếp)' if has_type else '❌ không có (sẽ auto-detect từ text)'}")

    # ── Split ─────────────────────────────────────────────────────────────
    print(f"\n[Split] Stratified split: val_ratio={args.val_ratio}, seed={args.seed}")
    train_df, val_df = stratified_split(
        df,
        val_ratio=args.val_ratio,
        seed=args.seed,
        min_val_per_stratum=args.min_val_per_stratum,
    )

    # ── Shuffle train (optional but recommended) ──────────────────────────
    if not args.no_shuffle_train:
        train_df = train_df.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    # ── Stats ─────────────────────────────────────────────────────────────
    stats = compute_split_stats(df, train_df, val_df)
    print_stats(stats)

    if args.dry_run:
        print("\n[Dry Run] Không lưu file. Bỏ --dry_run để lưu.")
        return

    # ── Save ──────────────────────────────────────────────────────────────
    train_path = os.path.join(output_dir, args.train_out)
    val_path   = os.path.join(output_dir, args.val_out)
    stats_path = os.path.join(output_dir, 'split_stats.json')

    train_df.to_csv(train_path, index=False, encoding='utf-8')
    val_df.to_csv(val_path,   index=False, encoding='utf-8')

    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"\n[Saved]")
    print(f"  Train → {train_path}  ({len(train_df):,} mẫu)")
    print(f"  Val   → {val_path}   ({len(val_df):,} mẫu)")
    print(f"  Stats → {stats_path}")

    print(f"""
[Sử dụng trong train.py]
  python train.py \\
    --train_csv {train_path} \\
    --val_csv   {val_path} \\
    --image_dir <image_folder> \\
    ... (các args khác)

⚠️  LƯU Ý: Vì đã có val_csv cố định, train.py sẽ KHÔNG random_split nữa.
   Mọi lần chạy train sẽ dùng đúng cùng một val set.
""")


if __name__ == '__main__':
    main()
