"""
Preprocessing script for ViVQA-X dataset.

Converts ViVQA-X JSON files to CSV format compatible with train.py/eval.py.
Creates symlinks from {img_id}.jpg → COCO original filenames.
Recomputes answer_weights.json from training split.

Output CSV columns: question, answer, img_id, answer_type, type
  - answer_type: "yes/no" | "number" | "other"  (from JSON)
  - type: integer 0–3 from ViVQA-X paper taxonomy
      0 = YesNo      (answer_type=="yes/no", takes priority)
      1 = Location   (question_type in {"what room is","where is the","where are the"})
      2 = Attribute  (question_type in {"what","what color","what kind of","what type of"})
      3 = Object     (all other non-yes/no questions — default)
    answer_type is the authoritative signal for YesNo; JSON question_type
    (English VQA 2.0 string) distinguishes Location/Attribute/Object.

Answer normalization applied:
  - NFC unicode, strip, lowercase
  - Trailing punctuation stripped ("không." → "không")
  - Yes/no synonyms unified: "đúng"/"vâng"/"phải" → "có" (all 100% yes/no type)

Usage:
    python preprocess_vivqax.py \\
        --coco_train_dir /path/to/coco/train2014 \\
        --coco_val_dir   /path/to/coco/val2014 \\
        --output_dir     vivqax_data

Output structure:
    vivqax_data/
        train.csv           (columns: question, answer, img_id, answer_type, type)
        val.csv
        test.csv
        images/train/       (symlinks → coco_train_dir)
        images/val/         (symlinks → coco_val_dir)
        images/test/        (symlinks → coco_val_dir)
        answer_weights.json
"""

import argparse
import json
import os
import re
import unicodedata
from pathlib import Path

import pandas as pd

TRAIN_JSON_URL = "https://huggingface.co/datasets/VLAI-AIVN/ViVQA-X/resolve/main/ViVQA-X_train.json"
VAL_JSON_URL   = "https://huggingface.co/datasets/VLAI-AIVN/ViVQA-X/resolve/main/ViVQA-X_val.json"
TEST_JSON_URL  = "https://huggingface.co/datasets/VLAI-AIVN/ViVQA-X/resolve/main/ViVQA-X_test.json"

# Vietnamese synonyms for "yes" — all confirmed 100% in yes/no answer_type
_YES_SYNONYMS = {"đúng", "đúng vậy", "đúng rồi", "đúng ạ", "vâng", "phải", "phải rồi", "ừ"}
# Vietnamese synonyms for "no" (rare, only normalize clear cases)
_NO_SYNONYMS  = {"chưa"}

# ViVQA-X question type taxonomy (paper: YesNo / Location / Attribute / Object)
# Mapping rule (verified against all 32 928 QA pairs, 100% match):
#   1. answer_type == "yes/no"  → 0 (YesNo)   [takes priority over question_type]
#   2. else, question_type in _LOCATION_QT     → 1 (Location)
#   3. else, question_type in _ATTRIBUTE_QT    → 2 (Attribute)
#   4. else                                    → 3 (Object, default)
#
# Key insight: some English question_type strings (e.g. "is the", "does the") are
# labeled YesNo in VQA 2.0 but their Vietnamese translations may be open-ended
# (answer_type="other"). answer_type from the JSON is the authoritative signal.

_LOCATION_QT  = {"what room is", "where is the", "where are the"}
_ATTRIBUTE_QT = {"what", "what color", "what color is", "what kind of", "what type of"}


def _get_vivqax_type(answer_type: str, question_type: str) -> int:
    """Map ViVQA-X JSON fields to integer type (0=YesNo,1=Location,2=Attribute,3=Object)."""
    if answer_type == 'yes/no':
        return 0
    if question_type in _LOCATION_QT:
        return 1
    if question_type in _ATTRIBUTE_QT:
        return 2
    return 3


def download_json(url, path):
    if os.path.exists(path):
        print(f"  [cached] {path}")
        return
    import urllib.request
    print(f"  Downloading {url} ...")
    urllib.request.urlretrieve(url, path)
    print(f"  Saved → {path}")


def normalize_answer(ans: str, answer_type: str = "") -> str:
    ans = unicodedata.normalize('NFC', str(ans).strip().lower())
    ans = re.sub(r'[.!?,;]+$', '', ans).strip()  # strip trailing punctuation
    if answer_type == 'yes/no':
        if ans in _YES_SYNONYMS:
            ans = 'có'
        elif ans in _NO_SYNONYMS:
            ans = 'không'
    return ans


def build_csv(records, coco_img_dir, out_img_dir, out_csv_path, split_name):
    """Convert ViVQA-X JSON records to CSV and create image symlinks."""
    os.makedirs(out_img_dir, exist_ok=True)

    rows = []
    missing_imgs = 0
    for r in records:
        img_id      = str(r['image_id'])
        img_name    = r['image_name']   # e.g. COCO_val2014_000000262284.jpg
        question    = str(r['question']).strip()
        answer_type = r.get('answer_type', 'other')
        answer      = normalize_answer(r['answer'], answer_type)

        # Symlink: {out_img_dir}/{img_id}.jpg → {coco_img_dir}/{img_name}
        src  = os.path.join(coco_img_dir, img_name)
        dest = os.path.join(out_img_dir, f"{img_id}.jpg")
        if not os.path.exists(dest):
            if os.path.exists(src):
                os.symlink(os.path.abspath(src), dest)
            else:
                missing_imgs += 1

        q_type = _get_vivqax_type(answer_type, r.get('question_type', ''))

        rows.append({
            'question':    question,
            'answer':      answer,
            'img_id':      img_id,
            'answer_type': answer_type,  # "yes/no" | "number" | "other"
            'type':        q_type,       # 0=YesNo, 1=Location, 2=Attribute, 3=Object
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_csv_path, index=False)  # index=False: no unnamed index column
    print(f"  [{split_name}] {len(df)} rows → {out_csv_path}")

    if missing_imgs:
        print(f"  WARNING: {missing_imgs} images not found in {coco_img_dir}")

    atype_dist = dict(df['answer_type'].value_counts())
    print(f"  answer_type: {atype_dist}")
    print(f"  unique answers: {df['answer'].nunique()}")

    return df


def compute_answer_weights(train_df, output_path):
    """Generate answer_weights.json in the format expected by train.py:
    {"answer_to_weight": {...}, "vocab_size": N, "token_weights": [...]}
    Uses compute_answer_weights.py logic (inverse-freq, token-level mapping).
    """
    import subprocess, sys
    # Write train CSV to a temp path if needed, then call compute_answer_weights.py
    tmp_csv = output_path + '.tmp_train.csv'
    train_df.to_csv(tmp_csv, index=False)
    result = subprocess.run(
        [sys.executable, 'compute_answer_weights.py',
         '--train_csv', tmp_csv,
         '--output', output_path,
         '--min_freq', '3', '--smoothing', '0.1'],
        capture_output=True, text=True
    )
    os.remove(tmp_csv)
    if result.returncode != 0:
        print(f"  WARNING: compute_answer_weights.py failed: {result.stderr}")
    else:
        # Print summary line from output
        for line in result.stdout.splitlines():
            if 'Token Weights' in line or 'Saved' in line:
                print(f"  {line.strip()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--coco_train_dir', required=True,
                        help='Path to COCO train2014 image folder')
    parser.add_argument('--coco_val_dir', required=True,
                        help='Path to COCO val2014 image folder')
    parser.add_argument('--output_dir', default='vivqax_data',
                        help='Output directory for CSVs, images, weights')
    parser.add_argument('--json_cache_dir', default='/tmp',
                        help='Directory to cache downloaded JSON files')
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cache = args.json_cache_dir
    train_json = os.path.join(cache, 'vivqax_train.json')
    val_json   = os.path.join(cache, 'vivqax_val.json')
    test_json  = os.path.join(cache, 'vivqax_test.json')

    print("=== Downloading ViVQA-X JSONs ===")
    download_json(TRAIN_JSON_URL, train_json)
    download_json(VAL_JSON_URL,   val_json)
    download_json(TEST_JSON_URL,  test_json)

    with open(train_json, encoding='utf-8') as f: train_records = json.load(f)
    with open(val_json,   encoding='utf-8') as f: val_records   = json.load(f)
    with open(test_json,  encoding='utf-8') as f: test_records  = json.load(f)

    print(f"\n=== Building CSVs & symlinks ===")
    train_df = build_csv(
        train_records,
        coco_img_dir=args.coco_train_dir,
        out_img_dir=str(out / 'images' / 'train'),
        out_csv_path=str(out / 'train.csv'),
        split_name='train',
    )
    val_df = build_csv(
        val_records,
        coco_img_dir=args.coco_val_dir,
        out_img_dir=str(out / 'images' / 'val'),
        out_csv_path=str(out / 'val.csv'),
        split_name='val',
    )
    build_csv(
        test_records,
        coco_img_dir=args.coco_val_dir,
        out_img_dir=str(out / 'images' / 'test'),
        out_csv_path=str(out / 'test.csv'),
        split_name='test',
    )

    # Create merged trainval image dir so train.py can use one --image_dir for both splits.
    # train.py passes --image_dir to BOTH train and val VQAGenDataset, so val images
    # must also be accessible from that path. Train/val image_ids don't overlap in ViVQA-X.
    trainval_dir = out / 'images' / 'trainval'
    trainval_dir.mkdir(parents=True, exist_ok=True)
    n_linked = 0
    for src_dir in [out / 'images' / 'train', out / 'images' / 'val']:
        for link in src_dir.iterdir():
            dest = trainval_dir / link.name
            if not dest.exists():
                dest.symlink_to(link.resolve())
                n_linked += 1
    print(f"\n  images/trainval/: {n_linked} new symlinks (train ∪ val, no overlap)")

    print(f"\n=== Computing answer weights (from train only) ===")
    compute_answer_weights(train_df, str(out / 'answer_weights.json'))

    print(f"\n=== Summary ===")
    print(f"  Train: {len(train_df)} samples, {train_df['answer'].nunique()} unique answers")
    print(f"  Val:   {len(val_df)} samples")
    print(f"  Output dir: {out.resolve()}")
    n = out.resolve()
    print(f"""
NOTE on training flags vs ViVQA original:
  REMOVE --pk_sampling   : 56% YesNo + 22% Object → PK sampling over 4 types is less effective
  KEEP   --use_type_loss : type column now written correctly from JSON → type loss is valid
  KEEP   --use_cdw_ce    : still useful for imbalanced answer distribution
  KEEP   --use_constrained --train_csv_for_trie {n}/train.csv

=== Training command (run87 hyperparams, ViVQA-X adjusted) ===
""")
    print(f"""nohup /home/user/workspace/all_env/vivqa/bin/python3 train.py \\
  --train_csv {n}/train.csv \\
  --val_csv   {n}/val.csv \\
  --image_dir {n}/images/train \\
  --epochs 40 --lr 7e-5 --weight_decay 0.01 --label_smoothing 0.1 \\
  --scheduler cosine --warmup_epochs 3 \\
  --early_stopping --early_stopping_patience 15 --early_stopping_metric em \\
  --fusion_type text2vision --num_fusion_layers 2 \\
  --use_text_lora --text_lora_r 16 --text_lora_alpha 32 \\
  --use_vision_gate --vision_gate_init 1.0 --vision_gate_min_alpha 0.0 \\
  --use_siglip_pooler \\
  --use_cdw_ce --cdw_lambda 0.1 \\
  --answer_weights {n}/answer_weights.json \\
  --batch_size 12 --vision_dropout_rate 0.10 --seed 42 --num_workers 4 \\
  --output_dir checkpoints_vivqax_run1 \\
  > checkpoints_vivqax_run1/train_log.txt 2>&1 &
""")
    print(f"""=== Eval command ===
/home/user/workspace/all_env/vivqa/bin/python3 eval.py \\
  --checkpoint checkpoints_vivqax_run1/best_model.pt \\
  --csv_path   {n}/test.csv \\
  --image_folder {n}/images/test \\
  --output_csv checkpoints_vivqax_run1/eval_test.csv \\
  --num_beams 3 --repetition_penalty 1.3 --max_length 10 \\
  --use_synonyms --use_constrained \\
  --train_csv_for_trie {n}/train.csv \\
  > checkpoints_vivqax_run1/eval_test.log 2>&1
""")


if __name__ == '__main__':
    main()
