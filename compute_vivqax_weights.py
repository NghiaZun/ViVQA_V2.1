"""
Compute answer weights for ViVQA-X using sqrt inverse frequency.

Unlike compute_answer_weights.py (ViVQA), this avoids the max(v/mean, 1.0)
floor clip that makes all common answers weight=1.0 on skewed distributions.

Formula: weight = sqrt(total / count), capped at max_weight.

Usage:
    python compute_vivqax_weights.py \
        --train_csv vivqax_data/train.csv \
        --output vivqax_data/answer_weights.json
"""

import pandas as pd
import json
import argparse
import math
from collections import Counter

import numpy as np
from transformers import BartphoTokenizer


def compute_vivqax_weights(csv_path, tokenizer_name="vinai/bartpho-syllable", max_weight=8.0):
    print(f"[Weights] Loading {csv_path} ...")
    df = pd.read_csv(csv_path)
    answers = df["answer"].tolist()
    total = len(answers)
    counts = Counter(answers)
    print(f"[Weights] {total} samples, {len(counts)} unique answers")

    print("\nTop-20 answers:")
    for ans, cnt in counts.most_common(20):
        print(f"  {ans!r:30s}: {cnt:5d}  ({cnt/total*100:.1f}%)")

    # sqrt inverse frequency — no floor clip
    answer_to_weight = {}
    for ans, cnt in counts.items():
        w = math.sqrt(total / cnt)
        answer_to_weight[ans] = min(float(w), max_weight)

    ws = list(answer_to_weight.values())
    print(f"\nWeight range: [{min(ws):.3f}, {max(ws):.3f}]  mean={sum(ws)/len(ws):.3f}")
    for ans in ["có", "không", "quần vợt", "bóng đá", "bơi lội"]:
        if ans in answer_to_weight:
            print(f"  {ans!r}: {answer_to_weight[ans]:.3f}")

    print(f"\n[Tokenizer] Loading {tokenizer_name} ...")
    tokenizer = BartphoTokenizer.from_pretrained(tokenizer_name)
    vocab_size = len(tokenizer)
    token_weights = np.ones(vocab_size, dtype=np.float32)
    for ans, w in answer_to_weight.items():
        for tok_id in tokenizer.encode(ans, add_special_tokens=False):
            token_weights[tok_id] = max(token_weights[tok_id], w)

    n_weighted = int((token_weights > 1.0).sum())
    print(f"[Token Weights] {vocab_size} tokens, {n_weighted} weighted (>{1.0})")

    return {
        "answer_to_weight": answer_to_weight,
        "vocab_size": vocab_size,
        "token_weights": token_weights.tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--output", default="vivqax_data/answer_weights.json")
    parser.add_argument("--tokenizer", default="vinai/bartpho-syllable")
    parser.add_argument("--max_weight", type=float, default=8.0,
                        help="Cap for very rare answers (default 8.0)")
    args = parser.parse_args()

    result = compute_vivqax_weights(args.train_csv, args.tokenizer, args.max_weight)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nSaved → {args.output}")


if __name__ == "__main__":
    main()
