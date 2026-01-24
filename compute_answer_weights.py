"""
Compute answer frequency weights for balanced loss

Usage:
    python compute_answer_weights.py --train_csv data/train.csv --output weights.json

Output:
    {
        "answer_to_weight": {"1": 2.5, "2": 1.8, "xanh": 3.2, ...},
        "vocab_size": 50000,
        "token_weights": [1.0, 1.0, 2.5, ...]  # for nn.CrossEntropyLoss(weight=...)
    }
"""

import pandas as pd
import json
import argparse
from collections import Counter
import numpy as np
from transformers import BartphoTokenizer


def compute_answer_weights(csv_path, tokenizer_name='vinai/bartpho-syllable', 
                          min_freq=5, smoothing=0.1):
    """
    Compute inverse frequency weights for answers
    
    Args:
        csv_path: Path to train CSV
        tokenizer_name: BARTpho tokenizer
        min_freq: Minimum frequency to consider (rare answers get capped weight)
        smoothing: Laplace smoothing to avoid inf weights
    
    Returns:
        dict with answer_to_weight mapping and token_weights tensor
    """
    print(f"[Compute Weights] Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Extract answers
    answers = df['answer'].tolist()
    print(f"[Compute Weights] Total samples: {len(answers)}")
    
    # Count answer frequencies
    answer_counts = Counter(answers)
    total = len(answers)
    
    print(f"\n[Answer Distribution] Top 20 most common:")
    for ans, count in answer_counts.most_common(20):
        pct = count / total * 100
        print(f"  '{ans}': {count} ({pct:.2f}%)")
    
    # Compute inverse frequency weights with smoothing
    answer_to_weight = {}
    
    for answer, count in answer_counts.items():
        # Inverse frequency with smoothing
        freq = count / total
        weight = 1.0 / (freq + smoothing)
        
        # Cap weight for very rare answers
        if count < min_freq:
            weight = min(weight, 10.0)  # Max 10x weight
        
        answer_to_weight[answer] = float(weight)
    
    # Normalize weights to have mean=1.0
    weights = list(answer_to_weight.values())
    mean_weight = np.mean(weights)
    answer_to_weight = {k: v/mean_weight for k, v in answer_to_weight.items()}
    
    print(f"\n[Weight Stats]")
    print(f"  Unique answers: {len(answer_to_weight)}")
    print(f"  Weight range: [{min(weights):.2f}, {max(weights):.2f}]")
    print(f"  Mean weight: {np.mean(list(answer_to_weight.values())):.2f}")
    
    # Build token-level weights for CrossEntropyLoss
    print(f"\n[Tokenizer] Loading {tokenizer_name}...")
    tokenizer = BartphoTokenizer.from_pretrained(tokenizer_name)
    vocab_size = len(tokenizer)
    
    # Initialize all token weights to 1.0
    token_weights = np.ones(vocab_size, dtype=np.float32)
    
    # Map answer weights to token weights
    # For multi-token answers, apply weight to all tokens
    for answer, weight in answer_to_weight.items():
        token_ids = tokenizer.encode(answer, add_special_tokens=False)
        for token_id in token_ids:
            # Use max weight if token appears in multiple answers
            token_weights[token_id] = max(token_weights[token_id], weight)
    
    print(f"[Token Weights] {vocab_size} tokens, {(token_weights > 1.0).sum()} weighted")
    
    return {
        'answer_to_weight': answer_to_weight,
        'vocab_size': vocab_size,
        'token_weights': token_weights.tolist()
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv', required=True, help='Path to train CSV')
    parser.add_argument('--output', default='answer_weights.json', help='Output JSON path')
    parser.add_argument('--tokenizer', default='vinai/bartpho-syllable')
    parser.add_argument('--min_freq', type=int, default=5, help='Min frequency for weight capping')
    parser.add_argument('--smoothing', type=float, default=0.1, help='Laplace smoothing')
    args = parser.parse_args()
    
    weights_dict = compute_answer_weights(
        csv_path=args.train_csv,
        tokenizer_name=args.tokenizer,
        min_freq=args.min_freq,
        smoothing=args.smoothing
    )
    
    # Save to JSON
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(weights_dict, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Saved weights to: {args.output}")
    print(f"\nUsage in training:")
    print(f"  --answer_weights {args.output}")


if __name__ == '__main__':
    main()
