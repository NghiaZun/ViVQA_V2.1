"""
Step 1: Clean train_combined.csv (remove duplicates)
Step 2: Compute counting answer weights for loss balancing
"""

import pandas as pd
import numpy as np
import json
from collections import Counter

print("="*80)
print("🧹 STEP 1: CLEANING TRAIN_COMBINED.CSV")
print("="*80)

# Load and deduplicate
df = pd.read_csv('train_combined.csv')
original_count = len(df)

df_clean = df.drop_duplicates(subset=['question', 'img_id'], keep='first')
clean_count = len(df_clean)

print(f"\n  Original:  {original_count:,} samples")
print(f"  Removed:   {original_count - clean_count:,} duplicates")
print(f"  Final:     {clean_count:,} samples")

# Save cleaned version
df_clean.to_csv('train_combined_clean.csv', index=False)
print(f"\n✅ Saved to: train_combined_clean.csv")

print("\n" + "="*80)
print("🔢 STEP 2: COMPUTING COUNTING ANSWER WEIGHTS")
print("="*80)

# Filter counting type
counting_df = df_clean[df_clean['type'] == 1]
counting_total = len(counting_df)
counting_answers = counting_df['answer'].value_counts()

print(f"\n  Total counting questions: {counting_total:,}")
print(f"  Unique answers: {len(counting_answers)}")

# Compute weights with sqrt-inverse + aggressive clipping
weights = {}
for ans, freq in counting_answers.items():
    # Sqrt inverse (less aggressive than plain inverse)
    w = 1.0 / np.sqrt(freq + 1.0)
    
    # Aggressive clipping for counting
    w = np.clip(w, 0.4, 3.0)
    
    weights[ans] = w

# Normalize to mean=1.0
mean_w = np.mean(list(weights.values()))
weights = {k: v/mean_w for k, v in weights.items()}

# Convert to float for JSON
weights = {k: float(v) for k, v in weights.items()}

# Save weights
output_file = 'counting_answer_weights.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(weights, f, ensure_ascii=False, indent=2)

print(f"\n✅ Saved to: {output_file}")

print(f"\n📊 WEIGHT DISTRIBUTION:")
print("-"*50)

# Show top answers with weights
for ans in ['hai', 'ba', 'bốn', 'năm', 'một', 'sáu', 'bảy', 'tám', 'chín', 'mười']:
    if ans in weights:
        freq = counting_answers.get(ans, 0)
        pct = freq / counting_total * 100
        w = weights[ans]
        
        # Visual indicator
        if w < 0.7:
            indicator = "⬇️ PENALIZE"
        elif w > 1.5:
            indicator = "⬆️ BOOST"
        else:
            indicator = "➡️ neutral"
        
        print(f"  {ans:6s}: freq={freq:4d} ({pct:5.2f}%) → weight={w:.3f} {indicator}")

# Statistics
weight_values = list(weights.values())
print(f"\n📈 STATISTICS:")
print(f"  Min weight:  {min(weight_values):.3f}")
print(f"  Max weight:  {max(weight_values):.3f}")
print(f"  Mean weight: {np.mean(weight_values):.3f}")
print(f"  Ratio max/min: {max(weight_values)/min(weight_values):.2f}x")

print("\n" + "="*80)
print("🎯 NEXT STEPS:")
print("="*80)
print("\n1. Use train_combined_clean.csv for training")
print("2. Add --answer_weights counting_answer_weights.json")
print("3. Add --use_question_prefix for better type conditioning")
print("\nExample command:")
print("""
python train_no_latent.py \\
  --train_csv train_combined_clean.csv \\
  --answer_weights counting_answer_weights.json \\
  --use_question_prefix \\
  ... (other args)
""")
print("="*80)
