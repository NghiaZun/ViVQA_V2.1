"""
Create hybrid dataset from train.csv + train_augmentation.csv
Strategy: Balance all 4 types to ~25% each
"""

import pandas as pd
import numpy as np

def create_hybrid_dataset():
    print("="*80)
    print("🔧 CREATING HYBRID DATASET")
    print("="*80)
    
    # Load datasets
    print("\n📂 Loading datasets...")
    train_orig = pd.read_csv('train.csv')
    train_aug = pd.read_csv('train_augmentation.csv')
    
    print(f"  train.csv:            {len(train_orig):5d} samples")
    print(f"  train_augmentation:   {len(train_aug):5d} samples")
    
    # Target: ~2000 samples per type for balance
    target_per_type = 2000
    
    print(f"\n🎯 Target: {target_per_type} samples per type")
    
    # Type 0: Object (downsample from original)
    object_samples = train_orig[train_orig['type'] == 0]
    if len(object_samples) > target_per_type:
        object_samples = object_samples.sample(n=target_per_type, random_state=42)
    print(f"  object    (0): {len(object_samples):4d} samples (from train.csv)")
    
    # Type 1: Counting (use augmented + some original)
    counting_aug = train_aug[train_aug['type'] == 1]
    counting_orig = train_orig[train_orig['type'] == 1]
    
    if len(counting_aug) >= target_per_type:
        counting_samples = counting_aug.sample(n=target_per_type, random_state=42)
    else:
        # Use all augmented + fill from original
        need_more = target_per_type - len(counting_aug)
        extra = counting_orig.sample(n=min(need_more, len(counting_orig)), random_state=42)
        counting_samples = pd.concat([counting_aug, extra])
    
    print(f"  counting  (1): {len(counting_samples):4d} samples ({len(counting_aug)} aug + {len(counting_samples)-len(counting_aug)} orig)")
    
    # Type 2: Color (mix augmented + original for diversity)
    color_aug = train_aug[train_aug['type'] == 2]
    color_orig = train_orig[train_orig['type'] == 2]
    
    # Use 50% aug, 50% orig
    n_aug = min(len(color_aug), target_per_type // 2)
    n_orig = target_per_type - n_aug
    
    color_samples = pd.concat([
        color_aug.sample(n=n_aug, random_state=42),
        color_orig.sample(n=min(n_orig, len(color_orig)), random_state=42)
    ])
    print(f"  color     (2): {len(color_samples):4d} samples ({n_aug} aug + {len(color_samples)-n_aug} orig)")
    
    # Type 3: Location (from original + augmented if available)
    location_aug = train_aug[train_aug['type'] == 3] if 3 in train_aug['type'].values else pd.DataFrame()
    location_orig = train_orig[train_orig['type'] == 3]
    
    if len(location_aug) > 0:
        n_aug = min(len(location_aug), target_per_type // 3)
        n_orig = target_per_type - n_aug
        location_samples = pd.concat([
            location_aug.sample(n=n_aug, random_state=42),
            location_orig.sample(n=min(n_orig, len(location_orig)), random_state=42)
        ])
    else:
        location_samples = location_orig.sample(n=min(target_per_type, len(location_orig)), random_state=42)
    
    print(f"  location  (3): {len(location_samples):4d} samples (from train.csv)")
    
    # Combine and shuffle
    print("\n🔀 Combining and shuffling...")
    hybrid = pd.concat([
        object_samples,
        counting_samples,
        color_samples,
        location_samples
    ])
    
    hybrid = hybrid.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save
    output_path = 'train_hybrid.csv'
    hybrid.to_csv(output_path, index=False)
    
    print(f"\n✅ Saved to: {output_path}")
    print(f"\n📊 FINAL DISTRIBUTION:")
    print("-"*50)
    
    total = len(hybrid)
    type_names = {0: "object", 1: "counting", 2: "color", 3: "location"}
    
    for tid in sorted(hybrid['type'].unique()):
        count = (hybrid['type'] == tid).sum()
        pct = count / total * 100
        name = type_names.get(tid, f"type_{tid}")
        print(f"  {name:10s} (type={tid}): {count:4d} ({pct:5.2f}%)")
    
    print(f"\n  {'TOTAL':10s}         : {total:4d} (100.00%)")
    
    # Verify answer distribution for counting
    print(f"\n🔢 COUNTING ANSWERS (Top 8):")
    print("-"*50)
    counting_df = hybrid[hybrid['type'] == 1]
    count_answers = counting_df['answer'].value_counts().head(8)
    for ans, count in count_answers.items():
        pct = count / len(counting_df) * 100
        print(f"  {ans:6s}: {count:4d} ({pct:5.2f}%)")
    
    print("\n" + "="*80)
    print("🎉 DONE! Use train_hybrid.csv for training")
    print("="*80)

if __name__ == "__main__":
    create_hybrid_dataset()
