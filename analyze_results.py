"""
Analyze test results and find improvement opportunities
"""
import pandas as pd
import sys

# Load results
csv_path = sys.argv[1] if len(sys.argv) > 1 else 'results_siglip_test.csv'

print(f"Loading {csv_path}...")
df = pd.read_csv(csv_path)

print("\n" + "="*80)
print("OVERALL PERFORMANCE")
print("="*80)
print(f"Total samples: {len(df)}")
print(f"Exact Match: {df['exact_match'].mean()*100:.2f}%")
print(f"F1 Score: {df['f1_score'].mean()*100:.2f}%")

# Detect question types
def detect_type(q):
    q = str(q).lower()
    if 'bao nhiêu' in q or 'mấy' in q:
        return 'COUNT'
    elif 'màu' in q:
        return 'COLOR'
    elif 'đâu' in q or 'ở đâu' in q:
        return 'LOCATION'
    else:
        return 'OBJECT'

df['type'] = df.get('question', [''] * len(df)).apply(detect_type)

print("\n" + "="*80)
print("PER-TYPE PERFORMANCE")
print("="*80)
type_stats = df.groupby('type').agg({
    'exact_match': ['count', 'mean'],
    'f1_score': 'mean'
}).round(4)
print(type_stats)

# Top failures
print("\n" + "="*80)
print("TOP 20 FAILURES (EM=0, but high F1)")
print("="*80)
failures = df[df['exact_match'] == 0].copy()
failures = failures.sort_values('f1_score', ascending=False).head(20)

for idx, row in failures.iterrows():
    print(f"\n{idx+1}. F1={row.get('f1_score', 0):.2f}")
    if 'question' in row:
        print(f"   Q: {row['question']}")
    print(f"   Pred: {row['prediction']}")
    print(f"   GT:   {row['ground_truth']}")

# Common error patterns
print("\n" + "="*80)
print("COMMON ERROR PATTERNS")
print("="*80)

# Prediction length analysis
df['pred_len'] = df['prediction'].str.split().str.len()
df['gt_len'] = df['ground_truth'].str.split().str.len()

print(f"\nPrediction length:")
print(f"  Mean: {df['pred_len'].mean():.1f} tokens")
print(f"  GT Mean: {df['gt_len'].mean():.1f} tokens")
print(f"  Too short (<GT): {(df['pred_len'] < df['gt_len']).sum()} samples")
print(f"  Too long (>GT): {(df['pred_len'] > df['gt_len']).sum()} samples")

# Check for common wrong predictions
print(f"\nMost common wrong predictions:")
wrong_preds = df[df['exact_match'] == 0]['prediction'].value_counts().head(10)
print(wrong_preds)

print("\n" + "="*80)
print("RECOMMENDATIONS")
print("="*80)
print("1. Focus on types with lowest EM")
print("2. Review 'almost correct' predictions (high F1 but EM=0)")
print("3. Add post-processing for common errors")
print("4. Check if using correct checkpoint (epoch 6 for validation best)")
print("5. Consider ensemble or TTA for +2-5% boost")
