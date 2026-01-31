"""
KAGGLE NOTEBOOK - EVALUATE AND SAVE RESULTS
Run this in Kaggle notebook cells
"""

# Cell 1: Setup
import os
import sys
sys.path.append('/kaggle/working/ViVQA_V2.1')

# Cell 2: Debug file saving
print("Testing file save capabilities...")
exec(open('/kaggle/working/ViVQA_V2.1/debug_kaggle_save.py').read())

# Cell 3: Run evaluation
import subprocess

cmd = [
    'python', '/kaggle/working/ViVQA_V2.1/eval_minimal.py',
    '--checkpoint', '/kaggle/input/sigclip-v1/transformers/default/1/best_model.pt',
    '--csv_path', '/kaggle/input/vivqa/data/test.csv',
    '--image_folder', '/kaggle/input/vivqa/data/images/test',
    '--vision_model', 'google/siglip-base-patch16-224',
    '--batch_size', '16',
    '--output_csv', '/kaggle/working/results_siglip_test.csv'
]

print("Running evaluation...")
result = subprocess.run(cmd, capture_output=False, text=True)

# Cell 4: Verify and load results
import pandas as pd

csv_path = '/kaggle/working/results_siglip_test.csv'

if os.path.exists(csv_path):
    print(f"✅ File found: {csv_path}")
    print(f"   Size: {os.path.getsize(csv_path):,} bytes")
    
    df = pd.read_csv(csv_path)
    print(f"   Rows: {len(df)}")
    print(f"\nFirst 5 rows:")
    display(df.head())
    
    print(f"\nMetrics:")
    print(f"EM: {df['exact_match'].mean()*100:.2f}%")
    print(f"F1: {df['f1_score'].mean()*100:.2f}%")
    
    # Enable download
    from IPython.display import FileLink
    print(f"\n📥 Download file:")
    display(FileLink(csv_path))
    
else:
    print(f"❌ File NOT found: {csv_path}")
    print(f"\nFiles in /kaggle/working:")
    for f in os.listdir('/kaggle/working'):
        if f.endswith('.csv'):
            print(f"  - {f}")

# Cell 5: Analyze results (if file exists)
if os.path.exists(csv_path):
    # Detect question types
    def detect_type(q):
        q = str(q).lower()
        if 'bao nhiêu' in q or 'mấy' in q:
            return 'COUNT'
        elif 'màu' in q:
            return 'COLOR'
        elif 'đâu' in q:
            return 'LOCATION'
        else:
            return 'OBJECT'
    
    df['type'] = df['question'].apply(detect_type)
    
    print("Per-type performance:")
    type_stats = df.groupby('type').agg({
        'exact_match': ['count', 'mean'],
        'f1_score': 'mean'
    })
    display(type_stats)
    
    print("\nTop 10 failures (EM=0 but high F1):")
    failures = df[df['exact_match'] == 0].sort_values('f1_score', ascending=False).head(10)
    display(failures[['question', 'prediction', 'ground_truth', 'f1_score']])
