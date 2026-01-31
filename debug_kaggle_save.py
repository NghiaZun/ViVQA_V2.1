"""
DEBUG: Test file saving on Kaggle
"""
import os
import pandas as pd

print("="*80)
print("FILE SAVING DEBUG")
print("="*80)

# Check current directory
print(f"\n1. Current working directory:")
print(f"   {os.getcwd()}")

# Check write permissions
print(f"\n2. Testing write permissions:")
test_dirs = [
    '/kaggle/working',
    '/kaggle/working/ViVQA_V2.1',
    '.'
]

for test_dir in test_dirs:
    try:
        test_file = os.path.join(test_dir, 'test_write.txt')
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
        print(f"   ✅ {test_dir} - WRITABLE")
    except Exception as e:
        print(f"   ❌ {test_dir} - ERROR: {e}")

# Test pandas save
print(f"\n3. Testing pandas CSV save:")
test_df = pd.DataFrame({
    'col1': [1, 2, 3],
    'col2': ['a', 'b', 'c']
})

test_paths = [
    '/kaggle/working/test_results.csv',
    'test_results.csv',
    './test_results.csv'
]

for test_path in test_paths:
    try:
        test_df.to_csv(test_path, index=False)
        if os.path.exists(test_path):
            size = os.path.getsize(test_path)
            print(f"   ✅ {test_path} - SAVED ({size} bytes)")
            os.remove(test_path)
        else:
            print(f"   ❌ {test_path} - NOT FOUND after save")
    except Exception as e:
        print(f"   ❌ {test_path} - ERROR: {e}")

# List files in working directory
print(f"\n4. Files in /kaggle/working:")
try:
    files = os.listdir('/kaggle/working')
    for f in files[:20]:  # First 20 files
        print(f"   - {f}")
    if len(files) > 20:
        print(f"   ... and {len(files)-20} more files")
except Exception as e:
    print(f"   ERROR: {e}")

print("\n" + "="*80)
print("RECOMMENDATION:")
print("Use absolute path: /kaggle/working/results.csv")
print("="*80)
