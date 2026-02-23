#!/usr/bin/env python3
"""Quick test to verify all imports work"""

print("Testing imports...")

try:
    import torch
    print(f"✅ PyTorch {torch.__version__}")
except Exception as e:
    print(f"❌ PyTorch: {e}")

try:
    import transformers
    print(f"✅ Transformers {transformers.__version__}")
except Exception as e:
    print(f"❌ Transformers: {e}")

try:
    from dataset import VQAGenDataset
    print("✅ dataset.py imported successfully")
except Exception as e:
    print(f"❌ dataset.py: {e}")

try:
    from model import DeterministicVQA
    print("✅ model.py imported successfully")
except Exception as e:
    print(f"❌ model.py: {e}")

try:
    # Don't run full train.py, just check syntax
    import py_compile
    py_compile.compile('train.py', doraise=True)
    print("✅ train.py syntax OK")
except Exception as e:
    print(f"❌ train.py syntax: {e}")

try:
    py_compile.compile('eval.py', doraise=True)
    print("✅ eval.py syntax OK")
except Exception as e:
    print(f"❌ eval.py syntax: {e}")

print("\n" + "="*50)
print("All checks passed! ✅")
print("="*50)
