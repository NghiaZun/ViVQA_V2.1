#!/usr/bin/env python3
"""
Quick fix for eval_siglip.py on Kaggle
Run this before evaluation
"""

import sys

# Read the file
with open('eval_siglip.py', 'r') as f:
    content = f.read()

# Fix the parameter name
content = content.replace(
    "text_model_name='vinai/bartpho-syllable'",
    "bartpho_model_name='vinai/bartpho-syllable'"
)

# Write back
with open('eval_siglip.py', 'w') as f:
    f.write(content)

print("✅ Fixed eval_siglip.py!")
print("   Changed: text_model_name → bartpho_model_name")
