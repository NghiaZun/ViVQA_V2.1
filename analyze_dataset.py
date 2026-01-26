"""
Analyze ViVQA dataset distribution:
- Question types breakdown
- Answer distribution
- Type confusion errors from validation predictions
"""

import pandas as pd
import json
import re
from collections import Counter
import sys

def classify_question_type(question):
    """Rule-based question type classifier"""
    q = question.lower()
    
    # Color questions
    if any(w in q for w in ["màu", "color"]):
        return "color"
    
    # Counting questions
    if any(w in q for w in ["bao nhiêu", "số lượng", "mấy cái", "có bao", "có mấy"]):
        return "count"
    
    # Location questions
    if any(w in q for w in ["ở đâu", "nằm ở", "đặt ở", "tại đâu", "chỗ nào"]):
        return "location"
    
    # Object identification (what/who)
    if any(w in q for w in ["cái gì", "những gì", "con gì", "ai đang", "người nào"]):
        return "object"
    
    # Action/verb questions
    if any(w in q for w in ["đang làm", "đang làm gì", "làm gì", "hoạt động"]):
        return "action"
    
    return "other"


def analyze_csv(csv_path):
    """Analyze question types and answers from CSV"""
    print(f"\n{'='*80}")
    print(f"📊 ANALYZING: {csv_path}")
    print(f"{'='*80}\n")
    
    df = pd.read_csv(csv_path)
    total = len(df)
    
    # Question type distribution
    print("🎯 QUESTION TYPE DISTRIBUTION:")
    print("-" * 50)
    
    df['q_type'] = df['question'].apply(classify_question_type)
    type_counts = df['q_type'].value_counts()
    
    for qtype, count in type_counts.items():
        pct = count / total * 100
        print(f"  {qtype:12s}: {count:5d} ({pct:5.2f}%)")
    
    print(f"\n  {'TOTAL':12s}: {total:5d} (100.00%)")
    
    # Answer distribution (top 20)
    print(f"\n📝 TOP 20 MOST COMMON ANSWERS:")
    print("-" * 50)
    answer_counts = df['answer'].value_counts()
    
    for i, (ans, count) in enumerate(answer_counts.head(20).items(), 1):
        pct = count / total * 100
        print(f"  {i:2d}. {ans:20s}: {count:4d} ({pct:4.2f}%)")
    
    # Answer diversity
    unique_answers = len(answer_counts)
    print(f"\n📊 ANSWER DIVERSITY:")
    print("-" * 50)
    print(f"  Unique answers: {unique_answers}")
    print(f"  Avg frequency: {total / unique_answers:.2f}")
    
    # Check for type column (if dataset is pre-labeled)
    if 'type' in df.columns or 'question_type' in df.columns:
        type_col = 'type' if 'type' in df.columns else 'question_type'
        print(f"\n✅ Dataset has pre-labeled '{type_col}' column!")
        print("-" * 50)
        labeled_types = df[type_col].value_counts()
        
        type_mapping = {
            0: "object_id",
            1: "counting", 
            2: "color",
            3: "location"
        }
        
        for type_id, count in labeled_types.items():
            pct = count / total * 100
            type_name = type_mapping.get(type_id, f"unknown_{type_id}")
            print(f"  {type_name:12s} (id={type_id}): {count:5d} ({pct:5.2f}%)")
    
    return df


def analyze_type_confusion(df, predictions_sample=None):
    """
    Analyze if predictions are confusing question types
    E.g., color question -> object answer
    """
    if predictions_sample is None:
        print("\n⚠️  No prediction sample provided - skipping confusion analysis")
        return
    
    print(f"\n🔍 TYPE CONFUSION ANALYSIS:")
    print("-" * 50)
    
    confusion_cases = []
    
    for pred in predictions_sample:
        q = pred['question']
        pred_ans = pred['predicted']
        gt_ans = pred['ground_truth']
        
        q_type = classify_question_type(q)
        
        # Detect confusion patterns
        if q_type == "location" and pred_ans in ["màu xanh", "màu đỏ", "màu vàng"]:
            confusion_cases.append({
                'question': q,
                'q_type': q_type,
                'predicted': pred_ans,
                'expected_type': 'color',
                'pattern': 'location->color'
            })
        elif q_type == "color" and pred_ans in ["phòng", "nhà", "cái ghế", "vali"]:
            confusion_cases.append({
                'question': q,
                'q_type': q_type,
                'predicted': pred_ans,
                'expected_type': 'location',
                'pattern': 'color->location'
            })
        elif q_type == "object" and any(c in pred_ans for c in ["màu", "color"]):
            confusion_cases.append({
                'question': q,
                'q_type': q_type,
                'predicted': pred_ans,
                'expected_type': 'color',
                'pattern': 'object->color'
            })
    
    if confusion_cases:
        print(f"  Found {len(confusion_cases)} type confusion cases:")
        for case in confusion_cases[:10]:  # Show first 10
            print(f"\n    ❌ {case['pattern']}")
            print(f"       Q: {case['question']}")
            print(f"       Pred: {case['predicted']}")
    else:
        print("  ✅ No obvious type confusion detected!")


if __name__ == "__main__":
    # Try to find CSV files
    import glob
    
    csv_files = glob.glob("data/**/*.csv", recursive=True)
    csv_files += glob.glob("*.csv")
    
    if not csv_files:
        print("❌ No CSV files found!")
        print("Usage: python analyze_dataset.py <train_csv_path>")
        sys.exit(1)
    
    # Analyze train CSV
    train_csv = None
    for f in csv_files:
        if 'train' in f.lower():
            train_csv = f
            break
    
    if not train_csv:
        train_csv = csv_files[0]
    
    print(f"\n🔍 Auto-detected CSV: {train_csv}")
    df = analyze_csv(train_csv)
    
    # Parse recent training log to extract sample predictions
    print("\n" + "="*80)
    print("💡 To analyze type confusion from your training log:")
    print("   Copy sample predictions and paste when prompted")
    print("="*80)
