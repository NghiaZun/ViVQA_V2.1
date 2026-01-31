# 🔧 FIX: CSV FILE NOT SAVING ON KAGGLE

## ❌ Vấn đề
Sau khi chạy eval, thấy message "Saved to results_siglip_test.csv" nhưng **file không tồn tại** trên Kaggle.

## ✅ Nguyên nhân và giải pháp

### 1️⃣ **Thiếu questions trong output**
**Fixed in**: `eval_minimal.py` (updated version)
- Added `all_questions` tracking
- Included in CSV output: question, prediction, ground_truth, EM, F1

### 2️⃣ **Relative path → Absolute path**
**Before**: `--output_csv results_siglip_test.csv` (không rõ save ở đâu)
**After**: `--output_csv /kaggle/working/results_siglip_test.csv` (chắc chắn)

### 3️⃣ **Thiếu error handling**
**Fixed**: Added try-except và verification:
```python
df.to_csv(args.output_csv, index=False, encoding='utf-8')

# Verify file exists
if os.path.exists(args.output_csv):
    print(f"✅ Saved to {args.output_csv}")
    print(f"   File size: {os.path.getsize(args.output_csv):,} bytes")
else:
    print(f"❌ ERROR: File not created")
```

---

## 🚀 Cách sử dụng (3 options)

### Option 1: Shell Script (RECOMMENDED)
```bash
cd /kaggle/working/ViVQA_V2.1
bash run_eval_minimal.sh
```

**Output sẽ có**:
- Debug info (file save test)
- Evaluation progress
- File verification
- Preview first 5 lines

---

### Option 2: Direct Python
```bash
python /kaggle/working/ViVQA_V2.1/eval_minimal.py \
    --checkpoint /kaggle/input/sigclip-v1/transformers/default/1/best_model.pt \
    --csv_path /kaggle/input/vivqa/data/test.csv \
    --image_folder /kaggle/input/vivqa/data/images/test \
    --vision_model google/siglip-base-patch16-224 \
    --batch_size 16 \
    --output_csv /kaggle/working/results_siglip_test.csv
```

---

### Option 3: Kaggle Notebook (BEST FOR INTERACTIVE)
Copy nội dung từ `kaggle_notebook_eval.py` vào notebook cells:

**Cell 1**: Setup paths
**Cell 2**: Debug file saving
**Cell 3**: Run evaluation
**Cell 4**: Load và verify results
**Cell 5**: Analyze per-type performance

**Benefits**:
- Interactive display
- Download link tự động
- Per-type analysis
- Error visualization

---

## 📂 Tìm file sau khi save

### Trong Kaggle Notebook:
```python
import os
print(os.listdir('/kaggle/working'))
```

### Download file:
```python
from IPython.display import FileLink
FileLink('/kaggle/working/results_siglip_test.csv')
```

### Check qua terminal:
```bash
ls -lh /kaggle/working/*.csv
```

---

## 📊 CSV Format

File output sẽ có các columns:

| Column | Description | Example |
|--------|-------------|---------|
| question | Câu hỏi gốc | "Có bao nhiêu con chó?" |
| prediction | Model prediction | "2 con" |
| ground_truth | Ground truth answer | "hai con" |
| exact_match | 0 hoặc 1 | 0 (không khớp chính xác) |
| f1_score | 0.0 - 1.0 | 0.67 (67% overlap) |

---

## 🐛 Debugging

Nếu vẫn không thấy file, chạy debug script:

```bash
python /kaggle/working/ViVQA_V2.1/debug_kaggle_save.py
```

Sẽ test:
- Current working directory
- Write permissions
- Pandas CSV save
- List files trong /kaggle/working

---

## 💡 Tips

1. **Always use absolute paths** trên Kaggle: `/kaggle/working/...`
2. **Check file ngay sau eval**:
   ```bash
   ls -lh /kaggle/working/results_siglip_test.csv
   ```
3. **Load và verify**:
   ```python
   import pandas as pd
   df = pd.read_csv('/kaggle/working/results_siglip_test.csv')
   print(f"Loaded {len(df)} rows")
   ```

---

## 📈 Next Steps sau khi có CSV

1. **Analyze results**:
   ```bash
   python analyze_results.py /kaggle/working/results_siglip_test.csv
   ```

2. **Find weak types** (COUNT? COLOR? LOCATION?)

3. **Target improvements** dựa trên analysis

4. **Re-train với anti-overfit** để tăng từ 61.45% → 70%+

---

## ✅ Checklist

- [ ] Updated `eval_minimal.py` (include questions)
- [ ] Use absolute path: `/kaggle/working/results_siglip_test.csv`
- [ ] Run `bash run_eval_minimal.sh`
- [ ] Verify file exists: `ls -lh /kaggle/working/*.csv`
- [ ] Load and check: `pd.read_csv(...)`
- [ ] Download hoặc analyze results
