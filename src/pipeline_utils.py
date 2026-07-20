import os
import shutil
import subprocess

import pandas as pd

from config import CFG

combined_image_dir = os.path.join(CFG['work_dir'], 'combined_images')


def rebuild_combined_image_dir(aug_image_dir=None):
    """
    Tạo lại thư mục combined_images bằng cách symlink ảnh gốc
    và copy ảnh augmented từ aug_image_dir (nếu có).
    """
    if os.path.exists(combined_image_dir):
        shutil.rmtree(combined_image_dir)
    os.makedirs(combined_image_dir)

    for f in os.listdir(CFG['image_dir']):
        os.symlink(
            os.path.join(CFG['image_dir'], f),
            os.path.join(combined_image_dir, f),
        )

    if aug_image_dir and os.path.isdir(aug_image_dir):
        for f in os.listdir(aug_image_dir):
            src = os.path.join(aug_image_dir, f)
            dst = os.path.join(combined_image_dir, f)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)

    total = len(os.listdir(combined_image_dir))
    print(f'✓ Combined image dir: {total} images')
    return combined_image_dir


def merge_dataset(current_train_csv, new_rows, loop_idx):
    """
    Ghép new_rows vào current_train_csv, lưu train_loop{loop_idx}.csv.
    Trả về path CSV mới (hoặc current nếu new_rows rỗng).
    """
    if not new_rows:
        print('  ⚠️  0 ảnh mới — skip merge, giữ nguyên train CSV')
        return current_train_csv

    df_current = pd.read_csv(current_train_csv)
    df_new     = pd.DataFrame(new_rows)
    df_merged  = pd.concat([df_current, df_new], ignore_index=True)

    new_csv_path = os.path.join(CFG['work_dir'], f'train_loop{loop_idx}.csv')
    df_merged.to_csv(new_csv_path, index=False)

    print(f'  {len(df_current)} → {len(df_merged)} samples (+{len(df_new)} new)')
    print(df_merged['type'].value_counts().sort_index().to_string())
    return new_csv_path


def recompute_answer_weights(train_csv, output='answer_weights.json'):
    subprocess.run(
        f'python compute_answer_weights.py --train_csv {train_csv} --output {output}',
        shell=True, check=True,
    )
    print(f'✓ Answer weights updated → {output}')
    return output
