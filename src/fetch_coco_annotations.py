"""
Quet toan bo shard cua detection-datasets/coco (COCO2017, instance bbox annotation) qua HTTP
range request (CHI doc cot image_id + objects, KHONG tai cot image nang) -- loc lay annotation
cho dung cac img_id can dung (ViVQA + ViVQA-X). Luu ra 1 file pickle: {img_id: {bbox, category,
area, width, height}} de dung lai nhieu lan, khong can quet lai.

Ly do can width/height: bbox trong COCO la toa do pixel goc (anh chua resize), can de quy doi
sang luoi patch 14x14 cua SigLIP (anh resize ve 224x224).
"""
import argparse
import pickle
import time

import pyarrow.parquet as pq
import fsspec
from huggingface_hub import HfApi, hf_hub_url


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target_ids_pkl', default='/tmp/target_img_ids.pkl')
    ap.add_argument('--repo_id', default='detection-datasets/coco')
    ap.add_argument('--out', default='coco_annotations_matched.pkl')
    args = ap.parse_args()

    target_ids = pickle.load(open(args.target_ids_pkl, 'rb'))
    print(f"Can tim annotation cho {len(target_ids)} img_id")

    api = HfApi()
    info = api.dataset_info(args.repo_id)
    shard_files = sorted([s.rfilename for s in info.siblings if s.rfilename.endswith('.parquet')])
    print(f"Tong {len(shard_files)} shard trong {args.repo_id}")

    fs = fsspec.filesystem('http')
    found = {}
    remaining = set(target_ids)

    for i, fname in enumerate(shard_files):
        if not remaining:
            print("Da tim du, dung som.")
            break
        url = hf_hub_url(repo_id=args.repo_id, repo_type='dataset', filename=fname)
        t0 = time.time()
        try:
            with fs.open(url, 'rb') as f:
                pf = pq.ParquetFile(f)
                tbl = pf.read(columns=['image_id', 'width', 'height', 'objects'])
        except Exception as e:
            print(f"  [{i+1}/{len(shard_files)}] {fname}: LOI {e}")
            continue
        df = tbl.to_pandas()
        hit = df[df['image_id'].isin(remaining)]
        for _, row in hit.iterrows():
            found[row['image_id']] = {
                'width': row['width'], 'height': row['height'],
                'bbox': row['objects']['bbox'], 'category': row['objects']['category'],
                'area': row['objects']['area'],
            }
        remaining -= set(hit['image_id'].tolist())
        print(f"  [{i+1}/{len(shard_files)}] {fname}: {len(df)} anh, khop moi {len(hit)}, "
              f"con thieu {len(remaining)}, {time.time()-t0:.1f}s")

    print(f"\nTong: tim duoc {len(found)}/{len(target_ids)} img_id "
          f"({len(found)/len(target_ids)*100:.1f}%)")
    pickle.dump(found, open(args.out, 'wb'))
    print(f"Saved to {args.out}")


if __name__ == '__main__':
    main()
