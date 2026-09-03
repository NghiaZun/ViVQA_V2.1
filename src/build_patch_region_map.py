"""
Tinh ban do patch->region (14x14=196 patch cua SigLIP) cho tung anh, tu annotation COCO da
quet (coco_annotations_matched.pkl). Voi moi patch: tim bbox COCO NHO NHAT (dien tich be nhat)
chua tam patch do (uu tien vat the cu the hon vung nen lon) -- 0 = nen (khong thuoc bbox nao).

Luu: dict {img_id: int16[196]} -- gia tri la CHI SO bbox (0=nen, 1..K=bbox thu K trong anh do,
danh so theo thu tu xuat hien trong annotation).
"""
import argparse
import pickle
import numpy as np


def compute_region_map(width, height, bboxes, grid=14):
    """Tra ve int16[grid*grid]: chi so bbox (1-indexed) nho nhat chua tam patch, 0=nen."""
    patch_w = width / grid
    patch_h = height / grid
    region_ids = np.zeros(grid * grid, dtype=np.int16)
    if len(bboxes) == 0:
        return region_ids
    # sap xep bbox theo DIEN TICH TANG DAN -> uu tien gan sau (bbox nho hon = cu the hon)
    areas = [(bb[2] * bb[3], idx) for idx, bb in enumerate(bboxes)]  # bbox = [x,y,w,h]
    areas.sort(key=lambda t: t[0], reverse=True)  # gan LON truoc, NHO SAU de NHO ghi de LON
    for row in range(grid):
        for col in range(grid):
            cx = (col + 0.5) * patch_w
            cy = (row + 0.5) * patch_h
            patch_idx = row * grid + col
            for area, idx in areas:
                x, y, w, h = bboxes[idx]
                if x <= cx <= x + w and y <= cy <= y + h:
                    region_ids[patch_idx] = idx + 1  # 1-indexed, 0 danh cho nen
    return region_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in_pkl', default='coco_annotations_matched.pkl')
    ap.add_argument('--out', default='patch_region_map.pkl')
    ap.add_argument('--grid', type=int, default=14)
    args = ap.parse_args()

    annots = pickle.load(open(args.in_pkl, 'rb'))
    print(f"Tinh region map cho {len(annots)} anh...")
    out = {}
    n_regions_hist = []
    for i, (img_id, a) in enumerate(annots.items()):
        bboxes = list(a['bbox'])  # list cua [x,y,w,h] (COCO format: xywh)
        rmap = compute_region_map(a['width'], a['height'], bboxes, grid=args.grid)
        out[int(img_id)] = rmap
        n_regions_hist.append(len(bboxes))
        if (i + 1) % 5000 == 0:
            print(f"  {i+1}/{len(annots)}")

    import statistics as st
    print(f"\nSo region/anh: mean={st.mean(n_regions_hist):.2f} median={st.median(n_regions_hist)} "
          f"max={max(n_regions_hist)} min={min(n_regions_hist)}")
    pickle.dump(out, open(args.out, 'wb'))
    print(f"Saved {len(out)} region-map toi {args.out}")


if __name__ == '__main__':
    main()
