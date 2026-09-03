"""MUC 5 CUA THAY — chay model tren MOT ANH THAT + MOT CAU HOI TIENG VIET bat ky.

    python3 demo_vqa.py --image anh.jpg --question "màu của chiếc xe là gì"
    python3 demo_vqa.py --image anh.jpg --question "có bao nhiêu con mèo" --heatmap out.png

ENCODER: SigLIP1 (google/siglip-base-patch16-224) — checkpoint run87, dung encoder cua BAI BAO.

VI SAO GOI LAI eval.py THAY VI TU DUNG MODEL:
  eval.py suy ra hang chuc co kien truc TU CHINH checkpoint (siglip_pooler, type head, gate min/max
  alpha, per-type floor, lora rank...). Du an nay da tung dinh mot lan load_model_from_checkpoint
  danh roi ortho_lambda/sc_skt, lam ca mot lo ket qua E28 thanh vo nghia. Dung lai duong nap cua
  eval.py thi demo chac chan chay DUNG con model da bao cao, khong phai mot ban giong giong.
  Doi lai la cham hon (moi lan goi nap lai model ~40s) — chap nhan duoc cho mot demo.

Dau ra: dap an, loai cau hoi model TU DOAN, va alpha trung binh (muc do model dua vao anh).
"""
import argparse, os, subprocess, sys, tempfile, shutil
import pandas as pd

MAIN = os.path.dirname(os.path.abspath(__file__))
PY = '/home/user/workspace/all_env/vivqa/bin/python3'
TYPE_NAME = {0: 'OBJECT', 1: 'COUNT', 2: 'COLOR', 3: 'LOCATION'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', required=True, help='duong dan anh bat ky (jpg/png)')
    ap.add_argument('--question', required=True, help='cau hoi tieng Viet')
    ap.add_argument('--checkpoint', default=f'{MAIN}/checkpoints_run87/best_model.pt',
                    help='mac dinh: run87 = SigLIP1, 72.34 test EM')
    ap.add_argument('--heatmap', default=None, help='neu dat, luu anh chong alpha ra duong dan nay')
    ap.add_argument('--num_beams', type=int, default=3)
    a = ap.parse_args()

    if not os.path.isfile(a.image):
        sys.exit(f'khong thay anh: {a.image}')
    if not os.path.isfile(a.checkpoint):
        sys.exit(f'khong thay checkpoint: {a.checkpoint}')

    tmp = tempfile.mkdtemp(prefix='demo_vqa_')
    try:
        imgdir = os.path.join(tmp, 'images')
        os.makedirs(imgdir)
        img_id = 999999                       # id gia, chi de eval.py ghep ten file
        shutil.copy(a.image, os.path.join(imgdir, f'{img_id}.jpg'))

        csv = os.path.join(tmp, 'one.csv')
        # cot 'answer' va 'type' la BAT BUOC theo dinh dang dataset, nhung khong anh huong du doan:
        # type_mode mac dinh la 'predicted', nen model tu doan loai, khong doc cot 'type'.
        pd.DataFrame([{'question': a.question, 'answer': '<khong biet>',
                       'img_id': img_id, 'type': 0}]).to_csv(csv)

        out_csv = os.path.join(tmp, 'out.csv')
        alpha_npy = os.path.join(tmp, 'alpha.npy')
        cmd = [PY, 'src/eval.py',
               '--checkpoint', a.checkpoint,
               '--csv_path', csv,
               '--image_folder', imgdir,
               '--output_csv', out_csv,
               '--num_beams', str(a.num_beams),
               '--repetition_penalty', '1.3', '--max_length', '10',
               '--use_synonyms', '--use_constrained',
               '--train_csv_for_trie', 'archive/train_split.csv',
               '--dump_model_alpha', alpha_npy]
        r = subprocess.run(cmd, cwd=MAIN, capture_output=True, text=True)
        if not os.path.isfile(out_csv):
            print(r.stdout[-3000:]); print(r.stderr[-3000:])
            sys.exit('eval.py khong sinh duoc ket qua')

        d = pd.read_csv(out_csv).iloc[0]
        print()
        print(f'  cau hoi   : {a.question}')
        print(f'  anh       : {a.image}')
        print(f'  DAP AN    : {d.prediction}')
        pt = d.get('pred_question_type', None)
        if pt is not None and not pd.isna(pt):
            print(f'  loai (model tu doan): {TYPE_NAME.get(int(pt), pt) if str(pt).isdigit() else pt}')

        if os.path.isfile(alpha_npy):
            import numpy as np
            al = np.load(alpha_npy)
            al = al[0] if al.ndim > 1 else al
            print(f'  alpha     : trung binh {al.mean():.4f}  (thap = dua vao cau hoi nhieu hon,'
                  f' cao = dua vao anh nhieu hon)')
            if a.heatmap:
                save_heatmap(a.image, al, a.heatmap)
                print(f'  heatmap   : {a.heatmap}')
        print()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def save_heatmap(img_path, alpha, out_path):
    """Chong alpha per-patch len anh goc. 196 patch = luoi 14x14 cua SigLIP-base-patch16-224."""
    import numpy as np
    from PIL import Image
    a = np.asarray(alpha, dtype=np.float32).ravel()
    n = int(round(len(a) ** 0.5))
    if n * n != len(a):                       # co the co token pooler o dau
        a = a[-n * n:] if len(a) > n * n else a
        n = int(round(len(a) ** 0.5))
    g = a[:n * n].reshape(n, n)
    g = (g - g.min()) / (np.ptp(g) + 1e-8)
    im = Image.open(img_path).convert('RGB')
    hm = Image.fromarray((g * 255).astype('uint8')).resize(im.size, Image.BILINEAR)
    hm = np.asarray(hm, dtype=np.float32) / 255.0
    base = np.asarray(im, dtype=np.float32)
    # do = alpha cao (model nhin vao day), xanh = alpha thap
    overlay = np.stack([hm * 255, np.zeros_like(hm), (1 - hm) * 255], -1)
    Image.fromarray((0.6 * base + 0.4 * overlay).clip(0, 255).astype('uint8')).save(out_path)


if __name__ == '__main__':
    main()
