import os, sys
from PIL import Image, ImageFilter
from concurrent.futures import ThreadPoolExecutor

SRC = "archive/data/images/test"
LEVELS = {"blur_mild": 3, "blur_med": 6, "blur_severe": 12}

def process(fname, out_dir, radius):
    src_path = os.path.join(SRC, fname)
    dst_path = os.path.join(out_dir, fname)
    if os.path.exists(dst_path):
        return
    try:
        img = Image.open(src_path).convert("RGB")
        img = img.filter(ImageFilter.GaussianBlur(radius=radius))
        img.save(dst_path, quality=95)
    except Exception as e:
        print(f"ERR {fname}: {e}", file=sys.stderr)

def main():
    files = os.listdir(SRC)
    print(f"{len(files)} images in {SRC}")
    for name, radius in LEVELS.items():
        out_dir = f"robustness/images_{name}"
        os.makedirs(out_dir, exist_ok=True)
        with ThreadPoolExecutor(max_workers=16) as ex:
            list(ex.map(lambda f: process(f, out_dir, radius), files))
        print(f"done {name} (radius={radius}) -> {out_dir}")

if __name__ == "__main__":
    main()
