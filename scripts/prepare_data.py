"""Turn data/raw/{A,B}/*.png into the CycleGAN folder layout at 256x256.

Plans are scaled so the longer side is SIZE (area averaging keeps thin wall lines
visible) and centred on a white square canvas, so aspect ratio is never distorted.
Output: data/cyclegan/{trainA,trainB,testA,testB}
"""
import random
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "cyclegan"
SIZE = 256
N_TEST = {"A": 50, "B": 40}
SEED = 0

Image.MAX_IMAGE_PIXELS = None


def load_square(path):
    im = Image.open(path)
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im)
    im = im.convert("RGB")
    w, h = im.size
    s = SIZE / max(w, h)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    # BOX = area averaging; do it in two stages for big inputs so it stays fast
    im = im.resize((nw, nh), Image.BOX if s < 1 else Image.BICUBIC)
    canvas = Image.new("RGB", (SIZE, SIZE), (255, 255, 255))
    canvas.paste(im, ((SIZE - nw) // 2, (SIZE - nh) // 2))
    return canvas


rng = random.Random(SEED)
for dom in ("A", "B"):
    files = sorted((RAW / dom).glob("*.png"))
    rng.shuffle(files)
    splits = {"test": files[: N_TEST[dom]], "train": files[N_TEST[dom]:]}
    for split, fs in splits.items():
        d = OUT / f"{split}{dom}"
        d.mkdir(parents=True, exist_ok=True)
        bad = 0
        for f in fs:
            try:
                load_square(f).save(d / f"{f.stem}.png")
            except Exception as e:
                bad += 1
                print("skip", f.name, e)
        print(f"{split}{dom}: {len(fs) - bad} images")
