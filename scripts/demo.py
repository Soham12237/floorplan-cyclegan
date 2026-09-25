r"""Run the trained CycleGAN on any image and save a side-by-side result.

  .venv\Scripts\python scripts\demo.py path\to\plan.png             # A -> B (blueprint -> colorful)
  .venv\Scripts\python scripts\demo.py path\to\plan.png --direction BtoA
  .venv\Scripts\python scripts\demo.py data\cyclegan\testA          # a whole folder

Output goes to demo_out\<name>_<direction>.png (input on the left, result on the right).
"""
import argparse
import sys
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "cyclegan"))
from models.networks import define_G  # noqa: E402

SIZE = 256
EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def load_generator(direction, epoch, device):
    name = "G_A" if direction == "AtoB" else "G_B"
    net = define_G(3, 3, 64, "resnet_9blocks", "instance", False, "normal", 0.02)
    path = ROOT / "checkpoints" / "floorplan_cyclegan" / f"{epoch}_net_{name}.pth"
    net.load_state_dict(torch.load(path, map_location="cpu"))
    return net.to(device).eval()


def square_canvas(im):
    """Same preprocessing as scripts/prepare_data.py: keep aspect ratio, pad with white."""
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im)
    im = im.convert("RGB")
    w, h = im.size
    s = SIZE / max(w, h)
    im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BOX if s < 1 else Image.BICUBIC)
    canvas = Image.new("RGB", (SIZE, SIZE), (255, 255, 255))
    canvas.paste(im, ((SIZE - im.width) // 2, (SIZE - im.height) // 2))
    return canvas


def to_tensor(im):
    t = torch.frombuffer(bytearray(im.tobytes()), dtype=torch.uint8).view(SIZE, SIZE, 3)
    return (t.permute(2, 0, 1).float() / 127.5 - 1).unsqueeze(0)


def to_image(t):
    a = ((t.squeeze(0).clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="image file or folder of images")
    ap.add_argument("--direction", choices=["AtoB", "BtoA"], default="AtoB")
    ap.add_argument("--epoch", default="latest", help="checkpoint prefix: latest, 50, 25, ...")
    ap.add_argument("--out", default=str(ROOT / "demo_out"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = load_generator(args.direction, args.epoch, device)

    src = Path(args.input)
    files = sorted(p for p in src.iterdir() if p.suffix.lower() in EXTS) if src.is_dir() else [src]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for f in files:
        x = square_canvas(Image.open(f))
        with torch.no_grad():
            y = to_image(net(to_tensor(x).to(device)))
        pair = Image.new("RGB", (SIZE * 2, SIZE), (255, 255, 255))
        pair.paste(x, (0, 0))
        pair.paste(y, (SIZE, 0))
        dst = out / f"{f.stem}_{args.direction}.png"
        pair.save(dst)
        print(dst)


if __name__ == "__main__":
    main()
