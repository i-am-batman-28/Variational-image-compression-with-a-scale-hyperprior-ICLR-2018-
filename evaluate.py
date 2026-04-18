"""
Evaluate a trained Scale Hyperprior checkpoint on Kodak.

Outputs:
  - per-image CSV with bpp, PSNR, MS-SSIM
  - a JPEG baseline sweep for comparison (same image set, quality 10..95)

Usage:
  python evaluate.py --ckpt checkpoints/lambda0013.pt --kodak data/kodak --out results/ours_lambda0013.csv
  python evaluate.py --jpeg-only --kodak data/kodak --out results/jpeg.csv
"""

import argparse
import csv
import io
import pathlib

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from models import ScaleHyperprior


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def psnr(x, y):
    mse = F.mse_loss(x, y).clamp(min=1e-10)
    return 10 * torch.log10(1.0 / mse)


def ms_ssim(x, y, device):
    # lightweight dependency: use pytorch_msssim if available, else skip
    try:
        from pytorch_msssim import ms_ssim as _ms_ssim
        return _ms_ssim(x, y, data_range=1.0, size_average=True)
    except ImportError:
        return torch.tensor(float("nan"), device=device)


def pad_to_multiple(x, m=64):
    _, _, h, w = x.shape
    nh = (h + m - 1) // m * m
    nw = (w + m - 1) // m * m
    return F.pad(x, (0, nw - w, 0, nh - h), mode="replicate"), (h, w)


def eval_model(args):
    device = pick_device()
    ckpt = torch.load(args.ckpt, map_location=device)
    cargs = ckpt.get("args", {})
    model = ScaleHyperprior(N=cargs.get("N", 128), M=cargs.get("M", 192)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    to_tensor = transforms.ToTensor()
    rows = []
    paths = sorted(pathlib.Path(args.kodak).glob("*.png"))
    with torch.no_grad():
        for p in paths:
            img = to_tensor(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
            x_padded, (h, w) = pad_to_multiple(img)
            out = model(x_padded)
            x_hat = out["x_hat"][:, :, :h, :w].clamp(0, 1)
            row = {
                "image": p.name,
                "bpp": out["bpp"].item(),
                "psnr": psnr(x_hat, img).item(),
                "ms_ssim": ms_ssim(x_hat, img, device).item(),
            }
            rows.append(row)
            print(row)
    write_csv(rows, args.out)


def eval_jpeg(args):
    to_tensor = transforms.ToTensor()
    paths = sorted(pathlib.Path(args.kodak).glob("*.png"))
    qualities = args.qualities
    rows = []
    for q in qualities:
        for p in paths:
            img = Image.open(p).convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=q)
            nbytes = buf.tell()
            buf.seek(0)
            recon = Image.open(buf).convert("RGB")

            x = to_tensor(img).unsqueeze(0)
            y = to_tensor(recon).unsqueeze(0)
            bpp = nbytes * 8 / (img.size[0] * img.size[1])
            rows.append({
                "image": p.name,
                "quality": q,
                "bpp": bpp,
                "psnr": psnr(y, x).item(),
                "ms_ssim": ms_ssim(y, x, "cpu").item(),
            })
            print(rows[-1])
    write_csv(rows, args.out)


def write_csv(rows, path):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", help="model checkpoint (omit with --jpeg-only)")
    p.add_argument("--kodak", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--jpeg-only", action="store_true")
    p.add_argument("--qualities", type=int, nargs="+",
                   default=[10, 20, 30, 50, 75, 95])
    args = p.parse_args()
    if args.jpeg_only:
        eval_jpeg(args)
    else:
        eval_model(args)


if __name__ == "__main__":
    main()
