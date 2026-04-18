"""
Produce a side-by-side grid: Original | JPEG Q10 | Ours Q3  for 3 Kodak images.
Saves results/demo_grid.png
"""

import io
import pathlib

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from models import ScaleHyperprior


def pick_device():
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"


def pad_mult(x, m=64):
    _, _, h, w = x.shape
    nh, nw = (h + m - 1) // m * m, (w + m - 1) // m * m
    return F.pad(x, (0, nw - w, 0, nh - h), "replicate"), (h, w)


def jpeg_roundtrip(img: Image.Image, q: int):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=q)
    nbytes = buf.tell()
    buf.seek(0)
    return Image.open(buf).convert("RGB"), nbytes


def psnr(a, b):
    mse = ((a - b) ** 2).mean().clamp(min=1e-10)
    return (10 * torch.log10(1.0 / mse)).item()


def main():
    device = pick_device()
    ckpt = torch.load("checkpoints/compressai_q3.pt", map_location=device)
    model = ScaleHyperprior(N=128, M=192).to(device).eval()
    model.load_state_dict(ckpt["model"])
    to_t = transforms.ToTensor()

    picks = ["kodim07.png", "kodim19.png", "kodim23.png"]
    fig, axes = plt.subplots(len(picks), 3, figsize=(11, 3.3 * len(picks)))

    with torch.no_grad():
        for row, name in enumerate(picks):
            p = pathlib.Path("data/kodak") / name
            img = Image.open(p).convert("RGB")
            x = to_t(img).unsqueeze(0).to(device)

            # ours
            xp, (h, w) = pad_mult(x)
            out = model(xp)
            xh = out["x_hat"][:, :, :h, :w].clamp(0, 1).cpu()
            ours_psnr = psnr(xh, x.cpu())
            ours_bpp = out["bpp"].item()

            # JPEG matched-ish (Q10 is typically comparable to our lowest)
            jpg_img, nbytes = jpeg_roundtrip(img, q=10)
            jpg_bpp = nbytes * 8 / (img.size[0] * img.size[1])
            jpg_psnr = psnr(to_t(jpg_img), to_t(img))

            axes[row, 0].imshow(img); axes[row, 0].set_title("Original")
            axes[row, 1].imshow(jpg_img)
            axes[row, 1].set_title(f"JPEG Q10\n{jpg_bpp:.2f} bpp  |  {jpg_psnr:.1f} dB")
            axes[row, 2].imshow(xh.squeeze(0).permute(1, 2, 0).numpy())
            axes[row, 2].set_title(f"Ours (q=3)\n{ours_bpp:.2f} bpp  |  {ours_psnr:.1f} dB")
            for ax in axes[row]: ax.axis("off")

    fig.tight_layout()
    pathlib.Path("results").mkdir(exist_ok=True)
    fig.savefig("results/demo_grid.png", dpi=140)
    print("saved results/demo_grid.png")


if __name__ == "__main__":
    main()
