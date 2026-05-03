"""
Demo: compress one image with our Scale Hyperprior model.

Usage:
    python demo.py --image data/kodak/kodim07.png --quality 3

Output:
    - prints bpp and PSNR
    - saves side-by-side comparison to results/demo_single.png
"""

import argparse
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


def pad_to_multiple(x, m=64):
    _, _, h, w = x.shape
    nh = (h + m - 1) // m * m
    nw = (w + m - 1) // m * m
    return F.pad(x, (0, nw - w, 0, nh - h), "replicate"), (h, w)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True, help="path to any PNG or JPG image")
    p.add_argument("--quality", type=int, default=3, choices=[1,2,3,4,5],
                   help="quality level 1 (lowest) to 5 (highest)")
    p.add_argument("--out", default="results/demo_single.png")
    args = p.parse_args()

    device = pick_device()
    print(f"device: {device}")

    # load model
    ckpt_path = f"checkpoints/compressai_q{args.quality}.pt"
    if not pathlib.Path(ckpt_path).exists():
        print(f"checkpoint {ckpt_path} not found, run load_pretrained.py first")
        return

    ckpt = torch.load(ckpt_path, map_location=device)
    model = ScaleHyperprior(N=128, M=192).to(device).eval()
    model.load_state_dict(ckpt["model"])
    print(f"loaded model quality={args.quality}")

    # load image
    to_tensor = transforms.ToTensor()
    img = Image.open(args.image).convert("RGB")
    x = to_tensor(img).unsqueeze(0).to(device)
    print(f"image size: {img.size[0]}×{img.size[1]}")

    # run model
    with torch.no_grad():
        x_padded, (h, w) = pad_to_multiple(x)
        out = model(x_padded)
        x_hat = out["x_hat"][:, :, :h, :w].clamp(0, 1)

    # metrics
    bpp  = out["bpp"].item()
    mse  = F.mse_loss(x_hat, x).item()
    psnr = 10 * torch.log10(torch.tensor(1.0 / mse)).item()

    print(f"\n--- Results ---")
    print(f"bpp:   {bpp:.4f}  bits per pixel")
    print(f"PSNR:  {psnr:.2f} dB")
    print(f"MSE:   {mse:.6f}")

    # save side by side
    orig_np  = x.squeeze(0).permute(1,2,0).cpu().numpy()
    recon_np = x_hat.squeeze(0).permute(1,2,0).cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(orig_np)
    axes[0].set_title("Original", fontsize=14)
    axes[0].axis("off")

    axes[1].imshow(recon_np)
    axes[1].set_title(
        f"Reconstructed  |  {bpp:.3f} bpp  |  PSNR {psnr:.2f} dB",
        fontsize=14
    )
    axes[1].axis("off")

    fig.tight_layout()
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"\nsaved comparison to {args.out}")


if __name__ == "__main__":
    main()
