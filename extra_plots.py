"""
Three additional plots for the viva:

  1) results/per_image_savings.png
     Per-image bpp savings vs JPEG at matched PSNR.

  2) results/latent_viz.png
     Reproduce paper Figure 2 for one Kodak image:
       |y| channel sample  |  sigma_hat  |  y/sigma_hat

  3) results/error_heatmap.png
     |x - x_hat| per-pixel heatmap for one Kodak image at q=3.

Uses existing CSVs and checkpoints; no retraining.
"""

import csv
import pathlib
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from models import ScaleHyperprior


RES = pathlib.Path("results")
KODAK = pathlib.Path("data/kodak")


def pick_device():
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"


def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def pad_mult(x, m=64):
    _, _, h, w = x.shape
    nh, nw = (h + m - 1) // m * m, (w + m - 1) // m * m
    return F.pad(x, (0, nw - w, 0, nh - h), "replicate"), (h, w)


# ---------- plot 1: per-image bpp savings vs JPEG at matched PSNR ----------
def plot_savings():
    # average JPEG curve per quality across images
    jpeg = load_csv(RES / "jpeg.csv")
    ours = []
    for q in range(1, 6):
        ours.extend([(q, r) for r in load_csv(RES / f"ours_q{q}.csv")])

    # for each Kodak image, pick our q=3 row, then interpolate JPEG bpp at the
    # same PSNR for that image
    # build per-image JPEG (quality -> (bpp, psnr))
    jpeg_by_img = defaultdict(list)
    for r in jpeg:
        jpeg_by_img[r["image"]].append(
            (float(r["quality"]), float(r["bpp"]), float(r["psnr"])))

    savings = []
    for q_idx, r in ours:
        if q_idx != 3: continue
        name = r["image"]
        ours_psnr = float(r["psnr"])
        ours_bpp = float(r["bpp"])
        jrows = sorted(jpeg_by_img[name], key=lambda t: t[2])  # by PSNR
        ps = [t[2] for t in jrows]
        bs = [t[1] for t in jrows]
        # need JPEG curve to bracket our PSNR; otherwise extrapolate linearly
        jpeg_bpp_at_same_psnr = float(np.interp(ours_psnr, ps, bs,
                                                left=bs[0], right=bs[-1]))
        savings.append({
            "image": name,
            "ours_bpp": ours_bpp,
            "jpeg_bpp": jpeg_bpp_at_same_psnr,
            "pct_saved": 100 * (1 - ours_bpp / jpeg_bpp_at_same_psnr),
        })

    savings.sort(key=lambda d: d["pct_saved"])
    labels = [s["image"].replace("kodim", "").replace(".png", "") for s in savings]
    vals = [s["pct_saved"] for s in savings]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    colors = ["#d62728" if v < 0 else "#2ca02c" for v in vals]
    ax.bar(labels, vals, color=colors)
    ax.axhline(0, color="k", linewidth=0.8)
    mean = np.mean(vals)
    ax.axhline(mean, color="blue", linestyle="--", linewidth=1,
               label=f"mean: {mean:.1f}%")
    ax.set_xlabel("Kodak image number")
    ax.set_ylabel("Bitrate saved vs JPEG at same PSNR (%)")
    ax.set_title("Per-image bitrate savings over JPEG (Ours q=3)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(RES / "per_image_savings.png", dpi=150)
    print(f"saved {RES/'per_image_savings.png'}  (mean savings: {mean:.1f}%)")


# ---------- plot 2: latent visualization (paper Fig 2) ----------
def plot_latents():
    device = pick_device()
    ckpt = torch.load("checkpoints/compressai_q3.pt", map_location=device)
    model = ScaleHyperprior(N=128, M=192).to(device).eval()
    model.load_state_dict(ckpt["model"])

    img = Image.open(KODAK / "kodim19.png").convert("RGB")
    x = transforms.ToTensor()(img).unsqueeze(0).to(device)
    x_padded, _ = pad_mult(x)

    with torch.no_grad():
        y = model.g_a(x_padded)
        z = model.h_a(y)
        z_hat, _ = model.entropy_bottleneck(z)
        sigma = model.h_s(z_hat)

    # pick a channel with strong structure (highest variance)
    var = y[0].var(dim=(1, 2))
    c = int(var.argmax().item())
    y_c = y[0, c].cpu().numpy()
    s_c = sigma[0, c].cpu().numpy().clip(min=1e-3)
    y_over_s = y_c / s_c

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
    axes[0].imshow(img)
    axes[0].set_title("Input image")
    axes[1].imshow(np.abs(y_c), cmap="gray")
    axes[1].set_title(f"|y| channel {c}\n(structured)")
    axes[2].imshow(s_c, cmap="gray")
    axes[2].set_title(r"$\hat{\sigma}$ from hyperprior")
    axes[3].imshow(y_over_s, cmap="gray")
    axes[3].set_title(r"$y / \hat{\sigma}$  (whitened)")
    for ax in axes: ax.axis("off")
    fig.suptitle("Reproduction of paper Figure 2: hyperprior captures spatial scale structure",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(RES / "latent_viz.png", dpi=150)
    print(f"saved {RES/'latent_viz.png'}")


# ---------- plot 3: reconstruction error heatmap ----------
def plot_error_heatmap():
    device = pick_device()
    ckpt = torch.load("checkpoints/compressai_q3.pt", map_location=device)
    model = ScaleHyperprior(N=128, M=192).to(device).eval()
    model.load_state_dict(ckpt["model"])

    img = Image.open(KODAK / "kodim07.png").convert("RGB")
    x = transforms.ToTensor()(img).unsqueeze(0).to(device)
    xp, (h, w) = pad_mult(x)
    with torch.no_grad():
        out = model(xp)
    x_hat = out["x_hat"][:, :, :h, :w].clamp(0, 1).cpu()

    err = (x.cpu() - x_hat).abs().squeeze(0).mean(dim=0).numpy()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img); axes[0].set_title("Original")
    axes[1].imshow(x_hat.squeeze(0).permute(1, 2, 0).numpy())
    axes[1].set_title(f"Reconstruction (bpp={out['bpp'].item():.2f})")
    im = axes[2].imshow(err, cmap="hot", vmin=0, vmax=err.max())
    axes[2].set_title("Per-pixel absolute error")
    for ax in axes: ax.axis("off")
    fig.colorbar(im, ax=axes[2], fraction=0.04)
    fig.tight_layout()
    fig.savefig(RES / "error_heatmap.png", dpi=150)
    print(f"saved {RES/'error_heatmap.png'}  (max err: {err.max():.3f})")


def main():
    RES.mkdir(exist_ok=True)
    plot_savings()
    plot_latents()
    plot_error_heatmap()


if __name__ == "__main__":
    main()
