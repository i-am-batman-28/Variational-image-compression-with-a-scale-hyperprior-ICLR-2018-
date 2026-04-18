"""
Rate-distortion curves: ours vs. JPEG.

Usage:
  python plots.py --ours results/ours_lambda0013.csv results/ours_lambda003.csv \
                  --jpeg results/jpeg.csv --out results/rd_psnr.png
"""

import argparse
import csv
import pathlib
from collections import defaultdict

import matplotlib.pyplot as plt


def read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def avg_by(rows, key):
    # group by `key`, average bpp/psnr/ms_ssim across images
    buckets = defaultdict(list)
    for r in rows:
        buckets[r[key]].append(r)
    out = []
    for k, bucket in buckets.items():
        out.append({
            "group": k,
            "bpp": sum(float(r["bpp"]) for r in bucket) / len(bucket),
            "psnr": sum(float(r["psnr"]) for r in bucket) / len(bucket),
            "ms_ssim": sum(float(r["ms_ssim"]) for r in bucket) / len(bucket),
        })
    out.sort(key=lambda r: r["bpp"])
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ours", nargs="*", default=[],
                   help="one CSV per trained lambda")
    p.add_argument("--jpeg", default=None)
    p.add_argument("--out", default="results/rd_psnr.png")
    p.add_argument("--metric", choices=["psnr", "ms_ssim"], default="psnr")
    args = p.parse_args()

    fig, ax = plt.subplots(figsize=(6, 4))

    # ours: one point per checkpoint (CSV)
    ours_points = []
    for path in args.ours:
        rows = read(path)
        bpp = sum(float(r["bpp"]) for r in rows) / len(rows)
        m = sum(float(r[args.metric]) for r in rows) / len(rows)
        ours_points.append((bpp, m, pathlib.Path(path).stem))
    ours_points.sort()
    if ours_points:
        ax.plot([p[0] for p in ours_points], [p[1] for p in ours_points],
                "o-", label="Ours (Scale Hyperprior)")

    # JPEG: group by quality
    if args.jpeg:
        jrows = read(args.jpeg)
        grouped = avg_by(jrows, "quality")
        ax.plot([g["bpp"] for g in grouped],
                [g[args.metric] for g in grouped],
                "s--", label="JPEG")

    ax.set_xlabel("bits per pixel (bpp)")
    ax.set_ylabel("PSNR (dB)" if args.metric == "psnr" else "MS-SSIM")
    ax.set_title(f"Rate-Distortion on Kodak ({args.metric.upper()})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
