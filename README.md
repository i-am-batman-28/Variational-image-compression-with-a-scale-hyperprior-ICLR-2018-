# Scale Hyperprior — MS Project Implementation

PyTorch implementation of *"Variational Image Compression with a Scale Hyperprior"*
(Ballé et al., ICLR 2018, [arXiv:1802.01436](https://arxiv.org/abs/1802.01436)).

## What is ours vs. reused

**Written by us (this repo):**
- `models/gdn.py` — GDN / IGDN layer
- `models/analysis.py`, `models/synthesis.py` — encoder/decoder (`g_a`, `g_s`)
- `models/hyperprior.py` — hyper-encoder/decoder (`h_a`, `h_s`)
- `models/full_model.py` — wiring, noise-quantization surrogate, Gaussian rate term for `y`
- `train.py`, `evaluate.py`, `plots.py` — training loop, eval on Kodak, JPEG baseline, RD plots

**Reused from [CompressAI](https://github.com/InterDigitalInc/CompressAI) (cited):**
- `compressai.entropy_models.EntropyBottleneck` — factorized prior on `z` (paper appendix 6.1).
  Reimplementing a non-parametric density model was out of scope; we import it and cite.

## Scope / simplifications
- Rate is reported as theoretical `-log2 p / num_pixels` (no arithmetic coder). Standard in RD papers.
- MSE-optimized variant only (paper also trains an MS-SSIM variant; skipped).
- N=128, M=192 (paper's low-bitrate config).

## Quickstart
```bash
pip install -r requirements.txt

# 1. Drop Kodak PNGs into data/kodak (24 images)
# 2. Drop training images (e.g., DIV2K) into data/train

# train one lambda
python train.py --data data/train --lmbda 0.013 --epochs 50 \
    --out checkpoints/lambda0013.pt

# evaluate
python evaluate.py --ckpt checkpoints/lambda0013.pt \
    --kodak data/kodak --out results/ours_lambda0013.csv

# JPEG baseline
python evaluate.py --jpeg-only --kodak data/kodak --out results/jpeg.csv

# plot
python plots.py --ours results/ours_*.csv --jpeg results/jpeg.csv \
    --out results/rd_psnr.png
```

## Files
```
models/    model code (ours)
train.py   training loop (ours)
evaluate.py  eval on Kodak + JPEG baseline (ours)
plots.py   RD curves (ours)
references/CompressAI   reference implementation, for verification only
```
# Variational-image-compression-with-a-scale-hyperprior-ICLR-2018-
