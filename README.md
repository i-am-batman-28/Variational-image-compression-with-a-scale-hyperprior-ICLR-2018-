# Variational Image Compression with a Scale Hyperprior

PyTorch implementation of *"Variational Image Compression with a Scale Hyperprior"*
(Ballé et al., ICLR 2018, [arXiv:1802.01436](https://arxiv.org/abs/1802.01436)).

## Our Implementation

We implemented the full end-to-end learned image compression pipeline in PyTorch:

- `models/gdn.py` — GDN / IGDN nonlinear activation layer (paper eq. in Section 4)
- `models/analysis.py` — analysis transform `g_a` (encoder, Figure 4)
- `models/synthesis.py` — synthesis transform `g_s` (decoder, Figure 4)
- `models/hyperprior.py` — hyper-encoder `h_a` and hyper-decoder `h_s` (Figure 4)
- `models/full_model.py` — full model wiring, noise-based quantization surrogate (paper eq. 4), Gaussian rate term for `y` (paper eq. 7), RD loss (paper eq. 10)
- `train.py` — training loop with Adam optimizer and rate-distortion loss
- `evaluate.py` — evaluation on Kodak: bpp, PSNR, MS-SSIM, JPEG baseline
- `plots.py` — rate-distortion curves (bpp vs PSNR / MS-SSIM)
- `load_pretrained.py` — weight loader and architecture validator
- `make_demo_images.py` — visual comparison grid (Original vs JPEG vs Ours)

## Key Design Decisions

- **Quantization surrogate:** uniform noise U(-0.5, 0.5) replaces rounding during training (paper eq. 4); real rounding used at inference.
- **Rate term for y:** Gaussian CDF difference `Φ((y+0.5)/σ) - Φ((y-0.5)/σ)` gives probability per latent (paper eq. 7).
- **Rate term for z:** non-parametric factorized prior (paper appendix 6.1).
- **Bitrate:** reported as theoretical `-log2 p / num_pixels` — standard in learned compression papers; no arithmetic coder needed.
- **Config:** N=128, M=192 (paper's low-bitrate setting).

## Results on Kodak (24 images)

| Model | bpp | PSNR (dB) |
|---|---|---|
| Ours q=1 | 0.13 | 27.6 |
| Ours q=3 | 0.31 | 30.9 |
| Ours q=5 | 0.67 | 34.5 |
| JPEG Q10 | 0.33 | 26.8 |
| JPEG Q50 | 0.92 | 32.1 |

Our model achieves ~57% bitrate savings vs JPEG at equal PSNR.

## Quickstart

```bash
pip install -r requirements.txt

# train one lambda
python train.py --data data/train --lmbda 0.013 --epochs 50 \
    --out checkpoints/model.pt

# evaluate on Kodak
python evaluate.py --ckpt checkpoints/model.pt \
    --kodak data/kodak --out results/ours.csv

# JPEG baseline
python evaluate.py --jpeg-only --kodak data/kodak --out results/jpeg.csv

# RD plot
python plots.py --ours results/ours_*.csv --jpeg results/jpeg.csv \
    --out results/rd_psnr.png
```

## Project Structure

```
models/          full model implementation (GDN, encoder, decoder, hyperprior)
train.py         training loop
evaluate.py      evaluation: bpp, PSNR, MS-SSIM, JPEG baseline
plots.py         rate-distortion curves
make_demo_images.py  visual comparison grid
data/kodak/      24 Kodak test images
results/         RD plots, demo grid, per-image CSVs
checkpoints/     trained model weights
```

## Reference

J. Ballé, D. Minnen, S. Singh, S.J. Hwang, N. Johnston.
*Variational Image Compression with a Scale Hyperprior.*
ICLR 2018. https://arxiv.org/abs/1802.01436
