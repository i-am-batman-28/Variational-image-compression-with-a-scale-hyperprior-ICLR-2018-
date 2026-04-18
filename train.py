"""
Training script for the Scale Hyperprior model.

Usage:
    python train.py --data data/train --epochs 50 --lambda 0.013 --out checkpoints/lambda0013.pt

Notes:
- Paper: 256x256 crops, batch 8, Adam lr=1e-4.
- On M1 Mac this is slow; intended for small-scale sanity runs.
  For real training use Colab/a GPU with the same script.
- Two optimizers: main (model params) and aux (EntropyBottleneck quantiles).
"""

import argparse
import pathlib
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

from models import ScaleHyperprior


class ImageFolder(Dataset):
    def __init__(self, root, patch_size=256):
        self.paths = sorted(
            p for p in pathlib.Path(root).rglob("*")
            if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )
        if not self.paths:
            raise FileNotFoundError(f"No images found under {root}")
        self.tf = transforms.Compose([
            transforms.RandomCrop(patch_size, pad_if_needed=True),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.tf(img)


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train(args):
    device = pick_device()
    print(f"device: {device}")

    ds = ImageFolder(args.data, patch_size=args.patch_size)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.workers, drop_last=True)

    model = ScaleHyperprior(N=args.N, M=args.M).to(device)

    # Main parameters vs entropy-bottleneck quantile parameters (paper appendix 6.1).
    aux_params = [p for n, p in model.named_parameters() if "quantiles" in n]
    main_params = [p for n, p in model.named_parameters() if "quantiles" not in n]
    opt = torch.optim.Adam(main_params, lr=args.lr)
    aux_opt = torch.optim.Adam(aux_params, lr=1e-3)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    step = 0
    for epoch in range(args.epochs):
        t0 = time.time()
        for x in dl:
            x = x.to(device)
            out_dict = model(x)
            mse = F.mse_loss(out_dict["x_hat"], x)
            bpp = out_dict["bpp"]
            loss = args.lmbda * 255.0 ** 2 * mse + bpp

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(main_params, 1.0)
            opt.step()

            aux_loss = model.entropy_bottleneck.loss()
            aux_opt.zero_grad()
            aux_loss.backward()
            aux_opt.step()

            if step % args.log_every == 0:
                psnr = 10 * torch.log10(1.0 / mse.clamp(min=1e-10))
                print(f"ep {epoch} step {step}  loss {loss.item():.3f}  "
                      f"bpp {bpp.item():.3f}  PSNR {psnr.item():.2f}  "
                      f"aux {aux_loss.item():.2f}")
            step += 1

        print(f"epoch {epoch} took {time.time()-t0:.1f}s, saving checkpoint")
        torch.save({"model": model.state_dict(), "args": vars(args)}, out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="folder of training images")
    p.add_argument("--out", default="checkpoints/model.pt")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--patch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lmbda", type=float, default=0.013,
                   help="RD trade-off; paper MSE set: 0.0018..0.0483")
    p.add_argument("--N", type=int, default=128)
    p.add_argument("--M", type=int, default=192)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--log-every", type=int, default=20)
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
