"""
Forward-pass sanity check: random 256x256 RGB -> model -> check shapes and rate/MSE.
If this runs cleanly the wiring is correct.
"""

import torch
import torch.nn.functional as F
from models import ScaleHyperprior


def main():
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"device: {device}")

    model = ScaleHyperprior(N=128, M=192).to(device)
    model.train()

    x = torch.rand(2, 3, 256, 256, device=device)
    out = model(x)

    print(f"x:       {tuple(x.shape)}")
    print(f"x_hat:   {tuple(out['x_hat'].shape)}")
    print(f"bpp_y:   {out['bpp_y'].item():.3f}")
    print(f"bpp_z:   {out['bpp_z'].item():.3f}")
    print(f"bpp:     {out['bpp'].item():.3f}")

    mse = F.mse_loss(out["x_hat"], x)
    loss = 0.013 * 255**2 * mse + out["bpp"]
    loss.backward()
    print(f"mse:     {mse.item():.4f}")
    print(f"loss:    {loss.item():.3f}")
    print("backward() OK")

    assert out["x_hat"].shape == x.shape, "shape mismatch"
    print("\nSANITY CHECK PASSED")


if __name__ == "__main__":
    main()
