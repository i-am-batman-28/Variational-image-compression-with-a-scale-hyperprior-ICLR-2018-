"""
Load CompressAI's pretrained bmshj2018_hyperprior weights into OUR model.

Why this is non-trivial:
  - CompressAI's GDN reparameterizes beta/gamma: stored = sqrt(effective + pedestal).
    Our GDN stores the effective values directly, so we convert:
        our_beta  = stored_beta^2  - pedestal        (clamped to beta_min)
        our_gamma = stored_gamma^2 - pedestal        (clamped to 0)
  - Both use the same conv layout for g_a, g_s, h_a, h_s, so conv weights/biases map 1:1.
  - EntropyBottleneck module is reused unchanged.

Usage:
    python load_pretrained.py --quality 3 --out checkpoints/compressai_q3.pt
"""

import argparse
import pathlib

import torch
from compressai.zoo import bmshj2018_hyperprior

from models import ScaleHyperprior


# Map: index of GDN layer in our Sequential -> index of stored GDN in theirs.
# g_a: conv, GDN, conv, GDN, conv, GDN, conv    (GDNs at 1, 3, 5)
# g_s: tconv, IGDN, tconv, IGDN, tconv, IGDN, tconv (IGDNs at 1, 3, 5)
GDN_INDICES = (1, 3, 5)
CONV_INDICES_GA = (0, 2, 4, 6)
CONV_INDICES_GS = (0, 2, 4, 6)  # transposed convs
HA_CONV_INDICES = (0, 2, 4)
HS_CONV_INDICES = (0, 2, 4)     # last one in our h_s Sequential is Conv at idx 4


def convert_gdn(stored_beta, stored_gamma, pedestal_beta, pedestal_gamma,
                beta_min=1e-6):
    """CompressAI stored sqrt(eff + pedestal)  ->  our effective values."""
    eff_beta = (stored_beta ** 2 - pedestal_beta).clamp(min=beta_min)
    eff_gamma = (stored_gamma ** 2 - pedestal_gamma).clamp(min=0.0)
    return eff_beta, eff_gamma


def copy_gdn(src_gdn, dst_gdn):
    peb = src_gdn.beta_reparam.pedestal.item()
    peg = src_gdn.gamma_reparam.pedestal.item()
    eb, eg = convert_gdn(src_gdn.beta.data, src_gdn.gamma.data, peb, peg,
                         beta_min=dst_gdn.beta_min)
    dst_gdn.beta.data.copy_(eb)
    dst_gdn.gamma.data.copy_(eg)


def copy_conv(src, dst):
    dst.weight.data.copy_(src.weight.data)
    if src.bias is not None and dst.bias is not None:
        dst.bias.data.copy_(src.bias.data)


def transfer(src_model, dst_model: ScaleHyperprior):
    # g_a
    for i_src, i_dst in zip(CONV_INDICES_GA, range(0, 8, 2)):
        copy_conv(src_model.g_a[i_src], dst_model.g_a.net[i_dst])
    for i in GDN_INDICES:
        copy_gdn(src_model.g_a[i], dst_model.g_a.net[i])

    # g_s
    for i_src, i_dst in zip(CONV_INDICES_GS, range(0, 8, 2)):
        copy_conv(src_model.g_s[i_src], dst_model.g_s.net[i_dst])
    for i in GDN_INDICES:
        copy_gdn(src_model.g_s[i], dst_model.g_s.net[i])

    # h_a
    for i in HA_CONV_INDICES:
        copy_conv(src_model.h_a[i], dst_model.h_a.net[i])

    # h_s: their layout = [Tconv, ReLU, Tconv, ReLU, Conv]
    #      ours         = [Tconv, ReLU, Tconv, ReLU, Conv, ReLU]
    for i in HS_CONV_INDICES:
        copy_conv(src_model.h_s[i], dst_model.h_s.net[i])

    # entropy bottleneck: skip coder-only buffers (_offset, _quantized_cdf,
    # _cdf_length); these are only populated after .update() and differ in shape
    # because our model hasn't called update(). They're not needed for forward pass.
    src_eb = src_model.entropy_bottleneck.state_dict()
    src_eb = {k: v for k, v in src_eb.items()
              if k not in ("_offset", "_quantized_cdf", "_cdf_length")}
    dst_model.entropy_bottleneck.load_state_dict(src_eb, strict=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quality", type=int, default=3,
                   help="CompressAI quality 1..8; 1-5 use N=128 M=192")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    if args.quality <= 5:
        N, M = 128, 192
    else:
        N, M = 192, 320

    src = bmshj2018_hyperprior(quality=args.quality, pretrained=True)
    src.eval()
    dst = ScaleHyperprior(N=N, M=M)
    dst.eval()

    transfer(src, dst)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": dst.state_dict(),
                "args": {"N": N, "M": M, "quality": args.quality}},
               out)
    print(f"saved {out}")

    # quick numerical check: outputs on a random input should be very close
    with torch.no_grad():
        x = torch.rand(1, 3, 256, 256)
        out_src = src(x)
        out_dst = dst(x)
        diff = (out_src["x_hat"] - out_dst["x_hat"]).abs().max().item()
        print(f"max |x_hat_src - x_hat_ours| on random input: {diff:.6f}")


if __name__ == "__main__":
    main()
