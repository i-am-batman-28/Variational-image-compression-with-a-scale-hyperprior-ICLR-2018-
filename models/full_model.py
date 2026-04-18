"""
Scale Hyperprior model (Balle et al., ICLR 2018).

Data flow (training):
    x --g_a--> y --(+uniform noise)--> y_tilde
    y --h_a--> z --(+uniform noise)--> z_tilde
    z_tilde --h_s--> sigma
    Rate_y = -log2 p(y_tilde | sigma)        (Gaussian, written here)
    Rate_z = -log2 p(z_tilde)                (factorized prior; EntropyBottleneck from CompressAI, cited)
    x_hat = g_s(y_tilde)
    Loss  = lambda * MSE(x, x_hat) + bpp_y + bpp_z

Test-time swaps uniform noise for rounding. We don't run a real range coder;
reported bpp is the theoretical -log2 p / num_pixels, standard in the paper.
"""

import torch
import torch.nn as nn

from .analysis import AnalysisTransform
from .synthesis import SynthesisTransform
from .hyperprior import HyperAnalysis, HyperSynthesis

# Reused from CompressAI for the non-parametric factorized prior on z.
# See references/CompressAI/compressai/entropy_models/entropy_models.py.
from compressai.entropy_models import EntropyBottleneck


class ScaleHyperprior(nn.Module):
    def __init__(self, N: int = 128, M: int = 192):
        super().__init__()
        self.N, self.M = N, M
        self.g_a = AnalysisTransform(N, M)
        self.g_s = SynthesisTransform(N, M)
        self.h_a = HyperAnalysis(N, M)
        self.h_s = HyperSynthesis(N, M)
        self.entropy_bottleneck = EntropyBottleneck(N)  # prior on z
        self.sigma_lower_bound = 0.11  # numerical floor for sigma (CompressAI uses this)

    # ---- quantization surrogate --------------------------------------------
    @staticmethod
    def _add_noise(x):
        # eq. 4: uniform(-0.5, 0.5) as a differentiable stand-in for rounding
        return x + torch.empty_like(x).uniform_(-0.5, 0.5)

    @staticmethod
    def _round_ste(x):
        # straight-through rounding for inference-style forward during training eval
        return (torch.round(x) - x).detach() + x

    # ---- Gaussian rate term for y ------------------------------------------
    def _gaussian_bits(self, y_tilde, sigma):
        """
        p(y_tilde_i | sigma_i) = N(0, sigma_i^2) convolved with U(-0.5, 0.5)
                              = Phi((y_tilde+0.5)/sigma) - Phi((y_tilde-0.5)/sigma)
        Returns total bits summed over all elements.
        """
        sigma = sigma.clamp(min=self.sigma_lower_bound)
        normal = torch.distributions.Normal(0.0, 1.0)
        upper = normal.cdf((y_tilde + 0.5) / sigma)
        lower = normal.cdf((y_tilde - 0.5) / sigma)
        likelihood = (upper - lower).clamp(min=1e-9)
        return -torch.log2(likelihood).sum()

    # ---- forward -----------------------------------------------------------
    def forward(self, x):
        y = self.g_a(x)
        z = self.h_a(y)

        # z path: EntropyBottleneck adds uniform noise in train mode, rounds in eval mode
        z_tilde, z_likelihood = self.entropy_bottleneck(z)

        # y path: noise in training (paper eq. 4), rounding at test time
        sigma = self.h_s(z_tilde)
        if self.training:
            y_tilde = self._add_noise(y)
        else:
            y_tilde = torch.round(y)
        x_hat = self.g_s(y_tilde)

        num_pixels = x.size(0) * x.size(2) * x.size(3)
        bpp_y = self._gaussian_bits(y_tilde, sigma) / num_pixels
        bpp_z = (-torch.log2(z_likelihood.clamp(min=1e-9))).sum() / num_pixels

        return {
            "x_hat": x_hat,
            "bpp_y": bpp_y,
            "bpp_z": bpp_z,
            "bpp": bpp_y + bpp_z,
        }
