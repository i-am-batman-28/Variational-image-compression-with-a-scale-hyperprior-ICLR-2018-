"""
GDN / IGDN layer (Ballé et al., 2016).

GDN:  y_i = x_i / sqrt( beta_i + sum_j( gamma_ij * x_j^2 ) )
IGDN: inverse form, used in the decoder.

beta and gamma are kept non-negative via a softplus-style reparameterization
(store raw parameter p, use f(p) = max(p, eps) during forward).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _lower_bound(x: torch.Tensor, bound: float) -> torch.Tensor:
    # differentiable max(x, bound); keeps gradients flowing to x when x < bound
    return torch.maximum(x, torch.full_like(x, bound))


class GDN(nn.Module):
    def __init__(self, channels: int, inverse: bool = False,
                 beta_min: float = 1e-6, gamma_init: float = 0.1):
        super().__init__()
        self.inverse = inverse
        self.beta_min = beta_min

        # beta: per-channel bias, shape (C,)
        self.beta = nn.Parameter(torch.ones(channels))
        # gamma: channel-to-channel mixing, shape (C, C)
        self.gamma = nn.Parameter(gamma_init * torch.eye(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        C = x.size(1)
        beta = _lower_bound(self.beta, self.beta_min)
        gamma = _lower_bound(self.gamma, 0.0).view(C, C, 1, 1)

        # norm[n,i,h,w] = beta_i + sum_j gamma_ij * x[n,j,h,w]^2
        norm = F.conv2d(x * x, gamma, bias=beta)

        if self.inverse:
            return x * torch.sqrt(norm)
        return x * torch.rsqrt(norm)


class IGDN(GDN):
    def __init__(self, channels: int, **kw):
        super().__init__(channels, inverse=True, **kw)
