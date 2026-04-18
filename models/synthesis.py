"""
Synthesis transform g_s (decoder), mirror of g_a.

Input:  y_hat  [B, M, H/16, W/16]
Output: x_hat  [B, 3, H, W]

Four transposed convs, each upsample 2x. IGDN between them; no IGDN after last.
"""

import torch.nn as nn
from .gdn import IGDN


class SynthesisTransform(nn.Module):
    def __init__(self, N: int = 128, M: int = 192):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(M, N, kernel_size=5, stride=2,
                               padding=2, output_padding=1),
            IGDN(N),
            nn.ConvTranspose2d(N, N, kernel_size=5, stride=2,
                               padding=2, output_padding=1),
            IGDN(N),
            nn.ConvTranspose2d(N, N, kernel_size=5, stride=2,
                               padding=2, output_padding=1),
            IGDN(N),
            nn.ConvTranspose2d(N, 3, kernel_size=5, stride=2,
                               padding=2, output_padding=1),
        )

    def forward(self, y_hat):
        return self.net(y_hat)
