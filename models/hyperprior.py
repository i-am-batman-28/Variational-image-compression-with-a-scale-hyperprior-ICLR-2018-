"""
Hyper-analysis h_a and hyper-synthesis h_s, Figure 4 right side.

h_a:  y      [B, M, h, w]   ->   z      [B, N, h/4, w/4]
h_s:  z_hat  [B, N, h/4, w/4] -> sigma  [B, M, h, w]

h_a: |y| -> 3x3 conv stride 1 -> ReLU -> 5x5 conv stride 2 -> ReLU -> 5x5 conv stride 2
  (Paper Figure 4; abs applied to y first so the hyper-encoder sees magnitudes.)
h_s: mirror with transposed convs, ends with ReLU to keep sigma >= 0.
"""

import torch
import torch.nn as nn


class HyperAnalysis(nn.Module):
    def __init__(self, N: int = 128, M: int = 192):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(M, N, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(N, N, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(N, N, kernel_size=5, stride=2, padding=2),
        )

    def forward(self, y):
        return self.net(torch.abs(y))


class HyperSynthesis(nn.Module):
    def __init__(self, N: int = 128, M: int = 192):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(N, N, kernel_size=5, stride=2,
                               padding=2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(N, N, kernel_size=5, stride=2,
                               padding=2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(N, M, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),  # sigma must be non-negative
        )

    def forward(self, z_hat):
        return self.net(z_hat)
