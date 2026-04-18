"""
Analysis transform g_a (encoder), Figure 4 of Balle et al. 2018.

Input:  x  [B, 3, H, W]        (RGB image)
Output: y  [B, M, H/16, W/16]  (latent)

Four conv layers, each stride 2 (total downsample = 16). GDN between them;
no GDN after the last conv (it feeds the hyper-encoder and quantizer).
"""

import torch.nn as nn
from .gdn import GDN


class AnalysisTransform(nn.Module):
    def __init__(self, N: int = 128, M: int = 192):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, N, kernel_size=5, stride=2, padding=2),
            GDN(N),
            nn.Conv2d(N, N, kernel_size=5, stride=2, padding=2),
            GDN(N),
            nn.Conv2d(N, N, kernel_size=5, stride=2, padding=2),
            GDN(N),
            nn.Conv2d(N, M, kernel_size=5, stride=2, padding=2),
        )

    def forward(self, x):
        return self.net(x)
