import torch
import torch.nn as nn
import torch.nn.functional as F

from compressai.layers import GDN, MaskedConv2d
from compressai.models.utils import conv, deconv

class HyperpriorAnalysisModel(nn.Module):
    def __init__(self, N, M, in_channels=3, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.N = N
        self.M = M

        self.model = nn.Sequential(
            conv(in_channels, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, M),
        )

    def forward(self, x):
        return self.model(x)


class HyperpriorSynthesisModel(nn.Module):
    def __init__(self, N, M, out_channels=3, **kwargs):
        super().__init__()
        self.out_channels = out_channels
        self.N = N
        self.M = M

        self.model =  nn.Sequential(
            deconv(M, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, out_channels),
        )

    def forward(self, x):
        return self.model(x)


class HyperpriorHyperAnalysisModel(nn.Module):
    def __init__(self, N, M, **kwargs):
        super().__init__()
        self.N = N
        self.M = M

        self.model = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
            conv(N, N),
            nn.ReLU(inplace=True),
            conv(N, N),
        )

    def forward(self, x):
        return self.model(x)


class HyperpriorHyperSynthesisModel(nn.Module):
    def __init__(self, N, M, **kwargs):
        super().__init__()
        self.N = N
        self.M = M

        self.model = nn.Sequential(
            deconv(N, N),
            nn.ReLU(inplace=True),
            deconv(N, N),
            nn.ReLU(inplace=True),
            conv(N, M, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.model(x)
