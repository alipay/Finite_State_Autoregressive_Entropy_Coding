# from compressai
from typing import Any

import torch
import torch.nn as nn

__all__ = [
    "MaskedConv2d", "MaskedConv3d"
]

class MaskedConv2d(nn.Conv2d):
    r"""Masked 2D convolution implementation, mask future "unseen" pixels.
    Useful for building auto-regressive network components.

    Introduced in `"Conditional Image Generation with PixelCNN Decoders"
    <https://arxiv.org/abs/1606.05328>`_.

    Inherits the same arguments as a `nn.Conv2d`. Use `mask_type='A'` for the
    first layer (which also masks the "current pixel"), `mask_type='B'` for the
    following layers.
    """

    def __init__(self, *args: Any, mask_type: str = "A", **kwargs: Any):
        super().__init__(*args, **kwargs)

        if mask_type not in ("A", "B", "Checkerboard"):
            raise ValueError(f'Invalid "mask_type" value "{mask_type}"')

        self.register_buffer("mask", torch.ones_like(self.weight.data))
        _, _, h, w = self.mask.size()
        if mask_type in ("A", "B"):
            self.mask[:, :, h // 2, w // 2 + (mask_type == "B") :] = 0
            self.mask[:, :, h // 2 + 1 :] = 0
        else:
            # checkerboard
            for i in range(h):
                for j in range(w):
                    if (i+j) % 2 == 0:
                        self.mask[:, :, i, j] = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO(begaintj): weight assigment is not supported by torchscript
        self.weight.data *= self.mask
        return super().forward(x)


class MaskedConv3d(nn.Conv3d):
    def __init__(self, *args: Any, mask_type: str = "A", **kwargs: Any):
        super().__init__(*args, **kwargs)

        if mask_type not in ("A", "B"):
            raise ValueError(f'Invalid "mask_type" value "{mask_type}"')

        self.register_buffer("mask", torch.ones_like(self.weight.data))
        _, _, c, h, w = self.mask.size()
        self.mask[:, :, c // 2, h // 2, w // 2 + (mask_type == "B") :] = 0
        self.mask[:, :, c // 2, h // 2 + 1 :] = 0
        self.mask[:, :, c // 2 + 1 :] = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO(begaintj): weight assigment is not supported by torchscript
        self.weight.data *= self.mask
        return super().forward(x)
