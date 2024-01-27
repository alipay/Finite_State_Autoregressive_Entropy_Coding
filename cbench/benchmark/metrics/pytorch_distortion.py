import math

import torch
import torch.nn as nn

import numpy as np

from pytorch_msssim import ms_ssim

from .base import BaseMetric

def _compute_psnr(a, b, max_val: float = 1.0) -> float:
    mse = torch.mean((a - b) ** 2).item()
    psnr = 20 * np.log10(max_val) - 10 * np.log10(mse)
    return psnr

def _compute_ms_ssim(a, b, max_val: float = 1.0) -> float:
    return ms_ssim(a, b, data_range=max_val).item()

class PytorchBatchedDistortion(BaseMetric):
    def __init__(self, metric_name="psnr", max_val=1.0):
        super().__init__()
        self.metric_name = metric_name
        self.max_val = max_val
        if metric_name == "psnr":
            self.metric = _compute_psnr
        elif metric_name == "ms-ssim":
            self.metric = _compute_ms_ssim
        else:
            raise NotImplementedError(f"{metric_name} is not implemented!")

    def __call__(self, output, target):
        output = output.type_as(target)
        return {
            self.metric_name : self.metric(output, target, max_val=self.max_val)
        }

