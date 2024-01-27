import itertools
import os
from typing import Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

import math
import numpy as np

from cbench.nn.base import NNTrainableModule, PLNNTrainableModule
from cbench.nn.models.vae import VAEEncoder, VAEDecoder
from cbench.nn.models.vqvae_model_v2 import Encoder, Decoder
from cbench.nn.trainer import make_optimizer, make_scheduler
from cbench.nn.utils import batched_cross_entropy

from .base import Preprocessor
from ..base import TrainableModuleInterface

class AutoEncoderPreprocessor(Preprocessor):
    def __init__(self, 
            encoder: NNTrainableModule, 
            decoder: NNTrainableModule, 
            *args, 
            input_channels=3,
            input_mean=0.0,
            input_scale=1.0,
            distortion_type="mse", # mse, ms-ssim, etc
            distortion_lambda=1.0,
            vr_lambda_list=None,
            freeze_encoder=False,
            freeze_decoder=False,
            num_complex_levels=1,
            **kwargs):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

        # input params
        self.input_channels = input_channels
        self.input_mean = input_mean
        self.input_scale = input_scale

        # loss params
        self.distortion_lambda = distortion_lambda
        self.vr_lambda_list = vr_lambda_list
        self.distortion_type = distortion_type
        
        if self.vr_lambda_list is not None:
            self.active_vr_level = 0

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.lr_modifier = 0.0

        if freeze_decoder:
            for p in self.decoder.parameters():
                p.lr_modifier = 0.0

        # scalable
        # self.scalable_default_dnn_configs = scalable_default_dnn_configs
        # if scalable_default_dnn_configs is not None:
        # NOTE: this is only a dummy parameter. We leave the implementation to subclasses
        self._num_complex_levels = num_complex_levels
        if self._num_complex_levels > 1:
            self.active_complex_level = 0

    def preprocess(self, data, *args, prior=None, **kwargs):
        # handle device
        if data.device != self.device:
            data = data.to(device=self.device)

        latent = self.encoder(data)
        return self.quantize(latent, mode="symbols")

    def postprocess(self, data, *args, prior=None, **kwargs):
        output = self.decoder(data)
        return self.quantize(output, mode="dequantize")

    def _calc_loss_distortion(self, x_hat, x):
        if self.distortion_type == "none":
            return None
        elif self.distortion_type == "mse":
            mse_loss = F.mse_loss(x_hat, x, reduction="sum")
            loss_distortion = mse_loss
            self.update_cache("metric_dict", mse=mse_loss / x.numel())
        # TODO: ms-ssim
        return loss_distortion / x.shape[0]

    # from CompressAI
    def quantize(
        self, inputs: torch.Tensor, mode: str, means: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        inputs.nelement
        if mode not in ("noise", "dequantize", "symbols"):
            raise ValueError(f'Invalid quantization mode: "{mode}"')

        if mode == "noise":
            half = float(0.5)
            noise = torch.empty_like(inputs).uniform_(-half, half)
            inputs = inputs + noise
            return inputs

        outputs = inputs.clone()
        if means is not None:
            outputs -= means

        outputs = torch.round(outputs)

        if mode == "dequantize":
            if means is not None:
                outputs += means
            return outputs

        assert mode == "symbols", mode
        outputs = outputs.int()
        return outputs

    def forward(self, data, *args, **kwargs):
        # select random level during training
        if self.vr_lambda_list is not None:
            if self.training:
                self.set_rate_level(np.random.randint(0, self.num_rate_levels))
            # TODO: what to do during validation?

        # handle device
        if data.device != self.device:
            data = data.to(device=self.device)

        x = data
        z = self.encoder((data - self.input_mean) / self.input_scale)
        z_hat = self.quantize(z, "noise" if self.training else "dequantize")
        x_hat = self.decoder(z_hat)
        loss_distortion = self._calc_loss_distortion(x_hat, x)

        losses = dict()
        if self.training:
            if loss_distortion is not None:
                losses.update(loss_distortion=self.distortion_lambda * loss_distortion)

            self.update_cache("loss_dict", **losses)

        return x_hat
