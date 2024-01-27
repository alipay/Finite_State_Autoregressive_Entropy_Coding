import itertools
import os
from typing import Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

import math
import numpy as np
from cbench.modules.prior_model.prior_coder import MultiChannelVQPriorCoder, NNPriorCoder
from cbench.modules.prior_model.prior_coder.base import PriorCoder

from cbench.nn.base import NNTrainableModule, PLNNTrainableModule
from cbench.nn.models.vae import VAEEncoder, VAEDecoder
from cbench.nn.models.vqvae_model_v2 import Encoder, Decoder
from cbench.nn.models.google import HyperpriorSynthesisModel, HyperpriorAnalysisModel
from cbench.nn.trainer import make_optimizer, make_scheduler
from cbench.nn.utils import batched_cross_entropy

from .base import PriorModel
from ..base import TrainableModuleInterface


class AutoEncoderPriorModel(PriorModel, NNTrainableModule):
    """Autoencoder (1-layer) based prior model.

        Args:
            encoder (NNTrainableModule): The encoder module.
            decoder (NNTrainableModule): The decoder module.
            prior_coder (NNPriorCoder): The prior coder module.
            input_channels (int, optional): Number of input channels. Defaults to 3.
            input_mean (float, optional): Mean of input, will be used to minus input. Defaults to 0.0.
            input_scale (float, optional): Mean of input, will be used to divide input. Defaults to 1.0.
            lambda_rd (float, optional): Controls distortion weight. For lossless this should always be 1.0. Defaults to 1.0.
            distortion_type (str, optional): Distortion format. If entropy coder is correctly setup, this should be "none". Defaults to "mse".
            freeze_encoder (bool, optional): Freeze encoder parameters. Defaults to False.
            freeze_decoder (bool, optional): Freeze decoder parameters. Defaults to False.
            train_mc_sampling (bool, optional): Beta. For monte-carlo optimization. Defaults to False.
            test_mc_sampling (bool, optional): Beta. For monte-carlo optimization. Defaults to False.
            test_mc_sampling_reduce_method (str, optional): Beta. For monte-carlo optimization. Defaults to "mean".
            mc_sampling_size (int, optional): Beta. For monte-carlo optimization. Defaults to 16.
            mc_sampling_use_kl_weight (bool, optional): Beta. For monte-carlo optimization. Defaults to False.
            use_vamp_prior (bool, optional): Beta. For VAMP prior. Defaults to False.
            vamp_input_size (tuple, optional): Beta. For VAMP prior. Defaults to (2, 3, 32, 32).
            train_em_update (bool, optional): Beta. For EMVB. Defaults to False.
            train_em_use_optimizer (bool, optional): Beta. For EMVB. Defaults to False.
            var_scale (float, optional): Beta. For SQVAE. Defaults to 1.0.
            var_scale_anneal (bool, optional): Beta. For SQVAE. Defaults to False.
            gap_control_loss (float, optional): Beta. For soft-to-hard VQ. Defaults to 1.0.
            gap_control_loss_anneal (bool, optional): Beta. For soft-to-hard VQ. Defaults to False.
    """    
    def __init__(self, 
            encoder: NNTrainableModule, 
            decoder: NNTrainableModule, 
            prior_coder: NNPriorCoder, 
            *args, 
            input_channels=3,
            input_mean=0.0,
            input_scale=1.0,
            lambda_rd=1.0,
            distortion_type="ce", # none, mse, ce, etc.
            freeze_encoder=False,
            freeze_decoder=False,
            train_mc_sampling=False, 
            test_mc_sampling=False,
            test_mc_sampling_reduce_method="mean",
            mc_sampling_size=16, 
            mc_sampling_use_kl_weight=False, 
            use_vamp_prior=False,
            vamp_input_size=(2, 3, 32, 32),
            train_em_update=False,
            train_em_use_optimizer=False,
            var_scale=1.0, var_scale_anneal=False,
            gap_control_loss=1.0, gap_control_loss_anneal=False,
            **kwargs):
   
        super().__init__()
        NNTrainableModule.__init__(self)
        self.encoder = encoder
        self.decoder = decoder
        self.prior_coder = prior_coder

        # input params
        self.input_channels = input_channels
        self.input_mean = input_mean
        self.input_scale = input_scale

        # loss params
        self.lambda_rd = lambda_rd
        self.distortion_type = distortion_type
        
        # monte-carlo sampling
        self.train_mc_sampling = train_mc_sampling
        self.test_mc_sampling = test_mc_sampling
        self.test_mc_sampling_reduce_method = test_mc_sampling_reduce_method
        self.mc_sampling_size = mc_sampling_size
        self.mc_sampling_use_kl_weight = mc_sampling_use_kl_weight

        # vamp prior
        self.use_vamp_prior = use_vamp_prior
        if use_vamp_prior:
            self.vamp_pseudo_input = nn.Parameter(torch.Tensor(*vamp_input_size))
            nn.init.uniform_(self.vamp_pseudo_input)

        # EMVB
        self.train_em_update = train_em_update
        self.train_em_use_optimizer = train_em_use_optimizer
        if train_em_update:
            self.em_state = False
            if train_em_use_optimizer:
                for param in self.prior_coder.parameters():
                    param.aux_id = 0
                for param in self.decoder.parameters():
                    param.aux_id = 0
                # self.estep_optimizer = make_optimizer(self.encoder)
                # self.mstep_optimizer = make_optimizer(nn.ModuleList([self.prior_coder, self.decoder]))

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.lr_modifier = 0.0

        if freeze_decoder:
            for p in self.decoder.parameters():
                p.lr_modifier = 0.0

        #
        if self.distortion_type == "normal_var" or self.distortion_type == "normal_var_quant":
            self.param_logvar = nn.Parameter(torch.zeros(1))
        elif self.distortion_type == "normal_var_fixed" or self.distortion_type == "normal_var_fixed_quant":
            self.register_buffer("param_logvar", torch.zeros(1))

        self.var_scale_anneal = var_scale_anneal
        if var_scale_anneal:
            self.var_scale = nn.Parameter(torch.tensor(var_scale), requires_grad=False)
        else:
            self.var_scale = var_scale

        self.gap_control_loss_anneal = gap_control_loss_anneal
        if gap_control_loss_anneal:
            self.gap_control_loss = nn.Parameter(torch.tensor(gap_control_loss), requires_grad=False)
        else:
            self.gap_control_loss = gap_control_loss

        # from torchmetrics.image.fid import FrechetInceptionDistance
        # self.fid = [FrechetInceptionDistance()]

    def extract(self, data, *args, **kwargs):
        # handle device
        if data.device != self.device:
            data = data.to(device=self.device)

        with self.profiler.start_time_profile("time_encoder"):
            z = self.encoder(data)
        with self.profiler.start_time_profile("time_prior_encoder"):
            byte_string = self.prior_coder.encode(z)
        # log parameter size (may not be correct!)
        # param_size = 0
        # for param in self.prior_coder.parameters():
        #     param_size += param.numel() * param.element_size()
        # self.profiler.update(params_prior_encoder=param_size)
        return byte_string

    def predict(self, data, *args, prior=None, **kwargs):
        with self.profiler.start_time_profile("time_prior_decoder"):
            prior = self.prior_coder.decode(prior)
        with self.profiler.start_time_profile("time_decoder"):
            output = self.decoder(prior)
            # move prob to last dim
            num_symbols = output.shape[1] // self.input_channels
            output = output.reshape(output.shape[0], self.input_channels, num_symbols, *output.shape[2:]).movedim(2, -1).squeeze(-1)
            # TODO: trim output spatial shape to be same as input
        # log parameter size (may not be correct!)
        # param_size = 0
        # for param in self.prior_coder.parameters():
        #     param_size += param.numel() * param.element_size()
        # self.profiler.update(params_prior_decoder=param_size)
        return output

    def update_state(self, *args, **kwargs) -> None:
        self.prior_coder.update_state(*args, **kwargs)
        # TODO: update scalable complex levels

    def _latent_metric(self, x, z, *args, **kwargs):
        pass

    def _calc_loss_distortion(self, x_hat, x, weight=None):
        if weight is None: 
            weight = torch.ones(x.shape[0]).type_as(x) / x.shape[0]

        if self.distortion_type == "none":
            self.update_cache("metric_dict", estimated_x_epd=0)
            return None
        elif self.distortion_type == "mse":
            mse_batch = (x_hat - x).pow(2).view(x.shape[0], -1).mean(dim=-1) # F.mse_loss(x_hat, x)
            loss_distortion = (mse_batch * weight).sum()
            self.update_cache("metric_dict", mse=mse_batch.mean())
        elif self.distortion_type == "ce":
            ce_batch = batched_cross_entropy(x_hat, x).view(x.shape[0], -1).mean(dim=-1)
            loss_distortion = (ce_batch * weight).sum()
            # update coding length metric
            # if not self.training:
            self.update_cache("metric_dict", estimated_x_epd=loss_distortion)
        elif self.distortion_type == "normal":
            mean, logvar = x_hat.chunk(2, dim=1)
            mse = (mean - x).pow(2) 
            loss = mse / (2*logvar.exp()) + logvar / 2
            loss_batch = loss.reshape(x.shape[0], -1).mean(dim=-1)
            loss_distortion = (loss_batch * weight).sum()
            self.update_cache("metric_dict", mse=mse.mean())
        elif self.distortion_type == "normal_sigmoid_mean":
            mean, logvar = x_hat.chunk(2, dim=1)
            mse = (mean - x).pow(2) 
            loss = mse / (2*logvar.exp()) + logvar / 2
            loss_batch = loss.reshape(x.shape[0], -1).mean(dim=-1)
            loss_distortion = (loss_batch * weight).sum()
            self.update_cache("metric_dict", mse=mse.mean())
        elif self.distortion_type == "normal_var" or self.distortion_type == "normal_var_fixed":
            mse = (x_hat - x).pow(2) 
            logvar = torch.ones_like(mse) * self.param_logvar + torch.log(self.var_scale + 1e-8)
            if self.training and self.var_scale_anneal:
                self.update_cache("moniter_dict", 
                    var_scale=self.var_scale
                )
            loss = mse / (2*logvar.exp()) + logvar / 2
            loss_batch = loss.view(x.shape[0], -1).mean(dim=-1)
            loss_distortion = (loss_batch * weight).sum()
            self.update_cache("moniter_dict", normal_var=2*logvar.exp().mean())
            self.update_cache("metric_dict", mse=mse.mean())
        elif self.distortion_type == "normal_arelbo":
            mse = (x_hat - x).pow(2) 
            loss_batch = mse.view(x.shape[0], -1).mean(dim=-1)
            loss_distortion = (loss_batch * weight).sum()
            dim_mult = torch.ones_like(mse).sum() / x.shape[0]
            loss_distortion = dim_mult * loss_distortion.log() / 2
            self.update_cache("metric_dict", mse=mse.mean())
        elif self.distortion_type == "normal_quant":
            mean, logvar = x_hat.chunk(2, dim=1)
            dist = D.Normal(mean, torch.exp(logvar))
            x_norm = x.clamp(0, 1)
            x_target = (x_norm * 255) # .detach()
            probs = dist.cdf(x_target + 0.5) - dist.cdf(x_target - 0.5)
            logits = -(probs + 1e-6).log()
            logits_batch = logits.view(x.shape[0], -1).mean(dim=-1)
            loss_distortion = (logits_batch * weight).sum()
            # update coding length metric
            # if not self.training:
            self.update_cache("metric_dict", estimated_x_epd=loss_distortion)
            # if not self.training:
            #     self.update_cache("image_dict", x_samples=(mean/255).clamp(0, 1))
            #     self.update_cache("image_dict", x_target=x_target/255)
            # calculate FID score
            # if not self.training:
            #     try:
            #         self.fid[0].to(self.device)
            #         self.fid[0].set_dtype(mean.dtype)
            #         self.fid[0].reset()
            #         self.fid[0].update(mean.byte(), False)
            #         self.fid[0].update(x_target.byte(), True)
            #         fid_score = self.fid[0].compute()
            #         self.update_cache("metric_dict", fid_score=fid_score)
            #     except:
            #         print("Skipping FID score...")
            # log samples

        elif self.distortion_type == "normal_quant_sigmoid_mean":
            mean, logvar = x_hat.chunk(2, dim=1)
            dist = D.Normal(torch.sigmoid(mean) * 255, torch.exp(logvar))
            x_norm = x.clamp(0, 1)
            x_target = (x_norm * 255) # .detach()
            probs = dist.cdf(x_target + 0.5) - dist.cdf(x_target - 0.5)
            logits = -(probs + 1e-6).log()
            logits_batch = logits.view(x.shape[0], -1).mean(dim=-1)
            loss_distortion = (logits_batch * weight).sum()
        elif self.distortion_type == "normal_var_quant" or self.distortion_type == "normal_var_fixed_quant":
            logvar = torch.ones_like(x_hat) * self.param_logvar + torch.log(self.var_scale + 1e-8)
            self.update_cache("moniter_dict", normal_var=2*logvar.exp().mean())
            dist = D.Normal(x_hat, torch.exp(logvar))
            x_norm = x.clamp(0, 1)
            x_target = (x_norm * 255) # .detach()
            probs = dist.cdf(x_target + 0.5) - dist.cdf(x_target - 0.5)
            logits = -(probs + 1e-6).log()
            logits_batch = logits.view(x.shape[0], -1).mean(dim=-1)
            loss_distortion = (logits_batch * weight).sum()
            # update coding length metric
            # if not self.training:
            self.update_cache("metric_dict", estimated_x_epd=loss_distortion)
        else:
            raise NotImplementedError("")

        return loss_distortion


    def forward(self, data, *args, **kwargs):
        # handle device
        if data.device != self.device:
            data = data.to(device=self.device)

        if self.training and self.train_em_update:
            if self.train_em_use_optimizer:
                self.em_state = (self.optim_state == 0)
                self.set_custom_state("EM-E" if self.em_state else "EM-M")
            else:
                # EM state switch
                if self.em_state:
                    self.set_custom_state("EM-M")
                else:
                    self.set_custom_state("EM-E")
                self.em_state = not self.em_state

        if self.use_vamp_prior:
            vamp_pseudo_input = F.hardtanh(self.vamp_pseudo_input, 0, 1)
            vamp_posterior = self.encoder(vamp_pseudo_input)
            self.prior_coder.set_vamp_posterior(vamp_posterior)

        x_hat, losses = self._forward_process(data)

        if self.training:
            self.update_cache("loss_dict", **losses)

        # EM update using optimizer
        # if self.training and self.train_em_use_optimizer:
            # self.estep_optimizer.zero_grad()
            # self.mstep_optimizer.zero_grad()

            # loss_dict = self.get_cache("loss_dict")
            # loss = sum(loss_dict.values())
            # loss.backward()
            # if self.em_state:
            #     self.estep_optimizer.step()
            # else:
            #     self.mstep_optimizer.step()

            # self.reset_cache("loss_dict")
            # # keep loss logs
            # self.update_cache("loss_dict",
            #                   **{k:v.detach() for k,v in loss_dict.items()})

        
        num_symbols = x_hat.shape[1] // self.input_channels
        output = x_hat.reshape(x_hat.shape[0], self.input_channels, num_symbols, *x_hat.shape[2:]).movedim(2, -1)

        return output


    def _forward_process(self, data):
        x = data
        z = self.encoder((x - self.input_mean) / self.input_scale)
        z_hat = self.prior_coder(z)
        self._latent_metric(x, z)
        x_hat = self.decoder(z_hat)
        loss_distortion = self._calc_loss_distortion(x_hat, x)

        # loss_rate and loss_distortion should be normalized together!
        losses = dict()
        if self.training:
            # gap control loss
            if self.gap_control_loss < 1.0:
                self.prior_coder.eval()
                z_hat_eval = self.prior_coder(z)
                x_hat_eval = self.decoder(z_hat_eval)
                loss_distortion_eval = self._calc_loss_distortion(x_hat_eval, x)
                self.prior_coder.train()
                loss_gap = (1-self.gap_control_loss) * self.lambda_rd * (loss_distortion_eval - loss_distortion)
                losses.update(loss_gap=loss_gap)
                if self.gap_control_loss_anneal:
                    self.update_cache("moniter_dict", 
                        gap_control_loss=self.gap_control_loss
                    )

            if self.train_mc_sampling:
                # for _ in range(self.mc_sampling_size-1):
                #     z_hat = self.prior_coder(z)
                #     x_hat = self.decoder(z_hat)
                #     loss_distortion += self._calc_loss_distortion(x_hat, x)
                repeat_dims = [self.mc_sampling_size if i==0 else 1 for i in range(len(z.shape))]
                z_mc = z.repeat(*repeat_dims)
                x_mc = x.repeat(*repeat_dims)
                if self.mc_sampling_use_kl_weight:
                    z_hat_mc = self.prior_coder(z_mc, calculate_sample_kl=True)
                    sample_kl = self.prior_coder.get_raw_cache("common").get('sample_kl')
                    if sample_kl is None:
                        raise ValueError("kl_weight is not available in prior_coder!")
                    kl_weight = torch.softmax(sample_kl, dim=0)
                    x_hat_mc = self.decoder(z_hat_mc)
                    loss_distortion = self._calc_loss_distortion(x_hat_mc, x_mc, weight=kl_weight)
                else:
                    z_hat_mc = self.prior_coder(z_mc)
                    x_hat_mc = self.decoder(z_hat_mc)
                    loss_distortion = self._calc_loss_distortion(x_hat_mc, x_mc)
            # loss = loss_rate + self.lambda_rd * loss_distortion
            # self.loss_dict.update(loss_autoencoder=loss)
            # self.update_cache("loss_dict", **losses)
            # assert('loss_distortion' not in losses)
            # self.update_cache("loss_dict", loss_distortion=self.lambda_rd * loss_distortion)
            if loss_distortion is not None:
                losses.update(loss_distortion=self.lambda_rd * loss_distortion)

            # normalize loss rate with data dimension (without batch size)
            loss_rate = self.prior_coder.get_raw_cache("loss_dict").get('loss_rate')
            if loss_rate is not None:
                loss_rate = loss_rate / (data.numel() / data.size(0))
                # self.prior_coder.update_cache("loss_dict",
                #     loss_rate = loss_rate,
                # )
                losses.update(loss_rate=loss_rate)
                self.prior_coder.get_raw_cache("loss_dict").pop('loss_rate')
            # else: # do something to warn about no loss rate?
        else:
            if self.test_mc_sampling:
                repeat_dims = [self.mc_sampling_size if i==0 else 1 for i in range(len(z.shape))]
                z_mc = z.repeat(*repeat_dims)
                x_mc = x.repeat(*repeat_dims)
                z_hat_mc = self.prior_coder(z_mc)
                x_hat_mc = self.decoder(z_hat_mc)
                loss_distortion = self._calc_loss_distortion(x_hat_mc, x_mc)
                if loss_distortion is not None:
                    self.update_cache("metric_dict", mc_distortion=loss_distortion)
        
        return x_hat, losses


class LosslessAutoEncoderPriorModel(AutoEncoderPriorModel):
    def __init__(self,
                 encoder: NNTrainableModule, 
                 decoder: NNTrainableModule, 
                 prior_coder: NNPriorCoder, 
                 distortion_type="ce",
                 lambda_rd=1.0, # dummy
                #  in_channels=3, out_channels=768, 
                #  hidden_dims : List = [32, 64, 128, 256], 
                 **kwargs):
        # encoder = VAEEncoder(in_channels, hidden_dims=hidden_dims)
        # decoder = VAEDecoder(out_channels, hidden_dims=hidden_dims)
        # encoder = Encoder(hidden_dims[-1], in_channels=in_channels, out_channels=hidden_dims[-1])
        # decoder = Decoder(hidden_dims[-1], in_channels=hidden_dims[-1], out_channels=out_channels)
        super().__init__(encoder=encoder, decoder=decoder, prior_coder=prior_coder,
            distortion_type=distortion_type,
            lambda_rd=1.0,
            **kwargs
        )

        # for param in self.encoder.parameters():
        #     param.requires_grad = False
        # for param in self.decoder.parameters():
        #     param.requires_grad = False


    def forward(self, data, *args, **kwargs):
        ret = super().forward(data, *args, **kwargs)
        # if not self.training:
        # update coding length metric
        prior_entropy = self.prior_coder.metric_dict.get("prior_entropy")
        data_entropy = self.metric_dict.get("estimated_x_epd") 
        if prior_entropy is not None and data_entropy is not None:
            data_entropy = data_entropy * data.numel() / data.size(0)
            self.update_cache("metric_dict",
                estimated_entropy = (data_entropy + prior_entropy),
                estimated_epd = (data_entropy + prior_entropy) / (data.numel() / data.size(0)),
            )

        return ret


class BaseLosslessAutoEncoderPriorModel(LosslessAutoEncoderPriorModel):
    def __init__(self,
                 prior_coder : PriorCoder, 
                 in_channels=3, out_channels=768, 
                 hidden_dims : List = [32, 64, 128, 256], 
                 **kwargs):
        encoder = VAEEncoder(in_channels, hidden_dims=hidden_dims)
        decoder = VAEDecoder(out_channels, hidden_dims=hidden_dims)
        # encoder = Encoder(hidden_dims[-1], in_channels=in_channels, out_channels=hidden_dims[-1])
        # decoder = Decoder(hidden_dims[-1], in_channels=hidden_dims[-1], out_channels=out_channels)

        super().__init__(encoder=encoder, decoder=decoder, prior_coder=prior_coder,
            **kwargs
        )

    # def _latent_metric(self, x, z):
    #     super()._latent_metric(x, z)
    #     latent_total_entropy = math.log(self.num_embeddings) * z.numel() / self.embedding_dim
    #     self.update_cache("metric_dict",
    #         latent_total_entropy=latent_total_entropy,
    #     )

class LosslessAutoEncoderPriorModelBackboneV2(LosslessAutoEncoderPriorModel):
    def __init__(self,
                 prior_coder : PriorCoder, 
                 in_channels=3, out_channels=768, hidden_channels=256,
                 num_downsample_layers=2, 
                 upsample_method="conv", 
                 num_residual_layers=2,
                 use_skip_connection=False,
                 encoder_use_batch_norm=True,
                 decoder_use_batch_norm=True,
                 decoder_batch_norm_track=True,
                 **kwargs):
        encoder = Encoder(hidden_channels, in_channels=in_channels, out_channels=hidden_channels,
            num_downsample_layers=num_downsample_layers, 
            num_residual_layers=num_residual_layers, use_skip_connection=use_skip_connection,
            use_batch_norm=encoder_use_batch_norm)
        decoder = Decoder(hidden_channels, in_channels=hidden_channels, out_channels=out_channels,
            num_upsample_layers=num_downsample_layers, upsample_method=upsample_method,
            num_residual_layers=num_residual_layers, use_skip_connection=use_skip_connection,
            use_batch_norm=decoder_use_batch_norm ,batch_norm_track=decoder_batch_norm_track)

        super().__init__(encoder=encoder, decoder=decoder, prior_coder=prior_coder,
            **kwargs
        )


class VAELosslessAutoEncoderPriorModel(BaseLosslessAutoEncoderPriorModel):
    def __init__(self,
                 latent_channels = 128,
                 hidden_dims : List = [32, 64, 128, 256], 
                 **kwargs):
        # if hidden_dims is None:
        #     hidden_dims = [32, 64, 128, 256, 512]

        prior_coder = GaussianPriorCoder(hidden_dims[-1], latent_channels=latent_channels)

        super().__init__(prior_coder=prior_coder,
            hidden_dims=hidden_dims,
            **kwargs
        )


class VQVAELosslessAutoEncoderPriorModel(BaseLosslessAutoEncoderPriorModel):
    def __init__(self,
                 latent_dim=8, num_embeddings=128,
                 hidden_dims : List = [32, 64, 128, 256], 
                 **kwargs):
        # if hidden_dims is None:
        #     hidden_dims = [32, 64, 128, 256, 512]
        self.latent_dim = latent_dim
        self.num_embeddings = num_embeddings
        self.embedding_dim = hidden_dims[-1] // latent_dim
        prior_coder = MultiChannelVQPriorCoder(self.latent_dim, self.num_embeddings, self.embedding_dim)

        super().__init__(prior_coder=prior_coder,
            hidden_dims=hidden_dims,
            **kwargs
        )

    # def _latent_metric(self, x, z):
    #     super()._latent_metric(x, z)
    #     latent_total_entropy = math.log(self.num_embeddings) * z.numel() / self.embedding_dim
    #     self.update_cache("metric_dict",
    #         latent_total_entropy=latent_total_entropy,
    #     )


class VQVAEV2BackboneAutoEncoderPriorModel(LosslessAutoEncoderPriorModel):
    def __init__(self,
                 in_channels=3, out_channels=768, 
                 hidden_channels=256,
                 latent_dim=8, num_embeddings=128,
                 distortion_type="ce",
                 **kwargs):
        self.latent_dim = latent_dim
        self.num_embeddings = num_embeddings
        self.embedding_dim = hidden_channels // latent_dim

        encoder = Encoder(hidden_channels, in_channels=in_channels, out_channels=hidden_channels)
        decoder = Decoder(hidden_channels, in_channels=hidden_channels, out_channels=out_channels)
        prior_coder = MultiChannelVQPriorCoder(self.latent_dim, self.num_embeddings, self.embedding_dim)

        super().__init__(encoder=encoder, decoder=decoder, prior_coder=prior_coder,
            distortion_type=distortion_type,
            lambda_rd=1.0,
            **kwargs
        )

    # def _latent_metric(self, x, z):
    #     super()._latent_metric(x, z)
    #     latent_total_entropy = math.log(self.num_embeddings) * z.numel() / self.embedding_dim
    #     self.update_cache("metric_dict",
    #         latent_total_entropy=latent_total_entropy,
    #     )


class BaseLossyAutoEncoderPriorModel(AutoEncoderPriorModel):
    def __init__(self,
                 prior_coder : PriorCoder, 
                 in_channels=3, out_channels=3, 
                 hidden_dims : List = [32, 64, 128, 256], 
                 distortion_type="mse",
                 **kwargs):
        encoder = VAEEncoder(in_channels, hidden_dims=hidden_dims)
        decoder = VAEDecoder(out_channels, hidden_dims=hidden_dims)
        # encoder = Encoder(hidden_dims[-1], in_channels=in_channels, out_channels=hidden_dims[-1])
        # decoder = Decoder(hidden_dims[-1], in_channels=hidden_dims[-1], out_channels=out_channels)

        super().__init__(encoder=encoder, decoder=decoder, prior_coder=prior_coder,
            distortion_type=distortion_type,
            **kwargs
        )

    def forward(self, data, *args, **kwargs):
        ret = super().forward(data, *args, **kwargs)
        # if not self.training:
        # update coding length metric
        prior_entropy = self.prior_coder.metric_dict.get("prior_entropy")
        if prior_entropy is not None:
            self.update_cache("metric_dict",
                estimated_epd = prior_entropy / (data.numel() / data.size(0)),
                estimated_bpd = prior_entropy / math.log(2) / (data.numel() / data.size(0)),
            )

        return ret


class GoogleLossyAutoEncoderPriorModel(AutoEncoderPriorModel):
    def __init__(self,
                 prior_coder : PriorCoder, 
                 in_channels=3, out_channels=3, 
                 mid_channels=128, latent_channels=192,
                 distortion_type="mse",
                 **kwargs):
        encoder = HyperpriorAnalysisModel(mid_channels, latent_channels, in_channels=in_channels)
        decoder = HyperpriorSynthesisModel(mid_channels, latent_channels, out_channels=out_channels)
        super().__init__(encoder=encoder, decoder=decoder, prior_coder=prior_coder,
            distortion_type=distortion_type,
            **kwargs
        )

    def forward(self, data, *args, **kwargs):
        ret = super().forward(data, *args, **kwargs)
        # if not self.training:
        # update coding length metric
        prior_entropy = self.prior_coder.metric_dict.get("prior_entropy")
        if prior_entropy is not None:
            self.update_cache("metric_dict",
                estimated_epd = prior_entropy / (data.numel() / data.size(0)),
                estimated_bpd = prior_entropy / math.log(2) / (data.numel() / data.size(0)),
            )

        return ret
