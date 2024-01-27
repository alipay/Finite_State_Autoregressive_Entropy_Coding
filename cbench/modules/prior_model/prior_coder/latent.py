import math
import struct
import copy
import itertools
from typing import Any, Dict, List, Tuple, Optional
import numpy as np

import torch
import torch.autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
import torch.distributed as distributed
from torch.distributions.utils import clamp_probs, probs_to_logits

from entmax import entmax_bisect

from cbench.nn.base import NNTrainableModule
from cbench.nn.layers import Upsample2DLayer, Downsample2DLayer, MaskedConv2d, MaskedConv3d
from cbench.nn.distributions.relaxed import CategoricalRSample, RelaxedOneHotCategorical, AsymptoticRelaxedOneHotCategorical, DoubleRelaxedOneHotCategorical, InvertableGaussianSoftmaxppRelaxedOneHotCategorical
from cbench.modules.entropy_coder.utils import BinaryHeadConstructor
from cbench.modules.entropy_coder.rans import pmf_to_quantized_cdf_serial, pmf_to_quantized_cdf_batched
from cbench.utils.bytes_ops import encode_shape, decode_shape
from cbench.utils.bytes_ops import merge_bytes, split_merged_bytes
from cbench.utils.ar_utils import create_ar_offsets

from cbench.ans import Rans64Encoder, Rans64Decoder, TansEncoder, TansDecoder

from cbench.rans import BufferedRansEncoder, RansEncoder, RansDecoder, pmf_to_quantized_cdf

from .base import PriorCoder
from .sqvae_coder import GaussianVectorQuantizer, VmfVectorQuantizer


class NNPriorCoder(PriorCoder, NNTrainableModule):
    def __init__(self):
        super().__init__()
        NNTrainableModule.__init__(self)

    def forward(self, input : torch.Tensor, prior : torch.Tensor = None, **kwargs):
        raise NotImplementedError()

    def set_vamp_posterior(self, posterior):
        raise NotImplementedError()

    def encode(self, input : torch.Tensor, *args, **kwargs) -> bytes:
        raise NotImplementedError()

    def decode(self, byte_string : bytes, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError()

    """
    A quick function that combines encode and decode. May be overrided to skip latent decoding for faster training.
    """
    def encode_and_decode(self, input : torch.Tensor, *args, **kwargs) -> Tuple[bytes, torch.Tensor]:
        byte_string = self.encode(input, *args, **kwargs)
        return byte_string, self.decode(byte_string, *args, **kwargs)

class HierarchicalNNPriorCoder(NNPriorCoder):
    """Hierarchical Autoencoder.
    """    
    def __init__(self, 
        encoders : List[nn.Module],
        decoders : List[nn.Module],
        prior_coders : List[NNPriorCoder],
        **kwargs):
        super().__init__()

        self.num_layers = len(prior_coders)
        self.encoders = nn.ModuleList(encoders)
        self.decoders = nn.ModuleList(decoders)
        self.prior_coders = nn.ModuleList(prior_coders)
        assert len(self.encoders) == self.num_layers-1
        assert len(self.decoders) == self.num_layers-1
        assert len(self.prior_coders) == self.num_layers

    def forward(self, input: torch.Tensor, **kwargs):
        latent = input
        latent_enc_nlayer = []
        latent_enc = latent
        for i in range(self.num_layers-1):
            latent_enc = self.encoders[i](latent_enc)
            latent_enc_nlayer.append(latent_enc)

        # prior decoding in inverse direction
        latent_dec = None
        for i in range(self.num_layers-2, -1, -1):
            latent_dec = self.prior_coders[i+1](latent_enc_nlayer[i], prior=latent_dec)
            # save latent_dec as the prior dist for the prev prior_coder
            latent_dec = self.decoders[i](latent_dec) 

        # final output
        latent = self.prior_coders[0](latent, prior=latent_dec)

        # collect all loss_rate
        if self.training:
            loss_rate = 0
            for idx, prior_coder in enumerate(self.prior_coders):
                loss_dict = prior_coder.get_raw_cache("loss_dict")
                if loss_dict.get("loss_rate") is not None:
                    # pop the loss_rate in submodule to avoid multiple losses
                    loss_rate += loss_dict.pop("loss_rate")
                else:
                    print(f"No loss_rate found in self.prior_coders[{idx}]! Check the implementation!")

            self.update_cache("loss_dict",
                loss_rate = loss_rate,
            )

        # collect all prior_entropy
        prior_entropy = 0
        for idx, prior_coder in enumerate(self.prior_coders):
            metric_dict = prior_coder.get_raw_cache("metric_dict")
            if metric_dict.get("prior_entropy") is not None:
                prior_entropy += metric_dict["prior_entropy"]
            else:
                print(f"No prior_entropy found in self.prior_coders[{idx}]! Check the implementation!")

        self.update_cache("metric_dict",
            prior_entropy = prior_entropy,
        )

        return latent

    def encode(self, input: torch.Tensor, **kwargs):
        latent = input
        latent_enc_nlayer = []
        latent_enc = latent
        for i in range(self.num_layers-1):
            latent_enc = self.encoders[i](latent_enc)
            latent_enc_nlayer.append(latent_enc)

        # prior decoding in inverse direction
        latent_byte_strings = []
        latent_dec = None
        for i in range(self.num_layers-2, -1, -1):
            latent_bytes, latent_dec = self.prior_coders[i+1].encode_and_decode(latent_enc_nlayer[i], prior=latent_dec)
            latent_byte_strings.append(latent_bytes)
            # save latent_dec as the prior dist for the prev prior_coder
            latent_dec = self.decoders[i](latent_dec) 

        # final output
        latent_bytes = self.prior_coders[0].encode(latent, prior=latent_dec)
        latent_byte_strings.append(latent_bytes)

        return merge_bytes(latent_byte_strings, num_segments=len(self.prior_coders))

    def decode(self, byte_string: bytes, *args, **kwargs) -> torch.Tensor:
        latent_byte_strings = split_merged_bytes(byte_string, num_segments=len(self.prior_coders))
        
        # prior decoding in inverse direction
        latent_dec = None
        for i in range(self.num_layers-2, -1, -1):
            latent_bytes = latent_byte_strings.pop(0)
            latent_dec = self.prior_coders[i+1].decode(latent_bytes, prior=latent_dec)
            # save latent_dec as the prior dist for the prev prior_coder
            latent_dec = self.decoders[i](latent_dec) 

        # final output
        latent_dec = self.prior_coders[0].decode(latent_byte_strings[-1], prior=latent_dec)
        return latent_dec
    
    # TODO:
    def encode_and_decode(self, input: torch.Tensor, *args, **kwargs) -> Tuple[bytes, torch.Tensor]:
        return super().encode_and_decode(input, *args, **kwargs)


class NNPriorCoderFlatLinearTransform(NNPriorCoder):
    """Reshape multi-dimensional input to shape [N, *, C] \
    and transform it to [N, *, self.latent_channels_in] \
    and to [N, *, self.latent_channels_out] with latent model,\
    and finally transform back to [N, *, C] and reshape as input.
    Used as the basis for 1D/2D/3D input support.
    Subclasses implement self._forward_flat to enable training, and\
    self._encode_transformed / self._decode_transformed to enable coding.

    Args:
        in_channels (int, optional): input channels. Defaults to 256.
        skip_layers_if_equal_channels (bool, optional): \
            if self.in_channels == self.latent_channels_in / self.latent_channels_out, skip self.input_layer / self.output_layer. \
            Defaults to False.
        freeze_input_layer (bool, optional): [description]. Defaults to False.
        freeze_output_layer (bool, optional): [description]. Defaults to False.
    """    
    def __init__(self, in_channels=256, 
                 skip_layers_if_equal_channels=False,
                 freeze_input_layer=False,
                 freeze_output_layer=False,
                 **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.skip_layers_if_equal_channels = skip_layers_if_equal_channels

        self.input_layer = nn.Linear(self.in_channels, self.latent_channels_in)
        if self.skip_layers_if_equal_channels and self.in_channels == self.latent_channels_in:
            self.input_layer = nn.Identity()
        # self.input_layer.weight.data = torch.eye(self.in_channels)
        # self.input_layer.bias.data = torch.zeros(self.in_channels)
        # self.input_layer_mu = nn.Linear(self.in_channels, self.latent_channels)
        # self.input_layer_logvar = nn.Linear(self.in_channels, self.latent_channels)
        self.output_layer = nn.Linear(self.latent_channels_out, self.in_channels)
        # self.output_layer.weight.data = torch.eye(self.in_channels)
        # self.output_layer.bias.data = torch.zeros(self.in_channels)
        if self.skip_layers_if_equal_channels and self.in_channels == self.latent_channels_out:
            self.output_layer = nn.Identity()

        if freeze_input_layer:
            for p in self.input_layer.parameters():
                p.lr_modifier = 0.0

        if freeze_output_layer:
            for p in self.output_layer.parameters():
                p.lr_modifier = 0.0

    @property
    def latent_channels_in(self):
        return self.in_channels

    @property
    def latent_channels_out(self):
        return self.in_channels

    def _forward_flat(self, input : torch.Tensor, input_shape : torch.Size, prior : torch.Tensor = None, **kwargs):
        raise NotImplementedError()

    def _encode_transformed(self, input : torch.Tensor, prior : torch.Tensor = None, **kwargs) -> bytes:
        raise NotImplementedError()

    def _decode_transformed(self, byte_string : bytes, prior : torch.Tensor = None, **kwargs) -> torch.Tensor:
        raise NotImplementedError()

    def forward(self, input : torch.Tensor, prior : torch.Tensor = None, **kwargs):
        input_shape = input.shape
        batch_size = input.shape[0]
        channel_size = input.shape[1]
        assert(channel_size == self.in_channels)

        if prior is not None:
            assert(prior.shape == input.shape)
            prior = prior.reshape(batch_size, channel_size, -1).permute(0, 2, 1).reshape(-1, channel_size).contiguous()
            prior = self.input_layer(prior)
        input = input.reshape(batch_size, channel_size, -1).permute(0, 2, 1).reshape(-1, channel_size).contiguous()
        # input = torch.flatten(input, start_dim=1)
        input = self.input_layer(input) #.reshape(batch_size, -1, self.latent_channels * 2).permute(0, 2, 1)

        output = self._forward_flat(input, input_shape, prior=prior, **kwargs)

        # if prior is not None:
        #     prior = self.output_layer(prior)
        #     prior = prior.reshape(batch_size, -1, channel_size).permute(0, 2, 1).reshape(*input_shape).contiguous()
        #     assert(prior.shape == output.shape)
        output = self.output_layer(output)
        output = output.reshape(batch_size, -1, channel_size).permute(0, 2, 1).reshape(*input_shape).contiguous()

        return output

    def encode(self, input : torch.Tensor, *args, prior : torch.Tensor = None, **kwargs) -> bytes:
        input_shape = input.shape
        batch_size = input.shape[0]
        channel_size = input.shape[1]
        assert(channel_size == self.in_channels)
        spatial_shape = input.shape[2:]

        if prior is not None:
            assert(prior.shape == input.shape)
            prior = prior.reshape(batch_size, channel_size, -1).permute(0, 2, 1).reshape(-1, channel_size).contiguous()
            prior = self.input_layer(prior)
            prior = prior.reshape(batch_size, -1, self.latent_channels_in).permute(0, 2, 1)\
                .reshape(batch_size, self.latent_channels_in, *spatial_shape).contiguous()
        input = input.reshape(batch_size, channel_size, -1).permute(0, 2, 1).reshape(-1, channel_size).contiguous()
        # input = torch.flatten(input, start_dim=1)
        input = self.input_layer(input)
        input = input.reshape(batch_size, -1, self.latent_channels_in).permute(0, 2, 1)\
            .reshape(batch_size, self.latent_channels_in, *spatial_shape).contiguous()

        return self._encode_transformed(input, prior=prior, **kwargs)

    def decode(self, byte_string : bytes, *args, prior : torch.Tensor = None, **kwargs) -> torch.Tensor:
        if prior is not None:
            batch_size = prior.shape[0]
            channel_size = prior.shape[1]
            assert(channel_size == self.in_channels)
            spatial_shape = prior.shape[2:]
            prior = prior.reshape(batch_size, channel_size, -1).permute(0, 2, 1).reshape(-1, channel_size).contiguous()
            prior = self.input_layer(prior)
            prior = prior.reshape(batch_size, -1, self.latent_channels_in).permute(0, 2, 1)\
                .reshape(batch_size, self.latent_channels_in, *spatial_shape).contiguous()
        
        output = self._decode_transformed(byte_string, prior=prior, *args, **kwargs)
        
        batch_size = output.shape[0]
        channel_size = output.shape[1]
        assert(channel_size == self.latent_channels_in)
        spatial_shape = output.shape[2:]
        output = output.reshape(batch_size, channel_size, -1).permute(0, 2, 1)\
            .reshape(-1, channel_size).contiguous()
        output = self.output_layer(output)
        output = output.reshape(batch_size, -1, self.in_channels).permute(0, 2, 1)\
            .reshape(batch_size, self.in_channels, *spatial_shape).contiguous()

        return output


class DistributionPriorCoder(NNPriorCoderFlatLinearTransform):
    """Simple VAE with Prior defined by torch.distributions.Distribution. Implements relative entropy coding* (not working).
    * https://arxiv.org/abs/2010.01185
    """    
    def __init__(self, in_channels=256, latent_channels=None, 
        # prior_trainable=False,
        train_em_update=False,
        coder_type="rans", # current support "rans", "tans"
        coder_freq_precision=16,
        coding_sampler="importance",
        coding_seed=0,
        coding_max_samples=256,
        coding_max_index=8,
        coding_max_aux_index=8,
        fixed_input_shape=None,
        **kwargs):
        self.latent_channels = (in_channels // self.num_posterior_params) if latent_channels is None else latent_channels
        super().__init__(in_channels=in_channels, **kwargs)
        
        # TODO: prior should be defined here?
        # self.prior_trainable = prior_trainable
        # prior_params = torch.zeros(self.latent_channels, self.num_prior_params)
        # if prior_trainable:
        #     self.prior_params = nn.Parameter(prior_params)
        # else:
        #     self.register_buffer("prior_params", prior_params, persistent=False)

        self.train_em_update = train_em_update
        if train_em_update:
            self.em_state = True

        self.coder_type = coder_type
        self.coder_freq_precision = coder_freq_precision
        self.coding_sampler = coding_sampler
        self.coding_seed = coding_seed
        self.coding_max_samples = coding_max_samples
        self.coding_max_index = coding_max_index
        self.coding_max_aux_index = coding_max_aux_index
        self.fixed_input_shape = fixed_input_shape

    @property
    def latent_channels_in(self):
        return self.latent_channels * self.num_posterior_params

    @property
    def latent_channels_out(self):
        return self.latent_channels * self.num_sample_params
    
    @property
    def num_posterior_params(self):
        return 1

    @property
    def num_prior_params(self):
        # Usually prior is the same type as posterior
        return self.num_posterior_params

    @property
    def num_sample_params(self):
        return 1

    def _encode_transformed(self, input : torch.Tensor, prior : torch.Tensor = None, **kwargs) -> bytes:
        raise NotImplementedError("Variational prior are not directly encodable!")            

    def _decode_transformed(self, byte_string : bytes, prior : torch.Tensor = None, **kwargs) -> torch.Tensor:
        raise NotImplementedError("Variational prior are not directly decodable!")

    def posterior_distribution(self, latent, **kwargs) -> distributions.Distribution:
        raise NotImplementedError()

    def prior_distribution(self, prior=None, **kwargs) -> distributions.Distribution:
        raise NotImplementedError()

    def kl_divergence(self, prior_dist : distributions.Distribution, posterior_dist : distributions.Distribution, input_shape : torch.Size = None, posterior_samples=None, **kwargs):
        """
        Default KL Divergence is calculated by stochastic sampling, rather than closed-form.
        Overwrite this to implement a closed-form kl divergence.

        Args:
            prior_dist (distributions.Distribution)
            posterior_dist (distributions.Distribution)

        Returns:
            _type_: kl_divergence post || prior
        """        
        if posterior_samples is None:
            posterior_samples = posterior_dist.rsample()
        logp = prior_dist.log_prob(posterior_samples)
        logq = posterior_dist.log_prob(posterior_samples)
        return logq - logp

    def sample_from_posterior(self, posterior_dist : distributions.Distribution, **kwargs):
        if self.training:
            output = posterior_dist.rsample()
        else:
            # TODO: use REC samples?
            output = posterior_dist.sample()
        return output
    
    def postprocess_samples(self, samples):
        return samples.reshape(-1, self.latent_channels_out)

    def set_custom_state(self, state: str = None):
        if state == "EM-E":
            self.em_state = True
        elif state == "EM-M":
            self.em_state = False
        # else:
        #     raise ValueError()
        return super().set_custom_state(state)

    def _forward_flat(self, input : torch.Tensor, input_shape : torch.Size, prior : torch.Tensor = None, **kwargs):
        if self.train_em_update:
            if self.em_state:
                posterior_dist = self.posterior_distribution(input)
                prior_dist = self.prior_distribution(prior=prior.detach())
            else:
                posterior_dist = self.posterior_distribution(input.detach())
                prior_dist = self.prior_distribution(prior=prior)
        else:
            posterior_dist = self.posterior_distribution(input)
            prior_dist = self.prior_distribution(prior=prior)

        samples = self.sample_from_posterior(posterior_dist)

        # TODO: check add posterior_samples?
        KLD = torch.sum(self.kl_divergence(prior_dist, posterior_dist, input_shape=input_shape, posterior_samples=samples))
        if self.training:
            self.update_cache("loss_dict",
                loss_rate=KLD / input_shape[0], # normalize by batch size
            )
        # if implementation has not provide prior_entropy, use kl as prior_entropy instead
        if not "prior_entropy" in self.get_raw_cache("metric_dict"):
            self.update_cache("metric_dict",
                prior_entropy = KLD / input_shape[0], # normalize by batch size
            )

        return self.postprocess_samples(samples)


class GaussianDistributionPriorCoder(DistributionPriorCoder):
    """Simple VAE with Gaussian Prior. Reimplement by inheriting from DistributionPriorCoder.
    Beta support for VAMP posterior.
    """
    def __init__(self, in_channels=256, latent_channels=None, **kwargs):
        super().__init__(in_channels, latent_channels=latent_channels, **kwargs)
        self.register_buffer("prior_means", torch.zeros(1))
        self.register_buffer("prior_scales", torch.ones(1))
        self._prior_dist = distributions.Normal(self.prior_means, self.prior_scales)
    
    @property
    def num_posterior_params(self):
        return 2

    def prior_distribution(self, prior=None, **kwargs):
        if prior is not None:
            return self.posterior_distribution(prior)
        else:
            mix = distributions.Categorical(torch.ones_like(self.prior_means) / self.prior_means.size(0))
            # TODO: per-element vamp
            comp = distributions.Normal(self.prior_means, self.prior_scales)
            return distributions.MixtureSameFamily(mix, comp)

    def posterior_distribution(self, latent, **kwargs) -> distributions.Distribution:
        mean, logvar = latent.chunk(2, dim=1)
        std = torch.exp(0.5 * logvar)
        return distributions.Normal(mean, std)

    def set_vamp_posterior(self, posterior):
        batch_size = posterior.shape[0]
        channel_size = posterior.shape[1]
        posterior = posterior.reshape(batch_size, channel_size, -1).permute(0, 2, 1)
        posterior = posterior.reshape(-1, self.num_posterior_params).contiguous()

        mean, logvar = posterior.chunk(2, dim=1)
        std = torch.exp(0.5 * logvar)
        self.prior_means = mean[:, 0]
        self.prior_scales = std[:, 0]


class AutoregressivePriorDistributionPriorCoder(DistributionPriorCoder):
    """Interface for autoregressive prior
    """    

    def _autoregressive_prior(self, prior : torch.Tensor = None, input_shape : torch.Size = None, posterior_samples : torch.Tensor = None, **kwargs) -> torch.Tensor:
        raise NotImplementedError()

    def _autoregressive_posterior(self, input : torch.Tensor = None, **kwargs) -> torch.Tensor:
        return input

    def _forward_flat(self, input : torch.Tensor, input_shape : torch.Size, prior : torch.Tensor = None, **kwargs):
        posterior_ar = self._autoregressive_posterior(input)
        if self.train_em_update and self.em_state == False:
            posterior_ar = posterior_ar.detach()

        posterior_dist = self.posterior_distribution(posterior_ar)
        samples = self.sample_from_posterior(posterior_dist)

        prior_ar = self._autoregressive_prior(prior=prior, input_shape=input_shape, posterior_samples=samples)

        if self.train_em_update and self.em_state == True:
            prior_ar = posterior_ar.detach()
        prior_dist = self.prior_distribution(prior=prior_ar)

        # TODO: check add posterior_samples?
        KLD = torch.sum(self.kl_divergence(prior_dist, posterior_dist, input_shape=input_shape, posterior_samples=samples))
        if self.training:
            self.update_cache("loss_dict",
                loss_rate=KLD / input_shape[0], # normalize by batch size
            )
        # if implementation has not provide prior_entropy, use kl as prior_entropy instead
        if not "prior_entropy" in self.get_raw_cache("metric_dict"):
            self.update_cache("metric_dict",
                prior_entropy = KLD / input_shape[0], # normalize by batch size
            )

        return self.postprocess_samples(samples)


class AutoregressivePriorImplDistributionPriorCoder(AutoregressivePriorDistributionPriorCoder):
    """Implementation for autoregressive prior
        Args:
            in_channels (int, optional): input channels. Defaults to 256.
            latent_channels ([type], optional): latent channels. If None, default to in_channels // self.num_posterior_params. Defaults to None.
            use_autoregressive_prior (bool, optional): Enable autoregressive model as prior. Defaults to False.
            ar_method (str, optional): Supports finitestate, maskconv3x3/5x5, maskconv3d3x3x3/5x5x5, checkerboard3x3/5x5. Defaults to "finitestate".
            ar_input_detach (bool, optional): Detach autoregressive model input. Use this if you want to train ar model standalone. Defaults to False.
            ar_input_straight_through (bool, optional): Use straight through on autoregressive model input. Defaults to False.
            ar_window_size (int, optional): For finitestate. \
                The channel-dim sliding window size used to get conditional variables. Conflict with ar_offsets. Defaults to 1.
            ar_offsets (List[tuple], optional): Conflict with ar_window_size. Defaults to None.\
                Manually set offsets used to get conditional variables. \
                Tuples start from channel dimension. e.g. (-1, ) gets [:, c-1, h, w], (0, -1, -1) gets [:, c, h-1, w-1].\
                Defaults to 1.
            ar_mlp_per_channel (bool, optional): Use per-channel fsar model instead of a single fsar model. Defaults to False.
            ar_mlp_bottleneck_expansion (int, optional): Expansion ratio for MLP based FSAR. Defaults to 2.
            use_autoregressive_posterior (bool, optional): Beta. Defaults to False.
            posterior_ar_window_size (int, optional): Beta. Defaults to 1.
    """
    def __init__(self, in_channels=256, latent_channels=None, 
                 # basic prior
                 prior_trainable=False,
                 # ar prior
                 use_autoregressive_prior=False, 
                 ar_method="finitestate", ar_input_detach=False, # ar_input_sample=True, ar_input_straight_through=False,
                 ar_window_size=1, ar_offsets : List[tuple] = None,
                 # for MLP based fsar
                 ar_fs_method="table", # deprecated, left for compability
                 ar_mlp_per_channel=False, ar_mlp_bottleneck_expansion=2,
                 # ar posterior
                 use_autoregressive_posterior=False,
                 posterior_ar_window_size=1,
                 **kwargs):
        super().__init__(in_channels, latent_channels, **kwargs)

        self.use_autoregressive_prior = use_autoregressive_prior
        self.ar_method = ar_method
        self.ar_input_detach = ar_input_detach
        # self.ar_input_sample = ar_input_sample
        # self.ar_input_straight_through = ar_input_straight_through
        self.ar_window_size = ar_window_size
        self.ar_offsets = ar_offsets
        # self.ar_fs_method = ar_fs_method
        # self.ar_prior_decomp_method = ar_prior_decomp_method
        # self.ar_prior_decomp_dim = ar_prior_decomp_dim
        self.ar_mlp_per_channel = ar_mlp_per_channel
        # full ar
        if self.ar_window_size is None:
            self.ar_window_size = self.latent_channels - 1
        # custom ar offset setting
        if self.ar_offsets is None:
            self.ar_offsets = [(-offset,) for offset in range(1, self.ar_window_size+1)]
        else:
            self.ar_window_size = len(ar_offsets)

        prior_params = torch.zeros(self.latent_channels, self.num_prior_params)
        if prior_trainable:
            self.prior_params = nn.Parameter(prior_params)
        else:
            self.register_buffer("prior_params", prior_params, persistent=False)

        if use_autoregressive_prior:
            ar_input_channels = self.num_sample_params
            self.ar_input_channels = ar_input_channels
            if self.ar_method == "finitestate":
                if self.ar_mlp_per_channel:
                    self.fsar_mlps_per_channel = nn.ModuleList(
                        [
                            nn.Sequential(
                                nn.Linear(ar_input_channels * self.ar_window_size, int(ar_mlp_bottleneck_expansion * self.ar_window_size * ar_input_channels)),
                                nn.LeakyReLU(),
                                nn.Linear(int(ar_mlp_bottleneck_expansion * self.ar_window_size * ar_input_channels), int(ar_mlp_bottleneck_expansion * self.num_prior_params)),
                                nn.LeakyReLU(),
                                nn.Linear(int(ar_mlp_bottleneck_expansion * self.num_prior_params), self.num_prior_params),
                            )
                            for _ in range(self.latent_channels)
                        ]
                    )
                else:
                    self.fsar_mlp = nn.Sequential(
                        nn.Linear(ar_input_channels * self.ar_window_size, int(ar_mlp_bottleneck_expansion * self.ar_window_size * ar_input_channels)),
                        nn.LeakyReLU(),
                        nn.Linear(int(ar_mlp_bottleneck_expansion * self.ar_window_size * ar_input_channels), int(ar_mlp_bottleneck_expansion * self.num_prior_params)),
                        nn.LeakyReLU(),
                        nn.Linear(int(ar_mlp_bottleneck_expansion * self.num_prior_params), self.num_prior_params),
                    )

        # model based ar
        if self.use_autoregressive_prior:
            ar_model = None
            if self.ar_method == "maskconv3x3":
                ar_model = MaskedConv2d(ar_input_channels * self.latent_channels, self.num_prior_params * self.latent_channels, 3, padding=1)
            elif self.ar_method == "maskconv5x5":
                ar_model = MaskedConv2d(ar_input_channels * self.latent_channels, self.num_prior_params * self.latent_channels, 5, padding=2)
            elif self.ar_method == "maskconv3d3x3x3":
                ar_model = MaskedConv3d(ar_input_channels, self.num_prior_params, 3, padding=1)
            elif self.ar_method == "maskconv3d5x5x5":
                ar_model = MaskedConv3d(ar_input_channels, self.num_prior_params, 5, padding=2)
            elif self.ar_method == "checkerboard3x3":
                ar_model = MaskedConv2d(ar_input_channels * self.latent_channels, self.num_prior_params * self.latent_channels, 3, padding=1, mask_type="Checkerboard")
            elif self.ar_method == "checkerboard5x5":
                ar_model = MaskedConv2d(ar_input_channels * self.latent_channels, self.num_prior_params * self.latent_channels, 5, padding=2, mask_type="Checkerboard")

            if ar_model is not None:
                self.ar_model = nn.Sequential(
                    ar_model,
                    # TODO: append a trans module
                    # nn.Conv2d(ar_input_channels * self.latent_dim * 6 // 3, ar_input_channels * self.latent_dim * 5 // 3, 1),
                    # nn.LeakyReLU(inplace=True),
                    # nn.Conv2d(ar_input_channels * self.latent_dim * 5 // 3, ar_input_channels * self.latent_dim * 4 // 3, 1),
                    # nn.LeakyReLU(inplace=True),
                    # nn.Conv2d(ar_input_channels * self.latent_dim * 4 // 3, ar_input_channels * self.latent_dim * 3 // 3, 1),
                )

        self.use_autoregressive_posterior = use_autoregressive_posterior
        self.posterior_ar_window_size = posterior_ar_window_size

    def _default_sample(self, samples : torch.Tensor = None) -> torch.Tensor:
        return torch.zeros_like(samples)
    
    def _finite_state_to_samples(self, states : torch.Tensor = None) -> torch.Tensor:
        raise NotImplementedError()

    def _autoregressive_prior(self, prior : torch.Tensor = None, input_shape : torch.Size = None, posterior_samples : torch.Tensor = None, **kwargs) -> torch.Tensor:
        if prior is None:
            prior = self.prior_params.unsqueeze(0)
        # TODO: process prior parameter if exists!
        if self.use_autoregressive_prior:
            # if input_shape is None:
            #     input_shape = posterior_dist.logits.shape[:-1]
            assert input_shape is not None
            assert posterior_samples is not None
            batch_size = input_shape[0]
            spatial_shape = input_shape[2:]
            posterior_samples_shape = posterior_samples.shape # N*spatial_dim*C*num_sample_params
            if self.ar_input_detach:
                posterior_samples = posterior_samples.detach()
            if self.ar_method == "finitestate":
                # find samples for ar
                # reshape as input format (N*spatial_dim*C*num_sample_params -> N*C*spatial_dim*num_sample_params)
                posterior_samples_reshape = posterior_samples.reshape(batch_size, *spatial_shape, self.latent_channels, self.num_sample_params).movedim(-2, 1)
                # merge prior logits
                autoregressive_samples = []
                for ar_offset in self.ar_offsets:
                    default_samples = self._default_sample(posterior_samples_reshape)
                    ar_samples = posterior_samples_reshape
                    for data_dim, data_offset in enumerate(ar_offset):
                        if data_offset >= 0: continue
                        batched_data_dim = data_dim + 1
                        assert batched_data_dim != ar_samples.ndim - 1 # ar could not include ar_input_channels
                        ar_samples = torch.cat((
                            default_samples.narrow(batched_data_dim, 0, -data_offset),
                            ar_samples.narrow(batched_data_dim, 0, posterior_samples_reshape.shape[batched_data_dim]+data_offset)
                        ), dim=batched_data_dim)
                    autoregressive_samples.append(ar_samples)
                # [batch_size, self.latent_channels, *spatial_shape, self.ar_window_size*self.ar_input_channels]
                autoregressive_samples = torch.cat(autoregressive_samples, dim=-1)
                if self.ar_mlp_per_channel:
                    autoregressive_samples_per_channel = autoregressive_samples.movedim(1, -2)\
                        .reshape(posterior_samples_shape[0], self.latent_channels, self.ar_window_size*self.ar_input_channels)
                    ar_logits = torch.stack([mlp(sample.squeeze(1)) for mlp, sample in zip(self.fsar_mlps_per_channel, autoregressive_samples_per_channel.split(1, dim=1))], dim=1)
                else:
                    autoregressive_samples_flat = autoregressive_samples.movedim(1, -2).reshape(-1, self.ar_window_size*self.ar_input_channels)
                    ar_logits = self.fsar_mlp(autoregressive_samples_flat)
                    # merge ar logits and prior logits
            else:
                assert len(spatial_shape) == 2
                posterior_samples_reshape = posterior_samples.reshape(batch_size, *spatial_shape, self.latent_channels * self.num_sample_params).movedim(-1, 1)
                if self.ar_method.startswith("maskconv"):
                    if self.ar_method.startswith("maskconv3d"):
                        posterior_samples_reshape = posterior_samples_reshape.reshape(batch_size, self.latent_channels, self.num_sample_params, *spatial_shape)\
                            .permute(0, 2, 1, 3, 4)
                    ar_logits_reshape = self.ar_model(posterior_samples_reshape)
                    if self.ar_method.startswith("maskconv3d"):
                        ar_logits_reshape = ar_logits_reshape.permute(0, 2, 1, 3, 4)\
                            .reshape(batch_size, self.latent_channels, *spatial_shape)
                elif self.ar_method.startswith("checkerboard"):
                    ar_logits_reshape = self.ar_model(posterior_samples_reshape)
                    checkerboard_mask_h_0 = torch.arange(0, spatial_shape[-2], 2, dtype=torch.long, device=posterior_samples_reshape.device)
                    checkerboard_mask_h_1 = torch.arange(1, spatial_shape[-2], 2, dtype=torch.long, device=posterior_samples_reshape.device)
                    checkerboard_mask_w_0 = torch.arange(0, spatial_shape[-1], 2, dtype=torch.long, device=posterior_samples_reshape.device)
                    checkerboard_mask_w_1 = torch.arange(1, spatial_shape[-1], 2, dtype=torch.long, device=posterior_samples_reshape.device)
                    checkerboard_index_h_01, checkerboard_index_w_01 = torch.meshgrid(checkerboard_mask_h_0, checkerboard_mask_w_1)
                    checkerboard_index_h_10, checkerboard_index_w_10 = torch.meshgrid(checkerboard_mask_h_1, checkerboard_mask_w_0)
                    # multi-indexed tensor cannot be used as mutable left value
                    # ar_logits_reshape[..., checkerboard_mask_h_0, :][..., checkerboard_mask_w_1] = prior_dist.logits.reshape(1, self.latent_channels, 1, 1)
                    # ar_logits_reshape[..., checkerboard_mask_h_1, :][..., checkerboard_mask_w_0] = prior_dist.logits.reshape(1, self.latent_channels, 1, 1)
                    # TODO: default prior params?
                    ar_logits_reshape[..., checkerboard_index_h_01, checkerboard_index_w_01] = self.prior_params.reshape(1, -1, 1, 1).repeat(ar_logits_reshape.shape[0], 1, ar_logits_reshape.shape[2]//2, ar_logits_reshape.shape[3]//2)
                    ar_logits_reshape[..., checkerboard_index_h_10, checkerboard_index_w_10] = self.prior_params.reshape(1, -1, 1, 1).repeat(ar_logits_reshape.shape[0], 1, ar_logits_reshape.shape[2]//2, ar_logits_reshape.shape[3]//2)
                else:
                    raise NotImplementedError(f"Unknown self.ar_method {self.ar_method}")
                ar_logits = ar_logits_reshape.movedim(1, -1).reshape(posterior_samples_shape[0], self.latent_channels, self.num_prior_params)
            prior = ar_logits + prior

        return prior


class CategoricalAutoregressivePriorDistributionPriorCoder(AutoregressivePriorImplDistributionPriorCoder):
    """
    VAE with categorical autoregressive prior. Implements FSAR-ANS entropy coding.
    Also implements Learnable State Number.
        Args:
            in_channels (int, optional): input channels. Defaults to 256.
            latent_channels ([type], optional): latent channels. Defaults to 8.
            categorical_dim (int, optional): Number of classes in categorical distribution. Defaults to 128.
            use_sample_kl (bool, optional): Calculate KL using posterior sample instead of posterior distributions. Defaults to False.
            sample_kl_use_log_mixture (bool, optional): Calculate KL using posterior sample log mixture. Defaults to False.
            kl_prior_detach_posterior (bool, optional): Detach posterior (or samples) when calculating KL. Defaults to False.
            use_gs_st_sample (bool, optional): Beta. Defaults to False.
            cat_reduce (bool, optional): Enable learnable category (state) number. Defaults to False.
            cat_reduce_method (str, optional): currently only support entmax. Defaults to "entmax".
            cat_reduce_channel_same (bool, optional): Use same channel mask for all channels. If False, coding will throw NotImplementedError. Defaults to True.
            cat_reduce_logit_init_range (float, optional): [description]. Defaults to 0.1.
            cat_reduce_entmax_alpha_trainable (bool, optional): Use trainable alpha for entmax. Defaults to False.
            cat_reduce_entmax_alpha (float, optional): entmax alpha. Defaults to 1.5.
            cat_reduce_entmax_alpha_min (float, optional): entmax alpha min for trainable/annealed alpha. Defaults to 1.0.
            cat_reduce_entmax_alpha_max (float, optional): entmax alpha max for trainable/annealed alpha. Defaults to 2.0.
            gs_temp (float, optional): Gumbel softmax temperature. Defaults to 0.5.
            gs_temp_anneal (bool, optional): Enable annealing parameter for Gumbel softmax temperature. Defaults to False.
            relax_temp (float, optional): Softmax temperature. Defaults to 0.5.
            relax_temp_anneal (bool, optional): Enable annealing parameter for softmax temperature. Defaults to False.
            entropy_temp (float, optional): Entropy temperature. Defaults to 1.0.
            entropy_temp_anneal (bool, optional): Enable annealing parameter for entropy temperature. Defaults to False.
            entropy_temp_threshold (float, optional): Beta. Defaults to 0.0.
            cat_reduce_temp (float, optional): Beta. Defaults to 1.0.
            cat_reduce_temp_anneal (bool, optional): Beta. Defaults to False.
    """    
    def __init__(self, in_channels=256, latent_channels=8, categorical_dim=128, 
                 # KL
                 use_sample_kl=False, sample_kl_use_log_mixture=False, kl_prior_detach_posterior=False,
                 # sampling
                 use_gs_st_sample=False,
                 # category reduction
                 cat_reduce=False, cat_reduce_method="entmax", cat_reduce_channel_same=True,
                 cat_reduce_logit_init_range=0.1,
                 cat_reduce_entmax_alpha_trainable=False, cat_reduce_entmax_alpha=1.5, cat_reduce_entmax_alpha_min=1.0, cat_reduce_entmax_alpha_max=2.0,
                 # anneal
                 gs_temp=0.5, gs_temp_anneal=False,
                 relax_temp=0.5, relax_temp_anneal=False,
                 entropy_temp=1.0, entropy_temp_anneal=False, entropy_temp_threshold=0.0,
                 cat_reduce_temp=1.0, cat_reduce_temp_anneal=False,
                 **kwargs):
        self.categorical_dim = categorical_dim
        self.use_sample_kl = use_sample_kl
        self.sample_kl_use_log_mixture = sample_kl_use_log_mixture
        self.kl_prior_detach_posterior = kl_prior_detach_posterior
        self.use_gs_st_sample = use_gs_st_sample
        super().__init__(in_channels, latent_channels, **kwargs)

        self.cat_reduce = cat_reduce
        self.cat_reduce_method = cat_reduce_method
        self.cat_reduce_channel_same = cat_reduce_channel_same
        self.cat_reduce_entmax_alpha_trainable = cat_reduce_entmax_alpha_trainable
        self.cat_reduce_entmax_alpha_min = cat_reduce_entmax_alpha_min
        self.cat_reduce_entmax_alpha_max = cat_reduce_entmax_alpha_max
        if self.cat_reduce_entmax_alpha_trainable:
            # inverse sigmoid
            self.cat_reduce_entmax_alpha = nn.Parameter(
                -torch.tensor([(1 / max(cat_reduce_entmax_alpha-self.cat_reduce_entmax_alpha_min, 1e-7) ) - 1]).log()
            )
        else:
            self.cat_reduce_entmax_alpha = cat_reduce_entmax_alpha
        if self.cat_reduce:
            cat_reduce_channel_dim = 1 if cat_reduce_channel_same else self.latent_channels
            cat_reduce_logprob = None
            if self.cat_reduce_method == "entmax":
                cat_reduce_logprob = torch.zeros(cat_reduce_channel_dim, categorical_dim) # - self.cat_reduce_logit_thres
            else:
                raise NotImplementedError(f"Unknown cat_reduce_method {cat_reduce_method}")
            if cat_reduce_logprob is not None:
                self.cat_reduce_logprob = nn.Parameter(cat_reduce_logprob)
                nn.init.uniform_(self.cat_reduce_logprob, -cat_reduce_logit_init_range, cat_reduce_logit_init_range) # add a small variation

        self.gs_temp_anneal = gs_temp_anneal
        if gs_temp_anneal:
            self.gs_temp = nn.Parameter(torch.tensor(gs_temp), requires_grad=False)
        else:
            self.register_buffer("gs_temp", torch.tensor(gs_temp), persistent=False)

        self.relax_temp_anneal = relax_temp_anneal
        if relax_temp_anneal:
            self.relax_temp = nn.Parameter(torch.tensor(relax_temp), requires_grad=False)
        else:
            self.register_buffer("relax_temp", torch.tensor(relax_temp), persistent=False)

        self.entropy_temp_anneal = entropy_temp_anneal
        if entropy_temp_anneal:
            self.entropy_temp = nn.Parameter(torch.tensor(entropy_temp), requires_grad=False)
            # self.register_buffer("entropy_temp_min", torch.tensor(entropy_temp_min, requires_grad=False), persistent=False)
        else:
            self.register_buffer("entropy_temp", torch.tensor(entropy_temp), persistent=False)
            # self.entropy_temp_min = entropy_temp_min
        self.entropy_temp_threshold = entropy_temp_threshold

        self.cat_reduce_temp_anneal = cat_reduce_temp_anneal
        if cat_reduce_temp_anneal:
            self.cat_reduce_temp = nn.Parameter(torch.tensor(cat_reduce_temp), requires_grad=False)
        else:
            self.cat_reduce_temp = cat_reduce_temp

        # TODO: tmp fix! remove this after implementing fsar for rans!
        if self.use_autoregressive_prior and self.ar_method == "finitestate":
            self.coder_type = "tans"
            self.coder_freq_precision = 10

    @property
    def num_posterior_params(self):
        return self.categorical_dim

    @property
    def num_sample_params(self):
        return self.categorical_dim

    def _get_entmax_probs(self):
        if self.cat_reduce_temp_anneal:
            alpha = self.cat_reduce_entmax_alpha_max - self.cat_reduce_temp * (self.cat_reduce_entmax_alpha_max - self.cat_reduce_entmax_alpha_min)
        else:
            if self.cat_reduce_entmax_alpha_trainable:
                alpha = self.cat_reduce_entmax_alpha_min + torch.sigmoid(self.cat_reduce_entmax_alpha) * \
                    (self.cat_reduce_entmax_alpha_max - self.cat_reduce_entmax_alpha_min)
                self.update_cache("metric_dict", 
                    cat_reduce_entmax_alpha=alpha,
                )
            else:
                alpha = self.cat_reduce_entmax_alpha

        if alpha <= 1.0:
            entmax_probs = torch.softmax(self.cat_reduce_logprob, dim=-1)
        else:
            entmax_probs = entmax_bisect(self.cat_reduce_logprob, alpha=alpha, dim=-1)

        return entmax_probs
    
    def _cat_reduce_entmax_probs(self, input):

        entmax_probs = self._get_entmax_probs()
        
        cat_reduce_percentage = (entmax_probs==0).sum() / self.cat_reduce_logprob.numel()
        self.update_cache("metric_dict", 
            cat_reduce_percentage=cat_reduce_percentage,
        )
        
        input_probs = torch.softmax(input, dim=-1) * entmax_probs.unsqueeze(0)
        return input_probs / (input_probs.sum(dim=-1, keepdim=True) + 1e-7) # renormalize

    def _cat_reduce_logits(self, logits):
        if self.cat_reduce_method == "entmax":
            probs = self._cat_reduce_entmax_probs(logits) # .unsqueeze(0)
            return (probs + 1e-9).log() / (probs + 1e-9).sum(dim=-1, keepdim=True)

    def _finite_state_to_samples(self, states: torch.LongTensor, add_default_samples=False) -> torch.Tensor:
        if add_default_samples:
            samples = F.one_hot((states-1).clamp(min=0), self.categorical_dim)
            return torch.where(states > 0 | states <= self.categorical_dim, samples, self._default_sample(samples))
        return F.one_hot(states, self.categorical_dim)

    def _latent_to_finite_state(self, latent: torch.Tensor) -> torch.LongTensor:
        if self.cat_reduce:
            latent = latent.index_select(-1, self._reduce_mask)
        return torch.argmax(latent, dim=-1)

    def prior_distribution(self, prior=None, **kwargs) -> distributions.Categorical:
        if prior is None:
            prior = self.prior_params.unsqueeze(0)
        prior = prior.view(-1, self.latent_channels, self.categorical_dim)
        if self.cat_reduce:
            prior = self._cat_reduce_logits(prior)
        return distributions.Categorical(logits=prior)

    def posterior_distribution(self, latent, **kwargs) -> distributions.RelaxedOneHotCategorical:
        latent_logits = latent.view(-1, self.latent_channels, self.categorical_dim)
        latent_logits = latent_logits / self.relax_temp
        if self.cat_reduce:
            latent_logits = self._cat_reduce_logits(latent_logits)
        return distributions.RelaxedOneHotCategorical(self.gs_temp, logits=latent_logits)
    
    def sample_from_posterior(self, posterior_dist: distributions.RelaxedOneHotCategorical, **kwargs):
        samples = super().sample_from_posterior(posterior_dist, **kwargs)
        if self.use_gs_st_sample:
            one_hot_samples = F.one_hot(samples.argmax(-1), samples.shape[-1])\
                .type_as(samples)
            samples = one_hot_samples + samples - samples.detach()
        return samples

    def kl_divergence(self, 
                      prior_dist: distributions.Categorical, 
                      posterior_dist: distributions.RelaxedOneHotCategorical, 
                      input_shape: torch.Size = None, posterior_samples=None, **kwargs):
        if self.use_sample_kl and posterior_samples is not None:                
            if self.sample_kl_use_log_mixture:
                posterior_entropy = (posterior_samples * posterior_dist.probs).sum(-1).clamp(min=1e-6).log()
                if self.kl_prior_detach_posterior:
                    prior_entropy = (posterior_samples.detach() * prior_dist.probs).sum(-1).clamp(min=1e-6).log()
                else:
                    prior_entropy = (posterior_samples * prior_dist.probs).sum(-1).clamp(min=1e-6).log()
            else:
                posterior_entropy = posterior_samples * posterior_dist.logits # posterior_samples.clamp(min=1e-6).log()
                posterior_entropy[posterior_samples == 0] = 0 # prevent nan
                if self.kl_prior_detach_posterior:
                    prior_entropy = posterior_samples.detach() * prior_dist.logits
                else:
                    prior_entropy = posterior_samples * prior_dist.logits
        else:
            posterior_entropy = posterior_dist.probs * posterior_dist.logits
            posterior_entropy[posterior_dist.probs == 0] = 0 # prevent nan
            if self.kl_prior_detach_posterior:
                prior_entropy = posterior_dist.probs.detach() * prior_dist.logits
            else:
                prior_entropy = posterior_dist.probs * prior_dist.logits

        entropy_temp = self.entropy_temp if self.entropy_temp >= self.entropy_temp_threshold else 0.0
        kld = posterior_entropy * entropy_temp - prior_entropy

        # moniter entropy gap for annealing
        if self.training:
            self.update_cache("moniter_dict", 
                qp_entropy_gap=(posterior_entropy.sum() / prior_entropy.sum()),
            )
            self.update_cache("moniter_dict", 
                posterior_entropy=posterior_entropy.sum(),
            )
            self.update_cache("moniter_dict", 
                prior_self_entropy=(prior_dist.probs * prior_dist.logits).sum(),
            )
            one_hot_samples = F.one_hot(posterior_dist.logits.argmax(-1), posterior_dist.logits.shape[-1])\
                .type_as(posterior_dist.logits)
            self.update_cache("moniter_dict", 
                prior_one_hot_entropy=-(one_hot_samples * prior_dist.logits).sum(),
            )

        if self.gs_temp_anneal:
            if self.training:
                self.update_cache("moniter_dict", 
                    gs_temp=self.gs_temp
                )
        if self.relax_temp_anneal:
            if self.training:
                self.update_cache("moniter_dict", 
                    relax_temp=self.relax_temp
                )
        if self.entropy_temp_anneal:
            if self.training:
                self.update_cache("moniter_dict", 
                    entropy_temp=self.entropy_temp
                )

        return kld

    def _normalize_prior_logits(self, prior_logits):
        if self.cat_reduce:
            if self.cat_reduce_method == "entmax":
                # cat_reduce_logprob = self.cat_reduce_logprob
                # if self.use_autoregressive_prior and self.ar_method == "finitestate" and self.ar_fs_method == "table":
                #     cat_reduce_logprob = self.cat_reduce_logprob.unsqueeze(1)
                prior_logits = self._cat_reduce_logits(prior_logits) # .unsqueeze(0)
        prior_logits = torch.log_softmax(prior_logits, dim=-1)
        return prior_logits

    def _get_ar_params(self, indexes) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if self.use_autoregressive_prior:
            # indices_shape = indexes.shape
            # indices = torch.zeros(*indexes.shape).reshape(indexes.shape[0], indexes.shape[1], -1)
            # channel_indices = torch.arange(indexes.shape[1], dtype=torch.int32).reshape(1, indices.shape[1], 1).expand_as(indices)
            # ar_indices = channel_indices.reshape(*indices_shape).contiguous().numpy()
            ar_indices = np.zeros_like(indexes)
            ar_offsets = create_ar_offsets(indexes.shape, self.ar_offsets)
            return ar_indices, ar_offsets
        else:
            return None, None

    def _encode_transformed(self, input : torch.Tensor, prior : torch.Tensor = None, **kwargs) -> bytes:
        batch_size = input.shape[0]
        channel_size = input.shape[1]
        spatial_shape = input.shape[2:]
        assert channel_size == self.latent_channels * self.num_posterior_params
        
        # posterior_dist = self.posterior_distribution(input.movedim(1, -1).reshape(-1, self.latent_channels, self.categorical_dim))
        # prior_dist = self.prior_distribution(prior=prior)

        # samples = self.sample_from_posterior(posterior_dist)

        # KLD = self.kl_divergence(prior_dist, posterior_dist, input_shape=(batch_size, self.latent_channels, *spatial_shape))

        input = input.movedim(1, -1).view(batch_size, *spatial_shape, self.latent_channels, self.num_sample_params)
        
        # non-finite autoregressive
        data_bytes = b''
        if self.use_autoregressive_prior and self.ar_method != "finitestate":
            samples = self._latent_to_finite_state(input)
            ar_input = self._finite_state_to_samples(samples).type_as(input)\
                .reshape(batch_size, *spatial_shape, self.latent_channels*self.num_sample_params).movedim(-1, 1)
            if self.ar_method.startswith("maskconv"):
                if self.ar_method.startswith("maskconv3d"):
                    ar_input = ar_input.reshape(batch_size, self.latent_channels, self.num_sample_params, *spatial_shape).movedim(2, 1)
                prior_logits_reshape = self.ar_model(ar_input)
                # move batched dimensions to last for correct decoding
                if self.ar_method.startswith("maskconv3d"):
                    prior_logits_reshape = prior_logits_reshape.movedim(0, -1)
                    samples = samples.movedim(0, -2)
                else:
                    prior_logits_reshape = prior_logits_reshape.reshape(batch_size, self.latent_channels, self.num_prior_params, *spatial_shape)
                    prior_logits_reshape = prior_logits_reshape.movedim(0, -1).movedim(0, -1)
                    samples = samples.movedim(0, -2)
                # move categorical dim
                prior_logits_reshape = prior_logits_reshape.movedim(0, -1)
                
                rans_encoder = RansEncoder()

                data = samples.detach().cpu().numpy().astype(np.int32)
                prior_probs = torch.softmax(prior_logits_reshape, dim=-1)
                cdfs = pmf_to_quantized_cdf_batched(prior_probs.reshape(-1, prior_probs.shape[-1]))
                cdfs = cdfs.detach().cpu().numpy().astype(np.int32)

                data = data.reshape(-1)
                indexes = np.arange(len(data), dtype=np.int32)
                cdf_lengths = np.array([len(cdf) for cdf in cdfs])
                offsets = np.zeros(len(indexes)) # [0] * len(indexes)

                with self.profiler.start_time_profile("time_rans_encoder"):
                    data_bytes = rans_encoder.encode_with_indexes_np(
                        data, indexes,
                        cdfs, cdf_lengths, offsets
                    )

            elif self.ar_method.startswith("checkerboard"):
                samples = samples.movedim(-1, 1) # move latent channels after batch
                prior_logits_reshape = self.ar_model(ar_input)
                checkerboard_mask_h_0 = torch.arange(0, spatial_shape[-2], 2, dtype=torch.long, device=input.device)
                checkerboard_mask_h_1 = torch.arange(1, spatial_shape[-2], 2, dtype=torch.long, device=input.device)
                checkerboard_mask_w_0 = torch.arange(0, spatial_shape[-1], 2, dtype=torch.long, device=input.device)
                checkerboard_mask_w_1 = torch.arange(1, spatial_shape[-1], 2, dtype=torch.long, device=input.device)
                # input_base = torch.cat([
                #     ar_input[..., checkerboard_mask_h_0, :][..., checkerboard_mask_w_1],
                #     ar_input[..., checkerboard_mask_h_1, :][..., checkerboard_mask_w_0],
                # ], dim=-1)
                # input_ar = torch.cat([
                #     ar_input[..., checkerboard_mask_h_0, :][..., checkerboard_mask_w_0],
                #     ar_input[..., checkerboard_mask_h_1, :][..., checkerboard_mask_w_1],
                # ], dim=-1)
                prior_logits_ar = torch.cat([
                    prior_logits_reshape[..., checkerboard_mask_h_0, :][..., checkerboard_mask_w_0],
                    prior_logits_reshape[..., checkerboard_mask_h_1, :][..., checkerboard_mask_w_1],
                ], dim=-1)
                prior_logits_ar = prior_logits_ar.reshape(batch_size, self.latent_channels, self.num_prior_params, *prior_logits_ar.shape[-2:]).movedim(2, -1)

                samples_base = torch.cat([
                    samples[..., checkerboard_mask_h_0, :][..., checkerboard_mask_w_1],
                    samples[..., checkerboard_mask_h_1, :][..., checkerboard_mask_w_0],
                ], dim=-1)
                data_base = samples_base.detach().cpu().numpy()
                indexes_base = torch.arange(self.latent_channels).unsqueeze(0).unsqueeze(-1)\
                    .repeat(batch_size, 1, np.prod(spatial_shape) // 2).reshape_as(samples_base).numpy()
                samples_ar = torch.cat([
                    samples[..., checkerboard_mask_h_0, :][..., checkerboard_mask_w_0],
                    samples[..., checkerboard_mask_h_1, :][..., checkerboard_mask_w_1],
                ], dim=-1)
                data_ar = samples_ar.detach().cpu().numpy()
                
                # prepare for coding (base)
                data_base = data_base.astype(np.int32).reshape(-1)
                indexes_base = indexes_base.astype(np.int32).reshape(-1)
                cdfs_base = self._prior_cdfs
                cdf_sizes_base = np.array([len(cdf) for cdf in self._prior_cdfs])
                offsets_base = np.zeros(len(self._prior_cdfs))

                # prepare for coding (ar)
                prior_probs_ar = torch.softmax(prior_logits_ar, dim=-1)
                cdfs_ar = pmf_to_quantized_cdf_batched(prior_probs_ar.reshape(-1, prior_probs_ar.shape[-1]))
                cdfs_ar = cdfs_ar.detach().cpu().numpy().astype(np.int32)
                data_ar = data_ar.reshape(-1)
                indexes_ar = np.arange(len(data_ar), dtype=np.int32)
                cdf_sizes_ar = np.array([len(cdf) for cdf in cdfs_ar])
                offsets_ar = np.zeros(len(indexes_ar)) # [0] * len(indexes)

                rans_encoder = BufferedRansEncoder()
                with self.profiler.start_time_profile("time_rans_encoder"):
                    rans_encoder.encode_with_indexes_np(
                        data_base, indexes_base,
                        cdfs_base, cdf_sizes_base, offsets_base
                    )
                    rans_encoder.encode_with_indexes_np(
                        data_ar, indexes_ar,
                        cdfs_ar, cdf_sizes_ar, offsets_ar
                    )
                    data_bytes = rans_encoder.flush()
            else:
                pass

        
        if len(data_bytes) == 0:

            # TODO: use iterative autoregressive for overwhelmed states
            if self.use_autoregressive_prior and self.ar_method == "finitestate" and len(self.ar_offsets) > 2:
                raise NotImplementedError("Overwhelmed states!")

            samples = self._latent_to_finite_state(input).movedim(-1, 1) # move latent channels back
            data = samples.contiguous().detach().cpu().numpy().astype(np.int32)
            # self._samples_cache = samples
            indexes = torch.arange(self.latent_channels).unsqueeze(0).unsqueeze(-1)\
                .repeat(batch_size, 1, np.prod(spatial_shape)).reshape_as(samples)\
                .contiguous().numpy().astype(np.int32)

            ar_indexes, ar_offsets = self._get_ar_params(indexes)
            
            data_bytes = self._encoder.encode_with_indexes(
                data, indexes, ar_indexes=ar_indexes, ar_offsets=ar_offsets,
            )

        if len(data_bytes) == 0:
            return b''

        # store sample shape in header
        byte_strings = []
        if self.fixed_input_shape is not None:
            assert batch_size == self.fixed_input_shape[0]
            assert spatial_shape == self.fixed_input_shape[1:]
        else:
            byte_head = [struct.pack("B", len(spatial_shape)+1)]
            byte_head.append(struct.pack("<H", batch_size))
            for dim in spatial_shape:
                byte_head.append(struct.pack("<H", dim))
            byte_strings.extend(byte_head)
        byte_strings.append(data_bytes)
        return b''.join(byte_strings)

    def _decode_transformed(self, byte_string : bytes, prior : torch.Tensor = None, **kwargs) -> torch.Tensor:
        # if len(byte_string) == 0:
        #     return torch.zeros(1, self.latent_channels*self.categorical_dim, 8, 8, device=self.device)

        # decode shape from header
        if self.fixed_input_shape is not None:
            byte_ptr = 0
            batch_dim = self.fixed_input_shape[0]
            spatial_shape = self.fixed_input_shape[1:]
            spatial_dim = np.prod(spatial_shape)
        else:
            num_shape_dims = struct.unpack("B", byte_string[:1])[0]
            flat_shape = []
            byte_ptr = 1
            for _ in range(num_shape_dims):
                flat_shape.append(struct.unpack("<H", byte_string[byte_ptr:(byte_ptr+2)])[0])
                byte_ptr += 2
            batch_dim = flat_shape[0]
            spatial_shape = flat_shape[1:]
            spatial_dim = np.prod(spatial_shape)

        if self.use_autoregressive_prior and self.ar_method != "finitestate":
            if self.ar_method.startswith("maskconv"):
                rans_decoder = RansDecoder()
                rans_decoder.set_stream(byte_string[byte_ptr:])
                samples = torch.zeros(batch_dim, *spatial_shape, self.latent_channels, dtype=torch.long, device=self.device)

                assert len(spatial_shape) == 2
                if self.ar_method.startswith("maskconv3d"):
                    c, h, w = (self.latent_channels, *spatial_shape)
                    for c_idx in range(c):
                        for h_idx in range(h):
                            for w_idx in range(w):
                                ar_input = self._finite_state_to_samples(samples).float().movedim(-1, 1)
                                prior_logits_ar = self.ar_model(ar_input).movedim(1, -1)[:, h_idx, w_idx, c_idx]
                                prior_probs_ar = torch.softmax(prior_logits_ar, dim=-1)
                                cdfs_ar = pmf_to_quantized_cdf_batched(prior_probs_ar.reshape(-1, prior_probs_ar.shape[-1]))
                                cdfs_ar = cdfs_ar.detach().cpu().numpy().astype(np.int32)
                                indexes_ar = np.arange(len(cdfs_ar), dtype=np.int32)
                                cdf_sizes_ar = np.array([len(cdf) for cdf in cdfs_ar])
                                offsets_ar = np.zeros(len(indexes_ar)) # [0] * len(indexes)

                                samples_ar = rans_decoder.decode_stream_np(
                                    indexes_ar, cdfs_ar, cdf_sizes_ar, offsets_ar
                                )
                                samples_ar = torch.as_tensor(samples_ar, dtype=torch.long, device=self.device)
                                samples[:, h_idx, w_idx, c_idx] = samples_ar
                else:
                    h, w = spatial_shape
                    error_flag = False
                    for h_idx in range(h):
                        for w_idx in range(w):
                                ar_input = self._finite_state_to_samples(samples).float().reshape(batch_dim, *spatial_shape, self.latent_channels*self.num_sample_params).movedim(-1, 1)
                                prior_logits_ar = self.ar_model(ar_input).reshape(batch_dim, self.latent_channels, self.num_prior_params, *spatial_shape).movedim(2, -1)[:, :, h_idx, w_idx, :]
                                prior_probs_ar = torch.softmax(prior_logits_ar, dim=-1)
                                cdfs_ar = pmf_to_quantized_cdf_batched(prior_probs_ar.reshape(-1, prior_probs_ar.shape[-1]))
                                cdfs_ar = cdfs_ar.detach().cpu().numpy().astype(np.int32)
                                indexes_ar = np.arange(len(cdfs_ar), dtype=np.int32)
                                cdf_sizes_ar = np.array([len(cdf) for cdf in cdfs_ar])
                                offsets_ar = np.zeros(len(indexes_ar)) # [0] * len(indexes)

                                samples_ar = rans_decoder.decode_stream_np(
                                    indexes_ar, cdfs_ar, cdf_sizes_ar, offsets_ar
                                )
                                samples_ar = torch.as_tensor(samples_ar, dtype=torch.long, device=self.device).reshape(-1, self.latent_channels)
                                if samples_ar.max() >= self.categorical_dim or samples_ar.min() < 0:
                                    # NOTE: early exit to avoid gpu indicing error!
                                    print("Decode error detected! The decompressed data may be corrupted!")
                                    error_flag = True
                                    break
                                samples[:, h_idx, w_idx, :] = samples_ar
                        if error_flag:
                            break

                # warn about decoding error and fixit!
                if samples.max() >= self.categorical_dim or samples.min() < 0:
                    print("Decode error detected! The decompressed data may be corrupted!")
                    samples.clamp_max_(self.categorical_dim-1).clamp_min_(0)
                samples = self._finite_state_to_samples(samples.movedim(1, -1)).float()
                samples = samples.reshape(batch_dim, *spatial_shape, self.latent_channels*self.num_sample_params)\
                    .movedim(-1, 1)

                return samples

            elif self.ar_method.startswith("checkerboard"):
                assert len(spatial_shape) == 2
                checkerboard_mask_h_0 = torch.arange(0, spatial_shape[-2], 2, dtype=torch.long, device=self.device)
                checkerboard_mask_h_1 = torch.arange(1, spatial_shape[-2], 2, dtype=torch.long, device=self.device)
                checkerboard_mask_w_0 = torch.arange(0, spatial_shape[-1], 2, dtype=torch.long, device=self.device)
                checkerboard_mask_w_1 = torch.arange(1, spatial_shape[-1], 2, dtype=torch.long, device=self.device)
                checkerboard_index_h_00, checkerboard_index_w_00 = torch.meshgrid(checkerboard_mask_h_0, checkerboard_mask_w_0)
                checkerboard_index_h_11, checkerboard_index_w_11 = torch.meshgrid(checkerboard_mask_h_1, checkerboard_mask_w_1)
                checkerboard_index_h_01, checkerboard_index_w_01 = torch.meshgrid(checkerboard_mask_h_0, checkerboard_mask_w_1)
                checkerboard_index_h_10, checkerboard_index_w_10 = torch.meshgrid(checkerboard_mask_h_1, checkerboard_mask_w_0)

                rans_decoder = RansDecoder()
                rans_decoder.set_stream(byte_string[byte_ptr:])
                indexes_base = torch.arange(self.latent_channels).unsqueeze(0).unsqueeze(-1)\
                    .repeat(batch_dim, 1, spatial_dim // 2).reshape(batch_dim, self.latent_channels, spatial_shape[0] // 2, spatial_shape[1])\
                    .numpy()

                # prepare for coding
                indexes_base = indexes_base.astype(np.int32).reshape(-1)
                cdfs_base = self._prior_cdfs
                cdf_sizes_base = np.array([len(cdf) for cdf in self._prior_cdfs])
                offsets_base = np.zeros(len(self._prior_cdfs))

                samples = torch.zeros(batch_dim, self.latent_channels, *spatial_shape, dtype=torch.long, device=self.device)
                with self.profiler.start_time_profile("time_rans_decoder"):
                    samples_base = rans_decoder.decode_stream_np(
                        indexes_base, cdfs_base, cdf_sizes_base, offsets_base
                    )
                    samples_base = torch.as_tensor(samples_base, dtype=torch.long, device=self.device)\
                        .reshape(batch_dim, self.latent_channels, spatial_shape[0] // 2, spatial_shape[1])
                    samples[..., checkerboard_index_h_01, checkerboard_index_w_01] = samples_base[..., :(spatial_shape[-1]//2)]
                    samples[..., checkerboard_index_h_10, checkerboard_index_w_10] = samples_base[..., (spatial_shape[-1]//2):]
                    ar_input = self._finite_state_to_samples(samples.movedim(1, -1)).float()
                    ar_input = ar_input.reshape(batch_dim, *spatial_shape, self.latent_channels*self.num_sample_params)\
                        .movedim(-1, 1)
                    
                    prior_logits_reshape = self.ar_model(ar_input)
                    prior_logits_ar = torch.cat([
                        prior_logits_reshape[..., checkerboard_mask_h_0, :][..., checkerboard_mask_w_0],
                        prior_logits_reshape[..., checkerboard_mask_h_1, :][..., checkerboard_mask_w_1],
                    ], dim=-1)
                    prior_logits_ar = prior_logits_ar.reshape(batch_dim, self.latent_channels, self.num_prior_params, *prior_logits_ar.shape[-2:]).movedim(2, -1)
                    
                    # prepare for coding (ar)
                    # NOTE: coding may be unstable on GPU!
                    prior_probs_ar = torch.softmax(prior_logits_ar, dim=-1)
                    cdfs_ar = pmf_to_quantized_cdf_batched(prior_probs_ar.reshape(-1, prior_probs_ar.shape[-1]))
                    cdfs_ar = cdfs_ar.detach().cpu().numpy().astype(np.int32)
                    data_length = samples[..., checkerboard_mask_h_0, :][..., checkerboard_mask_w_0].numel() + samples[..., checkerboard_mask_h_1, :][..., checkerboard_mask_w_1].numel()
                    indexes_ar = np.arange(data_length, dtype=np.int32)
                    cdf_sizes_ar = np.array([len(cdf) for cdf in cdfs_ar])
                    offsets_ar = np.zeros(len(indexes_ar)) # [0] * len(indexes)

                    samples_ar = rans_decoder.decode_stream_np(
                        indexes_ar, cdfs_ar, cdf_sizes_ar, offsets_ar
                    )
                    samples_ar = torch.as_tensor(samples_ar, dtype=torch.long, device=self.device)\
                        .reshape(batch_dim, self.latent_channels, spatial_shape[0] // 2, spatial_shape[1])
                    samples[..., checkerboard_index_h_00, checkerboard_index_w_00] = samples_ar[..., :(spatial_shape[-1]//2)]
                    samples[..., checkerboard_index_h_11, checkerboard_index_w_11] = samples_ar[..., (spatial_shape[-1]//2):]

                # warn about decoding error and fixit!
                if samples.max() >= self.categorical_dim or samples.min() < 0:
                    print("Decode error detected! The decompressed data may be corrupted!")
                    samples.clamp_max_(self.categorical_dim-1).clamp_min_(0)
                samples = self._finite_state_to_samples(samples.movedim(1, -1)).float()
                samples = samples.reshape(batch_dim, *spatial_shape, self.latent_channels*self.num_sample_params)\
                    .movedim(-1, 1)

                return samples

            else:
                pass

        # TODO: use iterative autoregressive for overwhelmed states
        if self.use_autoregressive_prior and self.ar_method == "finitestate" and len(self.ar_offsets) > 2:
            raise NotImplementedError("Overwhelmed states!")

        indexes = torch.arange(self.latent_channels).unsqueeze(0).unsqueeze(-1)\
            .repeat(batch_dim, 1, spatial_dim).reshape(batch_dim, self.latent_channels, *spatial_shape)\
            .contiguous().numpy().astype(np.int32)

        ar_indexes, ar_offsets = self._get_ar_params(indexes)
        
        samples = self._decoder.decode_with_indexes(
            byte_string[byte_ptr:], indexes, ar_indexes=ar_indexes, ar_offsets=ar_offsets,
        )

        samples = torch.as_tensor(samples, dtype=torch.long, device=self.device)\
            .reshape(batch_dim, self.latent_channels, *spatial_shape)
        # assert (samples == self._samples_cache).all()

        # cat_reduce transform
        if self.cat_reduce:
            samples = self._reduce_mask[samples]

        # merge categorical dim back to latent dim
        # samples = F.one_hot(samples.movedim(1, -1), self.categorical_dim).float()
        samples = self._finite_state_to_samples(samples.movedim(1, -1)).float()
        samples = samples.reshape(batch_dim, *spatial_shape, self.latent_channels*self.num_sample_params)\
            .movedim(-1, 1)

        return samples

    def update_state(self, *args, **kwargs) -> None:
        with torch.no_grad():
            if self.prior_trainable:
                prior_logits = self._normalize_prior_logits(self.prior_params)#.unsqueeze(-1)
            else:
                prior_logits = (torch.ones(self.latent_channels, self.categorical_dim) / self.categorical_dim).log().to(device=self.device)
            
            categorical_dim = self.categorical_dim # cat reduce moved after fsar
            if self.use_autoregressive_prior and self.ar_method == "finitestate":
                # TODO: this is a hard limit! may could be improved!
                if len(self.ar_offsets) > 2:
                    return
                else:
                    lookup_table_shape = [self.latent_channels] + [categorical_dim+1] * len(self.ar_offsets) + [categorical_dim]
                    ar_idx_all = list(itertools.product(range(self.categorical_dim+1), repeat=self.ar_window_size))
                    ar_idx_all = torch.tensor(ar_idx_all, device=self.device).reshape(-1, 1).repeat(1, self.latent_channels)
                    ar_input_all = self._finite_state_to_samples(ar_idx_all, add_default_samples=True).type_as(prior_logits)\
                        .reshape(-1, self.ar_window_size, self.latent_channels, self.num_sample_params).movedim(1, -2)\
                        .reshape(-1, self.latent_channels, self.ar_window_size*self.num_sample_params).movedim(1, 0)
                    if self.ar_mlp_per_channel:
                        ar_logits_reshape = torch.stack([mlp(ar_input) for (mlp, ar_input) in zip(self.fsar_mlps_per_channel, ar_input_all)], dim=0)
                    else:
                        ar_logits_reshape = self.fsar_mlp(ar_input_all)
                    prior_logits = prior_logits.unsqueeze(-2) + ar_logits_reshape
                    prior_logits = self._normalize_prior_logits(prior_logits)
                    prior_logits = prior_logits.reshape(*lookup_table_shape)

            prior_pmfs = None

            if self.cat_reduce:
                if self.cat_reduce_method == "entmax":
                    if not self.cat_reduce_channel_same:
                        # TODO: different transformation for different channels
                        raise NotImplementedError()
                        # pass
                    else:
                        reduce_mask = (self._get_entmax_probs()[0] > 0).nonzero(as_tuple=False).squeeze(-1)
                        categorical_dim = reduce_mask.shape[0]
                        prior_logits_reduced = prior_logits
                        if self.use_autoregressive_prior and self.ar_method == "finitestate":
                            # if self.ar_prior_decomp_dim is None:
                            #     prior_logits = prior_logits.reshape(self.latent_channels, self.ar_window_size, self.categorical_dim, self.categorical_dim)
                            # else:
                            #     prior_logits = prior_logits.reshape(self.latent_channels, self.ar_window_size, self.ar_prior_decomp_dim, self.categorical_dim, self.categorical_dim)
                            # prior_logits = prior_logits.index_select(-2, reduce_mask)
                            prior_logits_reduced = prior_logits_reduced.index_select(-1, reduce_mask)
                            for i in range(-2, -len(self.ar_offsets)-2, -1):
                                reduce_mask_ar = torch.cat([torch.zeros(1).type_as(reduce_mask), reduce_mask+1], dim=0)
                                prior_logits_reduced = prior_logits_reduced.index_select(i, reduce_mask_ar)
                        prior_logits = torch.log_softmax(prior_logits_reduced, dim=-1)
                        # self._reduce_mask = reduce_mask
                        self.register_buffer("_reduce_mask", reduce_mask, persistent=False)

            if prior_pmfs is None:
                prior_pmfs = prior_logits.exp()

            # TODO: customize freq precision
            if self.coder_type == "rans" or self.coder_type == "rans64":
                # for compability
                self._prior_cdfs = pmf_to_quantized_cdf_serial(prior_pmfs.reshape(-1, categorical_dim))
                self._encoder = Rans64Encoder(freq_precision=self.coder_freq_precision)
                self._decoder = Rans64Decoder(freq_precision=self.coder_freq_precision)
            elif self.coder_type == "tans":
                self._encoder = TansEncoder(table_log=self.coder_freq_precision, max_symbol_value=categorical_dim-1)
                self._decoder = TansDecoder(table_log=self.coder_freq_precision, max_symbol_value=categorical_dim-1)
            else:
                raise NotImplementedError(f"Unknown coder_type {self.coder_type}!")

            prior_cnt = (prior_pmfs * (1<<self.coder_freq_precision)).clamp_min(1).reshape(-1, categorical_dim)
            prior_cnt = prior_cnt.detach().cpu().numpy().astype(np.int32)
            num_symbols = np.zeros(len(prior_cnt), dtype=np.int32) + categorical_dim
            offsets = np.zeros(len(prior_cnt), dtype=np.int32)

            self._encoder.init_params(prior_cnt, num_symbols, offsets)
            self._decoder.init_params(prior_cnt, num_symbols, offsets)

            if self.use_autoregressive_prior and self.ar_method == "finitestate":
                ar_indexes = np.arange(len(prior_cnt), dtype=np.int32).reshape(1, *prior_pmfs.shape[:-1])

                self._encoder.init_ar_params(ar_indexes, [self.ar_offsets])
                self._decoder.init_ar_params(ar_indexes, [self.ar_offsets])


class StochasticVQAutoregressivePriorDistributionPriorCoder(CategoricalAutoregressivePriorDistributionPriorCoder):
    """Advanced VQ-based categorical VAE. Implements SQVAE, straight-through hardmax, EMA-VQ, EM-VQ and many other VQ-variants.

        Args:
            in_channels (int, optional): input channels. Defaults to 256.
            latent_channels ([type], optional): latent channels. Defaults to 8.
            categorical_dim (int, optional): Number of classes in categorical distribution. Defaults to 128.
            embedding_dim (int, optional): Dimension of vector codes . Defaults to 32.
            force_hardmax (bool, optional): Force hardmax quantization regardless of gradient issue. Defaults to False.
            use_st_hardmax (bool, optional): Use straight-through hardmax quantization (STHQ) for VQ. Defaults to False.
            hardmax_st_use_logits (bool, optional): Use probs instead of weighted samples for STHQ. Defaults to False.
            force_st (bool, optional): Force straight through on output quantized samples. Defaults to False.
            st_weight (float, optional): If force_st, straight through gradient weight on output quantized samples. Defaults to 1.0.
            use_st_below_entropy_threshold (bool, optional): If entropy_temp < entropy_temp_threshold, use straight through on output quantized samples. \
                Enabling force_st will override this. Defaults to False.
            channels_share_codebook (bool, optional): If all channels share the same codebook. \
                If True, the codebook size is [1, categorical_dim, embedding_dim]\
                If False, the codebook size is [latent_channels, categorical_dim, embedding_dim]. Defaults to False.
            fix_embedding (bool, optional): Do not train codebook. Defaults to False.
            ema_update_embedding (bool, optional): Use EMA-based codebook update. Defaults to False.
            ema_decay (float, optional): EMA decay. Defaults to 0.999.
            ema_epsilon ([type], optional): EMA epsilon. Defaults to 1e-5.
            train_em_mstep_samples (int, optional): Multisample size for EM-VQ training M-step. Defaults to 1.
            initialization_scale (float, optional): Scale for codebook initialization (normal init). Defaults to 1.0.
            one_hot_initialization (bool, optional): Use one hot initialization for codebook. Will override self.embedding_dim = categorical_dim. Defaults to False.
            embedding_init_method (str, optional): Specified initialization for codebook, could be "uniform" or "normal". Defaults to "uniform".
            embedding_variance (float, optional): Variance for gaussian-based codebook (for SQVAE). Defaults to 1.0.
            embedding_variance_per_channel (bool, optional): Enable per-channel variance. Defaults to False.
            embedding_variance_trainable (bool, optional): Trainable codebook variance. Defaults to True.
            embedding_variance_lr_modifier (float, optional): Learning rate multiplier for trainable codebook variance. Defaults to 1.0.
            distance_method (str, optional): How to calculate vector code distances. Defaults to "gaussian".
            cont_loss_weight (float, optional): continous loss (distance between input and quantized output) weight. Defaults to 1.0.
            seperate_update_cont_loss (bool, optional): Use VQ-VAE style continous loss. Defaults to False.
            vq_loss_weight (float, optional): For VQ-VAE style continous loss vq weight. Defaults to 1.0.
            commit_loss_weight (float, optional): For VQ-VAE style continous loss commitment weight. Defaults to 1.0.
            var_scale (float, optional): Annealing parameter for scaling embedding_variance. Defaults to 1.0.
            var_scale_anneal (bool, optional): Enable annealing parameter for scaling embedding_variance. Defaults to False.
    """    
    def __init__(self, in_channels=256, latent_channels=8, categorical_dim=128, embedding_dim=32,
                 # quantization
                 force_hardmax=False, use_st_hardmax=False, hardmax_st_use_logits=False,
                 force_st=False, st_weight=1.0, use_st_below_entropy_threshold=False,
                 # codebook
                 channels_share_codebook=False,
                 fix_embedding=False, ema_update_embedding=False, ema_decay=0.999, ema_epsilon=1e-5,
                 train_em_mstep_samples=1,
                 initialization_scale=1.0, # NOTE: for compability
                 one_hot_initialization=False, embedding_init_method="uniform",
                 # sqvae params
                 embedding_variance=1.0, embedding_variance_per_channel=False,
                 embedding_variance_trainable=True, embedding_variance_lr_modifier=1.0,
                 distance_method="gaussian", 
                 cont_loss_weight=1.0, seperate_update_cont_loss=False, 
                 vq_loss_weight=1.0, commit_loss_weight=1.0,
                 relax_temp=1.0, # fix default value
                 var_scale=1.0, var_scale_anneal=False,
                 **kwargs):
       
        self.embedding_dim = embedding_dim
        self.channels_share_codebook = channels_share_codebook
        self.force_hardmax = force_hardmax
        self.use_st_hardmax = use_st_hardmax
        self.hardmax_st_use_logits = hardmax_st_use_logits
        self.force_st = force_st
        self.st_weight = st_weight
        self.use_st_below_entropy_threshold = use_st_below_entropy_threshold
        
        self.distance_method = distance_method
        self.cont_loss_weight = cont_loss_weight
        self.seperate_update_cont_loss = seperate_update_cont_loss
        self.vq_loss_weight = vq_loss_weight
        self.commit_loss_weight = commit_loss_weight

        self.train_em_mstep_samples = train_em_mstep_samples

        super().__init__(in_channels, latent_channels, categorical_dim, relax_temp=relax_temp, **kwargs)
        
        embedding = torch.zeros(1 if self.channels_share_codebook else latent_channels, categorical_dim, embedding_dim)
        if one_hot_initialization:
            self.embedding_dim = categorical_dim # force embedding dim equal to categorical_dim
            embedding = torch.eye(categorical_dim).unsqueeze(0).repeat(latent_channels, 1, 1)
        else:
            if embedding_init_method == "normal":
                nn.init.normal_(embedding, 0, initialization_scale)
            else:
                nn.init.uniform_(embedding, -initialization_scale, initialization_scale)
        
        if initialization_scale is None:
            initialization_scale = 1/categorical_dim
        embedding.uniform_(-initialization_scale, initialization_scale)

        self.fix_embedding = fix_embedding
        self.ema_update_embedding = ema_update_embedding
        self.ema_decay = ema_decay
        self.ema_epsilon = ema_epsilon
        
        if self.fix_embedding:
            self.register_buffer("embedding", embedding)
        else:
            if self.ema_update_embedding:
                self.register_buffer("embedding", embedding)
                self.register_buffer("ema_count", torch.zeros(latent_channels, categorical_dim))
                self.register_buffer("ema_weight", self.embedding.clone())
            else:
                self.embedding = nn.Parameter(embedding)

        self.embedding_variance_trainable = embedding_variance_trainable
        self.embedding_variance_per_channel = embedding_variance_per_channel
        if embedding_variance > 0:
            if self.embedding_variance_per_channel:
                embedding_variance = torch.ones(self.latent_channels) * np.log(embedding_variance) # exponential reparameterization
            else:
                embedding_variance = torch.ones(1) * np.log(embedding_variance) # exponential reparameterization
            if embedding_variance_trainable:
                self.embedding_variance = nn.Parameter(embedding_variance)
                self.embedding_variance.lr_modifier = embedding_variance_lr_modifier
            else:
                self.register_buffer("embedding_variance", embedding_variance)
        else:
            self.embedding_variance = 1e-6

        self.var_scale_anneal = var_scale_anneal
        if var_scale_anneal:
            self.var_scale = nn.Parameter(torch.tensor(var_scale), requires_grad=False)
        else:
            self.var_scale = var_scale

        if self.use_autoregressive_posterior:
            self.posterior_ar_model = nn.Sequential(
                        nn.Linear(2 * self.embedding_dim, 3 * self.embedding_dim),
                        nn.LeakyReLU(),
                        nn.Linear(3 * self.embedding_dim, 2 * self.embedding_dim),
                        nn.LeakyReLU(),
                        nn.Linear(2 * self.embedding_dim, self.embedding_dim),
            )

    def _get_embedding_variance(self):
        return self.embedding_variance.exp() * self.var_scale # exponential reparameterization

    def _calc_distances(self, codewords, codebook, embedding_variance=None):
        if embedding_variance is None:
            embedding_variance = self._get_embedding_variance()
        if self.embedding_variance_per_channel:
            embedding_variance = embedding_variance.view(1, -1, 1, 1)
        if self.distance_method == "gaussian":
            # distance = ((codewords - codebook) ** 2).sum(-1) / embedding_variance
            distance = torch.sum(codewords**2, dim=-1, keepdim=True) \
                + torch.sum(codebook**2, dim=-1).unsqueeze(-2) \
                - 2 * torch.matmul(codewords, codebook.transpose(-2, -1))
            distance = distance / embedding_variance / 2
        elif self.distance_method == "vmf":
            # embedding_variance = self.embedding_variance.exp()
            codewords = F.normalize(codewords, p=2.0, dim=-1)
            codebook = F.normalize(codebook, p=2.0, dim=-1)
            distance = torch.matmul(codewords, codebook.transpose(-2, -1)) / embedding_variance
        else:
            raise NotImplementedError(f"Unknown distance method {self.distance_method}")

        if self.training:
            if isinstance(embedding_variance, torch.Tensor):
                self.update_cache("moniter_dict",
                    embedding_variance_mean = embedding_variance.mean(),
                )

        return distance

    def _calc_dist_logits(self, codewords, codebook, embedding_variance=None):
        distance = self._calc_distances(codewords, codebook, embedding_variance=embedding_variance)
        return torch.log_softmax(-distance, dim=-1)

    def _calc_cont_loss(self, latent, samples):
        embedding_variance = self._get_embedding_variance()
        if self.embedding_variance_per_channel:
            latent = latent.view(-1, self.latent_channels, self.embedding_dim)
            samples = samples.view(-1, self.latent_channels, self.embedding_dim)
            embedding_variance = embedding_variance.view(1, -1, 1)
        if self.distance_method == "gaussian":
            return torch.sum((latent - samples) ** 2 / embedding_variance / 2, dim=-1)
        elif self.distance_method == "vmf":
            # embedding_variance = self.embedding_variance.exp()
            latent = F.normalize(latent, p=2.0, dim=-1)
            samples = F.normalize(samples, p=2.0, dim=-1)
            return torch.sum(latent * (latent - samples) / embedding_variance, dim=-1)
        else:
            raise NotImplementedError(f"Unknown distance method {self.distance_method}")

    def _ema_update_embedding(self, input, samples):
            with torch.no_grad():
                input = input.view(-1, self.latent_channels, self.embedding_dim)
                samples = samples.view(-1, self.latent_channels, self.categorical_dim)
                total_count = samples.sum(dim=0)
                dw = torch.bmm(samples.permute(1, 2, 0), input.permute(1, 0, 2))
                if distributed.is_initialized():
                    distributed.all_reduce(total_count)
                    distributed.all_reduce(dw)
                self.ema_count = self.ema_decay * self.ema_count + (1 - self.ema_decay) * total_count
                n = torch.sum(self.ema_count, dim=-1, keepdim=True)
                self.ema_count = (self.ema_count + self.ema_epsilon) / (n + self.categorical_dim * self.ema_epsilon) * n
                self.ema_weight = self.ema_decay * self.ema_weight + (1 - self.ema_decay) * dw
                self.embedding = self.ema_weight / self.ema_count.unsqueeze(-1)

    @property
    def num_sample_params(self):
        return self.embedding_dim
    
    @property
    def num_posterior_params(self):
        return self.embedding_dim

    @property
    def num_prior_params(self):
        return self.categorical_dim

    def _finite_state_to_samples(self, states: torch.LongTensor, add_default_samples=False) -> torch.Tensor:
        # TODO: use indexing to optimize! 
        def _index_samples(states):
            # NOTE: this is just a try to accelerate! not sure which is faster.
            # or is batched index select possible?
            # if self.latent_channels > 10:
            #     samples = F.one_hot(states, self.categorical_dim).type_as(self.embedding)
            #     samples = torch.sum(samples.unsqueeze(-1) * self.embedding.unsqueeze(0), dim=-2)
            # else:
            #     samples = torch.stack([v.index_select(0, s.reshape(-1)) for s, v in zip(states.split(1, dim=-1), self.embedding)], dim=-1)\
            #         .reshape(*states.shape, self.embedding_dim)
            indexes = states.movedim(-1, 0).reshape(self.latent_channels, -1, 1).expand(-1, -1, self.embedding_dim)
            samples = torch.gather(self.embedding, 1, indexes).movedim(0,1)\
                .reshape(*states.shape, self.embedding_dim)
            # samples = torch.stack([v.index_select(0, s.reshape(-1)) for s, v in zip(states.split(1, dim=-1), self.embedding)], dim=-1)\
            #     .reshape(*states.shape, self.embedding_dim)
            return samples

        if add_default_samples:
            # samples = F.one_hot((states-1).clamp(min=0), self.categorical_dim).type_as(self.embedding)
            # samples = torch.sum(samples.unsqueeze(-1) * self.embedding.unsqueeze(0), dim=-2)
            samples = _index_samples((states-1).clamp(min=0))
            return torch.where(torch.logical_and(states > 0, states <= self.categorical_dim).unsqueeze(-1), samples, self._default_sample(samples))
        else:
            # samples = F.one_hot(states, self.categorical_dim).type_as(self.embedding)
            # samples = torch.sum(samples.unsqueeze(-1) * self.embedding.unsqueeze(0), dim=-2)
            samples = _index_samples(states)
            return samples

    def _latent_to_finite_state(self, latent: torch.Tensor) -> torch.LongTensor:
        # NOTE: different from forward process, this implementation only use 3 dims for distance calculation which is faster!
        embedding = self.embedding# .unsqueeze(0)
        latent_shape = latent.shape
        latent = latent.reshape(-1, self.latent_channels, self.embedding_dim).movedim(0, 1)
        # logits = self._calc_dist_logits(latent, embedding).squeeze(-2)
        distance = self._calc_distances(latent, embedding).movedim(0, 1)#.squeeze(-2)
        return super()._latent_to_finite_state(-distance).reshape(*latent_shape[:-1]) # last dim (embedding_dim) is removed

    def posterior_distribution(self, latent, **kwargs) -> distributions.RelaxedOneHotCategorical:
        latent = latent.reshape(-1, self.latent_channels, 1, self.embedding_dim)
        embedding = self.embedding.unsqueeze(0)
        logits = self._calc_dist_logits(latent, embedding).squeeze(-2)
        return super().posterior_distribution(logits, **kwargs)
        # if self.cat_reduce:
        #     logits = self._cat_reduce_logits(logits)
        # return distributions.RelaxedOneHotCategorical(self.gs_temp, logits=logits)

    def sample_from_posterior(self, posterior_dist: distributions.Distribution, **kwargs):
        if self.training and not self.force_hardmax:
            if self.train_em_update and self.em_state == False:
                # draw samples for M step
                dist_cat = distributions.Categorical(logits=posterior_dist.logits)
                samples = dist_cat.sample((self.train_em_mstep_samples, ))
                samples = F.one_hot(samples, posterior_dist.logits.shape[-1])\
                    .type_as(posterior_dist.logits)
                samples = samples.mean(dim=0)
            else:
                samples = super().sample_from_posterior(posterior_dist, **kwargs)
                if self.use_st_hardmax:
                    one_hot_samples = F.one_hot(posterior_dist.logits.argmax(-1), posterior_dist.logits.shape[-1])\
                        .type_as(posterior_dist.logits)
                    if self.hardmax_st_use_logits:
                        samples = posterior_dist.probs
                    samples = samples + one_hot_samples - samples.detach()                    
        else:
            samples = F.one_hot(posterior_dist.logits.argmax(-1), posterior_dist.logits.shape[-1])\
                .type_as(posterior_dist.logits)

        # NOTE: workaround for sample_kl                
        if self.use_sample_kl:
            self.update_cache(samples_gumbel=samples)            

        samples = torch.sum(samples.unsqueeze(-1) * self.embedding.unsqueeze(0), dim=-2)
        return samples

    def kl_divergence(self, prior_dist: distributions.Categorical, posterior_dist: distributions.RelaxedOneHotCategorical, input_shape: torch.Size = None, posterior_samples=None, **kwargs):
        # NOTE: workaround for sample_kl: we pass gumbel sample to super here
        if self.use_sample_kl:
            posterior_samples = self.get_raw_cache()["samples_gumbel"]
        kld = super().kl_divergence(prior_dist, posterior_dist, input_shape, posterior_samples, **kwargs)

        # use one_hot sample prior entropy as prior entropy during testing
        if not self.training:
            samples_one_hot = F.one_hot(posterior_dist.logits.argmax(-1), posterior_dist.logits.shape[-1])\
                .type_as(posterior_dist.logits)
            sample_prior_entropy = -(samples_one_hot * prior_dist.logits).sum()
            self.update_cache("metric_dict",
                prior_entropy=sample_prior_entropy / input_shape[0]
            )

        return kld

    def postprocess_samples(self, samples):
        samples = super().postprocess_samples(samples)
        return samples

    def _forward_flat(self, input: torch.Tensor, input_shape: torch.Size, prior: torch.Tensor = None, **kwargs):
        # output = super()._forward_flat(input, input_shape, prior, **kwargs)
        input = self._autoregressive_posterior(input)
        if self.train_em_update and self.em_state == False:
            input = input.detach()

        posterior_dist = self.posterior_distribution(input)
        samples = self.sample_from_posterior(posterior_dist)

        prior_ar = self._autoregressive_prior(prior=prior, input_shape=input_shape, posterior_samples=samples)

        if self.train_em_update and self.em_state == True:
            prior_ar = prior_ar.detach()
        prior_dist = self.prior_distribution(prior=prior_ar)

        # TODO: check add posterior_samples?
        KLD = torch.sum(self.kl_divergence(prior_dist, posterior_dist, input_shape=input_shape, posterior_samples=samples))
        if self.training:
            self.update_cache("loss_dict",
                loss_rate=KLD / input_shape[0], # normalize by batch size
            )
        # if implementation has not provide prior_entropy, use kl as prior_entropy instead
        if not "prior_entropy" in self.get_raw_cache("metric_dict"):
            self.update_cache("metric_dict",
                prior_entropy = KLD / input_shape[0], # normalize by batch size
            )

        avg_probs = torch.mean(posterior_dist.probs, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10), dim=-1))
        self.update_cache("metric_dict", 
            perplexity=perplexity.sum() # / B
        )

        output = self.postprocess_samples(samples)

        # add continuous vq loss to loss_rate?
        if self.training:
            # calculate cont loss for output
            if self.seperate_update_cont_loss:
                vq_loss = self._calc_cont_loss(input.detach(), output).sum() / input_shape[0] * self.vq_loss_weight
                commit_loss = self._calc_cont_loss(input, output.detach()).sum() / input_shape[0] * self.commit_loss_weight
                cont_loss = (vq_loss + commit_loss) * self.cont_loss_weight
            else:
                cont_loss = self._calc_cont_loss(input, output).sum() / input_shape[0] \
                    * self.cont_loss_weight
            loss_rate = self.get_raw_cache("loss_dict").get('loss_rate')
            self.update_cache("loss_dict",
                loss_rate=(loss_rate + cont_loss),
            )
            self.update_cache("moniter_dict",
                cont_loss=cont_loss,
            )
            if self.var_scale_anneal:
                self.update_cache("moniter_dict", 
                    var_scale=self.var_scale
                )

            if self.force_st or (self.use_st_below_entropy_threshold and self.entropy_temp < self.entropy_temp_threshold):
                output = output * (1 - self.st_weight) + input * self.st_weight + (output * self.st_weight - input * self.st_weight).detach()

            if self.ema_update_embedding:
                # ema update codebook only during M-step
                if not self.train_em_update or (self.train_em_update and self.em_state == False):
                    self._ema_update_embedding(input, posterior_dist.probs)

        # def _grad(grad):
        #     self._input_grad = grad
        # def _grad2(grad):
        #     self._output_grad = grad
        # def _grad3(grad):
        #     self._embedding_grad = grad
        # if input.requires_grad:
        #     input.register_hook(_grad)
        #     output.register_hook(_grad2)
        #     self.embedding.register_hook(_grad3)

        # if input.requires_grad:
        #     jacobian_io = torch.zeros(input.shape[0], input.shape[1], output.shape[1])
        #     for i in range(output.shape[-1]):
        #         grad_output = torch.zeros_like(output)
        #         grad_output[..., i] = 1
        #         grad = torch.autograd.grad(output, input, grad_output, create_graph=True)
        #         jacobian_io[..., i] = grad[0]
        #     self._jacobian_io = jacobian_io
        #     jacobian_bo = torch.zeros(*self.embedding.shape, output.shape[1])
        #     for i in range(output.shape[-1]):
        #         grad_output = torch.zeros_like(output)
        #         grad_output[..., i] = 1
        #         grad = torch.autograd.grad(output, self.embedding, grad_output, create_graph=True)
        #         jacobian_bo[..., i] = grad[0]
        #     self._jacobian_bo = jacobian_bo
        return output

    def _autoregressive_posterior(self, input : torch.Tensor = None, **kwargs) -> torch.Tensor:
        if self.use_autoregressive_posterior:
            input = input.view(-1, self.latent_channels, self.embedding_dim)
            ar_output = []
            for idx in range(self.latent_channels):
                if idx==0: 
                    ar_input = torch.cat([torch.zeros_like(input[:, 0]), input[:, 0]], dim=-1)
                else:
                    ar_input = input[:, (idx-1):(idx+1)].reshape(-1, 2*self.embedding_dim)
                ar_output.append(self.posterior_ar_model(ar_input))
            return torch.stack(ar_output, dim=1)
        else:
            return super()._autoregressive_posterior(input, **kwargs)


class MultiChannelVQPriorCoder(NNPriorCoder):
    """
    VQ codebase, as well as an outdated FSAR implementation. Keep only for reference and compability.
    Gumbel-softmax training vq from https://github.com/bshall/VectorQuantizedVAE/blob/master/model.py

    """    
    def __init__(self, latent_dim=8, num_embeddings=128, embedding_dim=32,
                 channels_share_codebook=False,
                 # smoothing
                 input_variance=0.0, input_variance_trainable=False,
                 # embedding
                 embedding_variance=0.0, embedding_variance_per_dimension=False,
                 embedding_variance_trainable=True, embedding_variance_lr_modifier=1.0,
                 # misc
                 dist_type=None, # RelaxedOneHotCategorical, AsymptoticRelaxedOneHotCategorical, DoubleRelaxedOneHotCategorical
                 use_soft_vq=False,
                 force_use_straight_through=False, st_weight=1.0,
                 # coding
                 coder_type="rans", # current support "rans", "tans"
                 coder_freq_precision=16,
                 fixed_input_shape=None,
                 # vamp
                 use_vamp_prior=False,
                 # code update
                 use_ema_update=False, ema_decay=0.999, ema_epsilon=1e-5, ema_reduce_ddp=True,
                 ema_adjust_sample=False,
                 embedding_lr_modifier=1.0,
                 # code prior
                 use_code_freq=False, code_freq_manual_update=False, update_code_freq_ema_decay=0.9,
                 use_code_variance=False,
                 # autoregressive prior
                 use_autoregressive_prior=False, 
                 ar_window_size=1, ar_offsets=None,
                 ar_method="finitestate", ar_mlp_per_channel=True,
                 ar_input_quantized=False,
                 ar_input_st_logits=False,
                #  ar_fs_method="table",
                 # autoregressive input (posterior)
                 use_autoregressive_posterior=False, autoregressive_posterior_method="maskconv3x3",
                 # loss
                 kl_cost=1.0, distance_detach_codebook=False,
                 use_st_gumbel=False, 
                 commitment_cost=0.25, commitment_over_exp=False, 
                 vq_cost=1.0, use_vq_loss_with_dist=False,
                 # testing
                 test_sampling=False, 
                 # init
                 initialization_mean=0.0, initialization_scale=None,
                 # monte-carlo sampling
                 train_mc_sampling=False, mc_loss_func=None, mc_sampling_size=64, mc_cost=1.0,
                 # annealing
                 relax_temp=1.0, relax_temp_anneal=False, gs_temp=0.5, gs_temp_anneal=False, 
                 entropy_temp=1.0, entropy_temp_min=1.0, entropy_temp_threshold=0.0, entropy_temp_anneal=False, 
                 use_st_below_entropy_threshold=False, use_vq_loss_below_entropy_threshold=False, use_commit_loss_below_entropy_threshold=False,
        ):
        super().__init__()

        self.latent_dim = latent_dim
        self.num_embeddings = num_embeddings
        self.categorical_dim = num_embeddings # alias
        self.embedding_dim = embedding_dim
        self.channels_share_codebook = channels_share_codebook

        self.coder_type = coder_type
        self.coder_freq_precision = coder_freq_precision

        self.input_variance = input_variance
        self.input_variance_trainable = input_variance_trainable
        if input_variance_trainable and input_variance > 0:
            self.input_variance = nn.Parameter(torch.tensor([np.log(input_variance)]))

        self.use_vamp_prior = use_vamp_prior
        
        self.use_ema_update = use_ema_update
        self.ema_decay = ema_decay
        self.ema_epsilon = ema_epsilon
        self.ema_reduce_ddp = ema_reduce_ddp
        self.ema_adjust_sample = ema_adjust_sample
        
        self.embedding_lr_modifier = embedding_lr_modifier
        embedding = torch.zeros(1 if self.channels_share_codebook else latent_dim, num_embeddings, embedding_dim)
        if initialization_scale is None:
            initialization_scale = 1/num_embeddings
        embedding.uniform_(initialization_mean-initialization_scale, initialization_mean+initialization_scale)
        if use_vamp_prior:
            # embedding should be set by set_vamp_posterior
            self.register_buffer("embedding", embedding)
        else:
            if use_ema_update:
                self.register_buffer("embedding", embedding)
                self.register_buffer("ema_count", torch.zeros(*self.embedding.shape[:-1]))
                self.register_buffer("ema_weight", self.embedding.clone())
                # if ema_adjust_sample:
                #     self.register_buffer("ema_weight_posterior", self.embedding.clone())
            else:
                self.embedding = nn.Parameter(embedding)
                self.embedding.lr_modifier = embedding_lr_modifier

        self.use_embedding_variance = (embedding_variance > 0)
        self.embedding_variance_trainable = embedding_variance_trainable
        self.embedding_variance_per_dimension = embedding_variance_per_dimension
        if self.use_embedding_variance:
            if self.embedding_variance_per_dimension:
                embedding_variance = torch.ones_like(self.embedding) * np.log(embedding_variance) # exponential reparameterization
            else:
                embedding_variance = torch.ones(1) * np.log(embedding_variance) # exponential reparameterization
            if embedding_variance_trainable:
                self.embedding_variance = nn.Parameter(embedding_variance)
                self.embedding_variance.lr_modifier = embedding_variance_lr_modifier
            else:
                self.register_buffer("embedding_variance", embedding_variance)

        self.use_code_freq = use_code_freq
        self.code_freq_manual_update = code_freq_manual_update
        self.update_code_freq_ema_decay = update_code_freq_ema_decay

        self.use_code_variance = use_code_variance

        self.use_autoregressive_prior = use_autoregressive_prior
        self.ar_window_size = ar_window_size
        self.ar_offsets = ar_offsets
        self.ar_method = ar_method
        self.ar_mlp_per_channel = ar_mlp_per_channel
        self.ar_input_quantized = ar_input_quantized
        self.ar_input_st_logits = ar_input_st_logits
        # self.ar_fs_method = ar_fs_method
        # full ar
        if self.ar_window_size is None:
            self.ar_window_size = self.latent_dim - 1
        # custom ar offset setting
        if self.ar_offsets is None:
            self.ar_offsets = [(-offset,) for offset in range(1, self.ar_window_size+1)]
        else:
            self.ar_window_size = len(ar_offsets)

        # full ar
        if self.ar_window_size is None:
            self.ar_window_size = self.latent_dim - 1
        if use_code_freq:
            prior_logprob = torch.zeros(*self.embedding.shape[:-1])

            # self.embedding_freq = nn.Parameter(torch.ones(latent_dim, num_embeddings) / num_embeddings)
            # if code_freq_manual_update:
            #     self.embedding_freq.requires_grad = False
            self.embedding_logprob = nn.Parameter(prior_logprob)
            if code_freq_manual_update:
                self.embedding_logprob.requires_grad = False
        else:
            self.register_buffer("embedding_logprob", 
                                 torch.zeros(*self.embedding.shape[:-1]) - math.log(self.num_embeddings), 
                                 persistent=False)
            
        if self.use_code_variance:
            embedding_logvar = torch.zeros(*self.embedding.shape[:-1])
            self.embedding_logvar = nn.Parameter(embedding_logvar)
        
        self.coder_type = coder_type
        # TODO: temp fix for no rans fsar impl! Remove this after fsar-rans is done!
        if self.use_autoregressive_prior and self.ar_method == "finitestate":
            if self.coder_type == "rans":
                print("Warning! rans fsar is not implemented! switching to tans!")
                self.coder_type = "tans"
        self.fixed_input_shape = fixed_input_shape

        if use_autoregressive_prior:
            if self.ar_input_quantized:
                ar_input_channels = self.embedding_dim
            else:
                ar_input_channels = self.categorical_dim + 1
            if self.ar_method == "finitestate":
                if self.ar_mlp_per_channel:
                    self.fsar_mlps_per_channel = nn.ModuleList(
                        [
                            nn.Sequential(
                                nn.Linear(ar_input_channels * self.ar_window_size, 2 * self.ar_window_size * ar_input_channels),
                                nn.LeakyReLU(),
                                nn.Linear(2 * self.ar_window_size * ar_input_channels, 2 * self.categorical_dim),
                                nn.LeakyReLU(),
                                nn.Linear(2 * self.categorical_dim, self.categorical_dim),
                            )
                            for _ in range(self.latent_dim)
                        ]
                    )
                else:
                    self.fsar_mlp = nn.Sequential(
                        nn.Linear((self.categorical_dim + 1) * self.ar_window_size, 2 * self.ar_window_size * (self.categorical_dim + 1)),
                        nn.LeakyReLU(),
                        nn.Linear(2 * self.ar_window_size * (self.categorical_dim + 1), 2 * self.categorical_dim),
                        nn.LeakyReLU(),
                        nn.Linear(2 * self.categorical_dim, self.categorical_dim),
                    )

        # model based ar
        if self.use_autoregressive_prior:
            ar_model = None
            if self.ar_input_quantized:
                ar_input_channels = self.embedding_dim
            else:
                ar_input_channels = self.categorical_dim
            if self.ar_method == "maskconv3x3":
                ar_model = MaskedConv2d(ar_input_channels * self.latent_dim, self.latent_dim * self.categorical_dim, 3, padding=1)
            elif self.ar_method == "maskconv5x5":
                ar_model = MaskedConv2d(ar_input_channels * self.latent_dim, self.latent_dim * self.categorical_dim, 5, padding=2)
            elif self.ar_method == "maskconv3d3x3x3":
                ar_model = MaskedConv3d(ar_input_channels, self.categorical_dim, 3, padding=1)
            elif self.ar_method == "maskconv3d5x5x5":
                ar_model = MaskedConv3d(ar_input_channels, self.categorical_dim, 5, padding=2)
            elif self.ar_method == "checkerboard3x3":
                ar_model = MaskedConv2d(ar_input_channels * self.latent_dim, self.latent_dim * self.categorical_dim, 3, padding=1, mask_type="Checkerboard")
            elif self.ar_method == "checkerboard5x5":
                ar_model = MaskedConv2d(ar_input_channels * self.latent_dim, self.latent_dim * self.categorical_dim, 5, padding=2, mask_type="Checkerboard")

            if ar_model is not None:
                self.ar_model = nn.Sequential(
                    ar_model,
                    # nn.Conv2d(ar_input_channels * self.latent_dim * 6 // 3, ar_input_channels * self.latent_dim * 5 // 3, 1),
                    # nn.LeakyReLU(inplace=True),
                    # nn.Conv2d(ar_input_channels * self.latent_dim * 5 // 3, ar_input_channels * self.latent_dim * 4 // 3, 1),
                    # nn.LeakyReLU(inplace=True),
                    # nn.Conv2d(ar_input_channels * self.latent_dim * 4 // 3, ar_input_channels * self.latent_dim * 3 // 3, 1),
                )

        self.use_autoregressive_posterior = use_autoregressive_posterior
        self.autoregressive_posterior_method = autoregressive_posterior_method
        if autoregressive_posterior_method == "maskconv3x3":
            self.input_ar_model = MaskedConv2d(self.embedding_dim * self.latent_dim, self.embedding_dim * self.latent_dim, 3, padding=1)
        
        self.dist_type = dist_type
        self.use_straight_through = (dist_type is None) or force_use_straight_through
        self.st_weight = st_weight
        self.use_soft_vq = use_soft_vq
                
        self.kl_cost = kl_cost
        self.distance_detach_codebook = distance_detach_codebook
        self.use_st_gumbel = use_st_gumbel
        self.commitment_cost = commitment_cost
        self.commitment_over_exp = commitment_over_exp
        self.vq_cost = vq_cost
        self.use_vq_loss_with_dist = use_vq_loss_with_dist

        self.test_sampling = test_sampling

        self.train_mc_sampling = train_mc_sampling
        self.mc_loss_func = mc_loss_func
        self.mc_sampling_size = mc_sampling_size
        self.mc_cost = mc_cost

        self.relax_temp_anneal = relax_temp_anneal
        if relax_temp_anneal:
            self.relax_temp = nn.Parameter(torch.tensor(relax_temp), requires_grad=False)
        else:
            self.relax_temp = relax_temp

        self.gs_temp_anneal = gs_temp_anneal
        if gs_temp_anneal:
            self.gs_temp = nn.Parameter(torch.tensor(gs_temp), requires_grad=False)
        else:
            self.gs_temp = gs_temp

        self.entropy_temp_anneal = entropy_temp_anneal
        if entropy_temp_anneal:
            self.entropy_temp = nn.Parameter(torch.tensor(entropy_temp), requires_grad=False)
            self.register_buffer("entropy_temp_min", torch.tensor(entropy_temp_min, requires_grad=False), persistent=False)
        else:
            self.entropy_temp = entropy_temp
            self.entropy_temp_min = entropy_temp_min
        self.entropy_temp_threshold = entropy_temp_threshold
        self.use_st_below_entropy_threshold = use_st_below_entropy_threshold
        self.use_vq_loss_below_entropy_threshold = use_vq_loss_below_entropy_threshold
        self.use_commit_loss_below_entropy_threshold = use_commit_loss_below_entropy_threshold

        # model state
        self.state_gs_perturb = True

        # initalize members for coding
        self.update_state()

    def _pairwise_distance(self, x1 : torch.Tensor, x2 : torch.Tensor, scale2 : torch.Tensor = None) -> torch.Tensor:
        """_summary_

        Args:
            x1 (torch.Tensor): Batch * Elements1 * Vector
            x2 (torch.Tensor): Batch * Elements2 * Vector
            scale2 (torch.Tensor): Batch * Elements2

        Returns:
            torch.Tensor): Batch * Elements1 * Elements2
        """        
        # dists = torch.baddbmm(torch.sum(x2 ** 2, dim=2).unsqueeze(1) +
        #                           torch.sum(x1 ** 2, dim=2, keepdim=True),
        #                           x1, x2.transpose(1, 2),
        #                           alpha=-2.0, beta=1.0) / x1.shape[-1]
        dists = (torch.sum(x1**2, dim=-1, keepdim=True) \
                + torch.sum(x2**2, dim=-1).unsqueeze(-2) \
                - 2 * torch.matmul(x1, x2.transpose(-2, -1))) / x1.shape[-1]
        if scale2 is not None:
            dists = dists / scale2.unsqueeze(1)
        return dists
    
    def _distance_loss(self, x1 : torch.Tensor, x2 : torch.Tensor) -> torch.Tensor:
        return self._pairwise_distance(x1, x2).mean()

    def _logits_from_distances(self, distances):
        # NOTE: the original code use the l2-sum distance!
        return -distances * self.embedding_dim

    def _sample_from_param(self, param) -> torch.Tensor:
        return param
        
    def _sample_from_embedding(self, samples, embedding=None) -> torch.Tensor:
        if embedding is None:
            embedding = self.embedding
        return torch.matmul(samples, embedding)

    def _calculate_kl_from_dist(self, dist : distributions.Distribution, prior_logits=None):
        # KL: N, B, spatial_dim, M
        entropy_temp = max(self.entropy_temp, self.entropy_temp_min)
        # KL = dist.probs * (dist.logits * entropy_temp - prior_logits)
        # KL[(dist.probs == 0).expand_as(KL)] = 0
        # KL = KL.mean(dim=1).sum() # mean on batch dim
        posterior_entropy = dist.probs * dist.logits
        posterior_entropy[dist.probs == 0] = 0 # prevent nan
        prior_entropy = dist.probs.reshape(dist.probs.shape[0], -1, self.num_embeddings) * prior_logits

        KL = posterior_entropy * entropy_temp - prior_entropy.reshape_as(posterior_entropy)
        KL = KL.mean(dim=1).sum()
        return KL

    def _calculate_ar_prior_logits(self, samples=None, input_shape=None):
        if self.use_autoregressive_prior:
            assert samples is not None # N * flat_dim * num_embeddings
            assert input_shape is not None # 
            batch_size = input_shape[0]
            spatial_shape = input_shape[2:]
            flat_dim = batch_size * np.prod(spatial_shape)
            prior_logits = self.embedding_logprob.unsqueeze(0)
            samples = samples.transpose(0, 1)
            if self.ar_method == "finitestate":
                autoregressive_samples = []
                if self.ar_input_quantized:
                    ar_samples_reshape = samples.reshape(batch_size, *spatial_shape, self.latent_dim, self.embedding_dim).movedim(-2, 1)
                    for ar_offset in self.ar_offsets:
                        default_samples = torch.zeros_like(ar_samples_reshape)# [..., :1]
                        ar_samples = ar_samples_reshape
                        # take ar samples
                        # ar_samples = torch.cat(
                        #     [
                        #         default_sample,
                        #         ar_samples_reshape,
                        #     ], dim=-1
                        # )
                        # leave 0 as unknown sample, let total categories categorical_dim+1
                        # default_samples = torch.cat(
                        #     [
                        #         default_sample + 1,
                        #         torch.zeros_like(ar_samples_reshape),
                        #     ], dim=-1
                        # )
                        for data_dim, data_offset in enumerate(ar_offset):
                            if data_offset >= 0: continue
                            batched_data_dim = data_dim + 1
                            assert batched_data_dim != ar_samples.ndim - 1 # ar could not include categorical_dim
                            ar_samples = torch.cat((
                                default_samples.narrow(batched_data_dim, 0, -data_offset),
                                ar_samples.narrow(batched_data_dim, 0, ar_samples.shape[batched_data_dim]+data_offset)
                            ), dim=batched_data_dim)
                        autoregressive_samples.append(ar_samples)
                else:
                    ar_samples_reshape = samples.reshape(batch_size, *spatial_shape, self.latent_dim, self.categorical_dim).movedim(-2, 1)
                    for ar_offset in self.ar_offsets:
                        default_sample = torch.zeros_like(ar_samples_reshape)[..., :1]
                        ar_samples = ar_samples_reshape
                        # take ar samples
                        ar_samples = torch.cat(
                            [
                                default_sample,
                                ar_samples_reshape,
                            ], dim=-1
                        )
                        # leave 0 as unknown sample, let total categories categorical_dim+1
                        default_samples = torch.cat(
                            [
                                default_sample + 1,
                                torch.zeros_like(ar_samples_reshape),
                            ], dim=-1
                        )
                        for data_dim, data_offset in enumerate(ar_offset):
                            if data_offset >= 0: continue
                            batched_data_dim = data_dim + 1
                            assert batched_data_dim != ar_samples.ndim - 1 # ar could not include categorical_dim
                            ar_samples = torch.cat((
                                default_samples.narrow(batched_data_dim, 0, -data_offset),
                                ar_samples.narrow(batched_data_dim, 0, ar_samples.shape[batched_data_dim]+data_offset)
                            ), dim=batched_data_dim)
                        autoregressive_samples.append(ar_samples)
                # [batch_size, self.latent_dim, *spatial_shape, self.ar_window_size*(self.categorical_dim+1)]
                autoregressive_samples = torch.cat(autoregressive_samples, dim=-1)
                if self.ar_mlp_per_channel:
                    autoregressive_samples_per_channel = autoregressive_samples.movedim(1, -2)\
                        .reshape(flat_dim, self.latent_dim, -1)
                    ar_logits_reshape = torch.stack([mlp(sample_channel.squeeze(1)) for mlp, sample_channel in zip(self.fsar_mlps_per_channel, autoregressive_samples_per_channel.split(1, dim=1))], dim=1)
                    prior_logits = ar_logits_reshape + prior_logits
                else:
                    autoregressive_samples_flat = autoregressive_samples.movedim(1, -2).reshape(flat_dim * self.latent_dim, -1)
                    ar_logits_reshape = self.fsar_mlp(autoregressive_samples_flat)
                    # merge ar logits and prior logits
                    prior_logits = ar_logits_reshape.reshape_as(samples) + prior_logits
            # TODO: ar models for vqvae
            else:
                assert len(spatial_shape) == 2
                ar_samples_reshape = samples.reshape(batch_size, *spatial_shape, -1).movedim(-1, 1)
                if self.ar_method.startswith("maskconv"):
                    if self.ar_method.startswith("maskconv3d"):
                        ar_samples_reshape = ar_samples_reshape.reshape(batch_size, self.latent_dim, -1, *spatial_shape)\
                            .permute(0, 2, 1, 3, 4)
                    prior_logits_reshape = self.ar_model(ar_samples_reshape)
                    if self.ar_method.startswith("maskconv3d"):
                        prior_logits_reshape = prior_logits_reshape.permute(0, 2, 1, 3, 4)\
                            .reshape(batch_size, self.latent_dim*self.categorical_dim, *spatial_shape)
                elif self.ar_method.startswith("checkerboard"):
                    prior_logits_reshape = self.ar_model(ar_samples_reshape)
                    checkerboard_mask_h_0 = torch.arange(0, spatial_shape[-2], 2, dtype=torch.long, device=ar_samples_reshape.device)
                    checkerboard_mask_h_1 = torch.arange(1, spatial_shape[-2], 2, dtype=torch.long, device=ar_samples_reshape.device)
                    checkerboard_mask_w_0 = torch.arange(0, spatial_shape[-1], 2, dtype=torch.long, device=ar_samples_reshape.device)
                    checkerboard_mask_w_1 = torch.arange(1, spatial_shape[-1], 2, dtype=torch.long, device=ar_samples_reshape.device)
                    checkerboard_index_h_01, checkerboard_index_w_01 = torch.meshgrid(checkerboard_mask_h_0, checkerboard_mask_w_1)
                    checkerboard_index_h_10, checkerboard_index_w_10 = torch.meshgrid(checkerboard_mask_h_1, checkerboard_mask_w_0)
                    # multi-indexed tensor cannot be used as mutable left value
                    # prior_logits_reshape[..., checkerboard_mask_h_0, :][..., checkerboard_mask_w_1] = prior_dist.logits.reshape(1, self.latent_channels, 1, 1)
                    # prior_logits_reshape[..., checkerboard_mask_h_1, :][..., checkerboard_mask_w_0] = prior_dist.logits.reshape(1, self.latent_channels, 1, 1)
                    prior_logits_reshape[..., checkerboard_index_h_01, checkerboard_index_w_01] = prior_logits.reshape(1, self.latent_dim*self.categorical_dim, 1, 1)
                    prior_logits_reshape[..., checkerboard_index_h_10, checkerboard_index_w_10] = prior_logits.reshape(1, self.latent_dim*self.categorical_dim, 1, 1)
                else:
                    raise NotImplementedError(f"Unknown self.ar_method {self.ar_method}")
                prior_logits = prior_logits_reshape.movedim(1, -1).reshape(samples.shape[0], self.latent_dim, self.categorical_dim)
            # normalize logits
            prior_logits = torch.log_softmax(prior_logits, dim=-1).transpose(0, 1) ## N*flat_dim*M
        else:
            # prior_logits = torch.log_softmax(self.embedding_logprob, dim=-1).unsqueeze(1) # N*1*M
            if self.use_code_freq:
                # NOTE: maybe it's better to use log freq as parameter?
                # prior_logits = torch.log(self.embedding_freq / self.embedding_freq.sum(-1, keepdim=True))
                prior_logits = torch.log_softmax(self.embedding_logprob, dim=-1).unsqueeze(1) # N*1*M
            else:
                prior_logits = -math.log(self.num_embeddings)

        return prior_logits

    def _manual_update_code_freq(self, samples : torch.Tensor) -> None:
        with torch.no_grad():
            # NOTE: should soft samples be allowed here?
            total_count = samples.sum(dim=1)
            # sum over all gpus
            if self.ema_reduce_ddp and distributed.is_initialized():
                distributed.all_reduce(total_count)

            # normalize to probability.
            normalized_freq = total_count / total_count.sum(-1, keepdim=True)

            # ema update
            # ema = (1 - self.update_code_freq_ema_decay) * normalized_freq + self.update_code_freq_ema_decay * self.embedding_freq
            # self.embedding_freq.copy_(ema)

            ema = (1 - self.update_code_freq_ema_decay) * normalized_freq + \
                self.update_code_freq_ema_decay * torch.softmax(self.embedding_logprob, dim=-1)
            self.embedding_logprob.copy_(torch.log(ema))

    def set_custom_state(self, state: str = None):
        if state == "perturbed":
            self.state_gs_perturb = True
        else:
            self.state_gs_perturb = False
        return super().set_custom_state(state)

    def forward(self, x : torch.Tensor, calculate_sample_kl=False, **kwargs):
        x_shape = x.shape
        spatial_shape = x.shape[2:]
        # B, C, H, W = x.size()
        B, C = x.shape[:2]
        spatial_dim = np.prod(spatial_shape)
        N, M, D = self.latent_dim, self.num_embeddings, self.embedding_dim # self.embedding.size()
        x_embedding_dim = C // N
        assert C == N * x_embedding_dim

        if self.training and self.input_variance > 0:
            x = x + torch.normal(torch.zeros_like(x)) * self.input_variance.exp()
            self.update_cache("moniter_dict", 
                input_variance_mean=torch.mean(self.input_variance.exp()),
            )

        if self.use_autoregressive_posterior:
            x = self.input_ar_model(x)

        # x = x.view(B, N, x_embedding_dim H, W).permute(1, 0, 3, 4, 2)
        x = x.view(B, N, x_embedding_dim, spatial_dim).permute(1, 0, 3, 2) # N*B*spatial_dim*x_embedding_dim
        x_sample = self._sample_from_param(x)
        flat_dim = B * spatial_dim
        x_flat = x.reshape(N, flat_dim, x_embedding_dim)

        embedding = self.embedding
        if self.training and self.use_embedding_variance:
            embedding = embedding + torch.normal(torch.zeros_like(embedding)) * self.embedding_variance.exp()
            self.update_cache("moniter_dict", 
                embedding_variance_mean=torch.mean(self.embedding_variance.exp()),
            )

        # detach x_flat for straight through optimization
        if self.dist_type is None:
            x_flat = x_flat.detach()

        distances = self._pairwise_distance(x_flat, embedding.detach() if self.distance_detach_codebook else embedding)
        if self.use_code_variance:
            distances = distances / self.embedding_logvar.exp().unsqueeze(1)
        # distances = distances.view(N, B, H, W, M)
        distances = distances.view(N, B, spatial_dim, M)

        dist : distributions.Distribution = None # for lint
        logits = self._logits_from_distances(distances)
        eps = 1e-6
        if self.dist_type is None:
            dist = None
        elif self.dist_type == "CategoricalRSample":
            logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True) 
            dist = CategoricalRSample(logits=logits)
        elif self.dist_type == "RelaxedOneHotCategorical":
            logits_norm = logits - torch.logsumexp(logits, dim=-1, keepdim=True) 
            logits = logits_norm / self.relax_temp
            dist = RelaxedOneHotCategorical(self.gs_temp if self.state_gs_perturb else eps,
                logits=logits,
            )
        elif self.dist_type == "AsymptoticRelaxedOneHotCategorical":
            dist = AsymptoticRelaxedOneHotCategorical(self.gs_temp, self.relax_temp if self.state_gs_perturb else eps,
                logits=logits,
            )
        elif self.dist_type == "DoubleRelaxedOneHotCategorical":
            dist = DoubleRelaxedOneHotCategorical(self.gs_temp, self.relax_temp if self.state_gs_perturb else eps,
                logits=logits,
            )
        else:
            raise ValueError(f"Unknown dist_type {self.dist_type} !")
        
        # soft vq
        if self.training and self.use_soft_vq:
            samples = torch.softmax(logits / self.relax_temp, dim=-1).view(N, flat_dim, M)
        # do sampling from dist
        elif dist is not None and (self.training or self.test_sampling):
            # if not dist.has_rsample:
            #     if self.training:
            #         raise ValueError(f"distribution {self.dist_type} cannot be used for training!")
            #     samples = dist.sample().view(N, -1, M)
            # else:
            if self.train_mc_sampling:
                samples = dist.sample_n(self.mc_sampling_size)
                quantized = self._sample_from_embedding(samples, embedding)
                loss_mc = self.mc_loss_func(quantized) / self.mc_sampling_size
                self.update_cache("loss_dict", loss_mc=loss_mc * self.mc_cost)
            else:
                samples = dist.rsample().view(N, flat_dim, M)
                if self.use_st_gumbel:
                    _, ind = samples.max(dim=-1)
                    samples_hard = torch.zeros_like(samples).view(N, flat_dim, M)
                    samples_hard.scatter_(-1, ind.view(N, flat_dim, 1), 1)
                    samples_hard = samples_hard.view(N, flat_dim, M)
                    samples = samples_hard - samples.detach() + samples
                if not self.training:
                    _, ind = samples.max(dim=-1)
                    self.update_cache("hist_dict",
                        code_hist=ind.view(N, -1).float().cpu().detach_()
                    )
                    samples = torch.zeros_like(samples).view(N, flat_dim, M)
                    samples.scatter_(-1, ind.view(N, flat_dim, 1), 1)
        else:
            samples = torch.argmin(distances, dim=-1)
            if not self.training:
                self.update_cache("hist_dict",
                    code_hist=samples.view(N, -1).float().cpu().detach_()
                )
            samples = F.one_hot(samples, M).float()
            samples = samples.view(N, flat_dim, M)

        quantized = self._sample_from_embedding(samples, embedding)
        quantized = quantized.view_as(x_sample)
        
        if self.ar_input_quantized:
            samples_quantized = quantized
            if self.use_straight_through:
                samples_quantized = x_sample + (samples_quantized - x_sample).detach() # straight through
            samples_quantized = samples_quantized.reshape(N, flat_dim, D)
            prior_logits = self._calculate_ar_prior_logits(samples_quantized, x_shape)
        else:
            if self.ar_input_st_logits and self.use_straight_through:
                ar_st_logits = torch.softmax(logits.view(N, flat_dim, M), dim=-1)
                samples = ar_st_logits + (samples - ar_st_logits).detach()
            prior_logits = self._calculate_ar_prior_logits(samples, x_shape)

        if calculate_sample_kl:
            if self.use_code_freq or self.use_autoregressive_prior:
                sample_kl = (samples * prior_logits).sum(-1)
            else:
                sample_kl = torch.ones(N, flat_dim).type_as(samples) * -math.log(self.num_embeddings)
            sample_kl = sample_kl.view(N, B, spatial_dim).sum((0, 2))
            self.update_cache("common", sample_kl=sample_kl)

        if self.training:
            # manual code freq update
            if self.use_code_freq and self.code_freq_manual_update:
                self._manual_update_code_freq(samples)

            # vq loss / kl loss
            if self.use_ema_update:
                    # TODO: fix ema for shared codebook?
                    with torch.no_grad():
                        total_count = samples.sum(dim=1)
                        dw = torch.bmm(samples.transpose(1, 2), x_flat)
                        if self.ema_reduce_ddp and distributed.is_initialized():
                            distributed.all_reduce(total_count)
                            distributed.all_reduce(dw)
                        self.ema_count = self.ema_decay * self.ema_count + (1 - self.ema_decay) * total_count
                        n = torch.sum(self.ema_count, dim=-1, keepdim=True)
                        self.ema_count = (self.ema_count + self.ema_epsilon) / (n + M * self.ema_epsilon) * n
                        self.ema_weight = self.ema_decay * self.ema_weight + (1 - self.ema_decay) * dw
                        self.embedding = self.ema_weight / self.ema_count.unsqueeze(-1)
            else:
                if dist is None \
                    or self.use_vq_loss_below_entropy_threshold and self.entropy_temp < self.entropy_temp_threshold\
                    or self.use_vq_loss_with_dist:
                    # loss_vq = F.mse_loss(x.detach(), quantized)
                    # distances_x_detached = self._pairwise_distance(embedding, x_flat.detach())
                    # loss_vq = torch.bmm(samples, distances_x_detached).mean() * self.embedding_dim
                    # loss_vq = self._pairwise_distance(quantized.reshape(N*flat_dim, 1, x_embedding_dim), x.detach().reshape(N*flat_dim, 1, x_embedding_dim)).mean()
                    # loss_vq = self._pairwise_distance(x.detach().reshape(N*flat_dim, 1, x_embedding_dim), quantized.reshape(N*flat_dim, 1, x_embedding_dim)).mean()
                    if self.vq_cost != 0.0:
                        loss_vq = self._distance_loss(x_sample.reshape(N*flat_dim, 1, x_sample.shape[-1]).detach(), 
                            quantized.reshape(N*flat_dim, 1, x_sample.shape[-1]))
                        self.update_cache("loss_dict", loss_vq=loss_vq * self.vq_cost)
                
            # update prior logits
            if dist is None:
                # code freq should be updated with loss_rate
                if self.use_code_freq or self.use_autoregressive_prior:
                    loss_rate = -(samples * prior_logits).sum() / B
                    self.update_cache("loss_dict", loss_rate=loss_rate)
            else:
                KL = self._calculate_kl_from_dist(dist, prior_logits=prior_logits)
                self.update_cache("loss_dict", loss_rate=KL * self.kl_cost)

            # commitment loss
            commitment_cost = self.commitment_cost
            if self.use_commit_loss_below_entropy_threshold:
                if self.entropy_temp < self.entropy_temp_threshold:
                    commitment_cost = 0.25

            if commitment_cost != 0:
                if self.commitment_over_exp and dist is not None:
                    loss_commitment = (dist.probs * distances).mean()
                else:
                    # loss_commitment = F.mse_loss(x, quantized.detach())
                    # distances_embedding_detached = self._pairwise_distance(x_flat, embedding.detach())
                    # loss_commitment = self._pairwise_distance(x.reshape(N*flat_dim, 1, D), quantized.detach().reshape(N*flat_dim, 1, D)).mean()
                    loss_commitment = self._distance_loss(x_sample.reshape(N*flat_dim, 1, x_sample.shape[-1]), 
                        quantized.reshape(N*flat_dim, 1, x_sample.shape[-1]).detach())

                # loss = self.commitment_cost * e_latent_loss
                self.update_cache("loss_dict", loss_commitment = commitment_cost * loss_commitment)

        # TODO: add kl entropy metric?
        if self.use_code_freq or self.use_autoregressive_prior:
            # normalized_freq = self.embedding_freq / self.embedding_freq.sum(-1, keepdim=True)
            # prior_entropy = torch.bmm(samples, -torch.log(normalized_freq).unsqueeze(-1)).sum() / B
            prior_entropy = -(samples * prior_logits).sum() / B
        else:
            prior_entropy = math.log(self.num_embeddings) * (samples.numel() / M) / B
        self.update_cache("metric_dict", 
            prior_entropy=prior_entropy,
        )

        # kl
        # if dist is not None:
        #     KL = dist.probs * (dist.logits + math.log(M))
        #     KL[(dist.probs == 0).expand_as(KL)] = 0
        #     KL = KL.mean(dim=1).sum() # mean on batch dim

        # perplexity
        avg_probs = torch.mean(samples, dim=1)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10), dim=-1))
        self.update_cache("metric_dict", 
            perplexity=perplexity.sum() # / B
        )

        # annealing
        if self.gs_temp_anneal:
            if self.training:
                self.update_cache("metric_dict", 
                    gs_temp=self.gs_temp
                )
        if self.relax_temp_anneal:
            if self.training:
                self.update_cache("metric_dict", 
                    relax_temp=self.relax_temp
                )
        if self.entropy_temp_anneal:
            if self.training:
                self.update_cache("metric_dict", 
                    entropy_temp=self.entropy_temp
                )

        # ema adjust sample
        if self.use_ema_update and self.ema_adjust_sample:
            with torch.no_grad():
                total_count = samples.sum(dim=1)
                dw = torch.bmm(samples.transpose(1, 2), torch.bmm(samples, embedding))
                if self.ema_reduce_ddp and distributed.is_initialized():
                    distributed.all_reduce(total_count)
                    distributed.all_reduce(dw)
                ema_count = self.ema_decay * self.ema_count + (1 - self.ema_decay) * total_count
                n = torch.sum(ema_count, dim=-1, keepdim=True)
                ema_count = (ema_count + self.ema_epsilon) / (n + M * self.ema_epsilon) * n
                ema_weight = self.ema_decay * embedding + (1 - self.ema_decay) * dw
                ema_weight = ema_weight / ema_count.unsqueeze(-1)
                quantized = torch.bmm(samples, ema_weight).view_as(x_sample)
        # output
        # quantized = quantized.permute(1, 0, 4, 2, 3).reshape(*x_shape)
        if self.use_straight_through or \
            self.use_st_below_entropy_threshold and self.entropy_temp < self.entropy_temp_threshold:
            quantized = quantized * (1 - self.st_weight) + x_sample * self.st_weight + (quantized * self.st_weight - x_sample * self.st_weight).detach()
        quantized = quantized.permute(1, 0, 3, 2).reshape(B, -1, *spatial_shape) #.reshape(*x_shape)
        
        return quantized

    def set_vamp_posterior(self, posterior):
        if not self.use_vamp_prior:
            raise RuntimeError("Should specify use_vamp_prior=True!")

        # check shape
        spatial_shape = posterior.shape[2:]
        # B, C, H, W = x.size()
        B, C = posterior.shape[:2]
        spatial_dim = np.prod(spatial_shape)
        N, M, D = self.embedding.size()
        assert C == N * D
        assert M == B * spatial_dim

        posterior = posterior.view(B, N, D, spatial_dim).permute(1, 0, 3, 2) # N*B*spatial_dim*D
        posterior = posterior.reshape(N, M, D).contiguous()
        self.embedding = posterior

    def encode(self, input, *args, **kwargs) -> bytes:
        spatial_shape = input.shape[2:]
        # B, C, H, W = x.size()
        B, C = input.shape[:2]
        spatial_dim = np.prod(spatial_shape)
        N, M, D = self.embedding.size()
        x_embedding_dim = C // N
        assert C == N * x_embedding_dim

        # x = x.view(B, N, x_embedding_dim H, W).permute(1, 0, 3, 4, 2)
        x = input.view(B, N, x_embedding_dim, spatial_dim).permute(1, 0, 3, 2) # N*B*spatial_dim*x_embedding_dim
        x_sample = self._sample_from_param(x)
        flat_dim = B * spatial_dim
        x_flat = x.reshape(N, flat_dim, x_embedding_dim)

        # detach x_flat for straight through optimization
        if self.dist_type is None:
            x_flat = x_flat.detach()

        distances = self._pairwise_distance(x_flat, self.embedding)
        # distances = distances.view(N, B, H, W, M)
        distances = distances.view(N, B, spatial_dim, M)

        # dist : distributions.Distribution = None # for lint
        # logits = self._logits_from_distances(distances)
        # eps = 1e-6
        # if self.dist_type is None:
        #     dist = None
        # elif self.dist_type == "CategoricalRSample":
        #     logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True) 
        #     dist = CategoricalRSample(logits=logits)
        # elif self.dist_type == "RelaxedOneHotCategorical":
        #     logits_norm = logits - torch.logsumexp(logits, dim=-1, keepdim=True) 
        #     logits = logits_norm / self.relax_temp
        #     dist = RelaxedOneHotCategorical(self.gs_temp if self.state_gs_perturb else eps,
        #         logits=logits,
        #     )
        # elif self.dist_type == "AsymptoticRelaxedOneHotCategorical":
        #     dist = AsymptoticRelaxedOneHotCategorical(self.gs_temp, self.relax_temp if self.state_gs_perturb else eps,
        #         logits=logits,
        #     )
        # elif self.dist_type == "DoubleRelaxedOneHotCategorical":
        #     dist = DoubleRelaxedOneHotCategorical(self.gs_temp, self.relax_temp if self.state_gs_perturb else eps,
        #         logits=logits,
        #     )
        # else:
        #     raise ValueError(f"Unknown dist_type {self.dist_type} !")

        # do sampling from dist
        # if dist is not None and self.test_sampling:
        #     samples = dist.rsample().view(N, flat_dim, M)
        #     _, samples = samples.max(dim=-1)
        # else:
        
        samples = torch.argmin(distances, dim=-1)
        samples = samples.movedim(1, 0).reshape(B, N, *spatial_shape)
        data = samples.detach().cpu().contiguous().numpy()
        indexes = torch.arange(N).unsqueeze(0).unsqueeze(-1)\
            .repeat(B, 1, spatial_dim).reshape(B, N, *spatial_shape)\
            .contiguous().numpy().astype(np.int32)

        data_bytes = self._encoder.encode_with_indexes(data, indexes)

        # store sample shape in header
        byte_head = [struct.pack("B", len(spatial_shape)+1)]
        byte_head.append(struct.pack("<H", B))
        for dim in spatial_shape:
            byte_head.append(struct.pack("<H", dim))
        byte_head.append(data_bytes)
        return b''.join(byte_head)

    def decode(self, byte_string, *args, **kwargs) -> Any:
        # decode shape from header
        num_shape_dims = struct.unpack("B", byte_string[:1])[0]
        flat_shape = []
        byte_ptr = 1
        for _ in range(num_shape_dims):
            flat_shape.append(struct.unpack("<H", byte_string[byte_ptr:(byte_ptr+2)])[0])
            byte_ptr += 2
        flat_dim = np.prod(flat_shape)
        batch_dim = flat_shape[0]
        spatial_shape = flat_shape[1:]
        spatial_dim = np.prod(spatial_shape)

        indexes = torch.arange(self.latent_dim).unsqueeze(0).unsqueeze(-1)\
            .repeat(batch_dim, 1, spatial_dim).reshape(batch_dim, self.latent_dim, *spatial_shape)\
            .contiguous().numpy().astype(np.int32)

        samples = self._decoder.decode_with_indexes(byte_string[byte_ptr:], indexes)

        samples = torch.as_tensor(samples).to(dtype=torch.long, device=self.device)
        samples = F.one_hot(samples, self.num_embeddings).float()
        samples = samples.movedim(1, 0).reshape(self.latent_dim, flat_dim, self.num_embeddings)

        quantized = self._sample_from_embedding(samples)
        quantized = quantized.view(self.latent_dim, batch_dim, spatial_dim, self.embedding_dim)
        quantized = quantized.permute(1, 0, 3, 2).reshape(batch_dim, self.latent_dim * self.embedding_dim, *spatial_shape)
        return quantized

    def update_state(self, *args, **kwargs) -> None:
        with torch.no_grad():
            if self.use_code_freq:
                prior_pmfs = torch.softmax(self.embedding_logprob, dim=-1)#.unsqueeze(-1)
            else:
                prior_pmfs = torch.ones(self.latent_dim, self.num_embeddings) / self.num_embeddings

            # TODO: autoregressive vq coding
            # categorical_dim = self.categorical_dim
            # if self.use_autoregressive_prior and self.ar_method == "finitestate":
            #     # TODO: this is a hard limit! may could be improved!
            #     if len(self.ar_offsets) > 2:
            #         pass
            #     else:
            #         lookup_table_shape = [self.latent_channels] + [categorical_dim+1] * len(self.ar_offsets) + [categorical_dim]
            #         ar_idx_all = list(itertools.product(range(self.categorical_dim+1), repeat=self.ar_window_size))
            #         ar_idx_all = torch.tensor(ar_idx_all, device=self.device).reshape(-1, 1).repeat(1, self.latent_channels)
            #         ar_input_all = self._finite_state_to_samples(ar_idx_all, add_default_samples=True).type_as(prior_logits)\
            #             .reshape(-1, self.ar_window_size, self.latent_channels, self.num_sample_params).movedim(1, -2)\
            #             .reshape(-1, self.latent_channels, self.ar_window_size*self.num_sample_params).movedim(1, 0)
            #         if self.ar_mlp_per_channel:
            #             ar_logits_reshape = torch.stack([mlp(ar_input) for (mlp, ar_input) in zip(self.fsar_mlps_per_channel, ar_input_all)], dim=0)
            #         else:
            #             ar_logits_reshape = self.fsar_mlp(ar_input_all)
            #         prior_logits = prior_logits.unsqueeze(-2) + ar_logits_reshape
            #         prior_logits = self._normalize_prior_logits(prior_logits)
            #         prior_logits = prior_logits.reshape(*lookup_table_shape)

            # prior_pmfs = prior_logits.exp()

            # TODO: customize freq precision
            if self.coder_type == "rans" or self.coder_type == "rans64":
                self._encoder = Rans64Encoder(freq_precision=self.coder_freq_precision)
                self._decoder = Rans64Decoder(freq_precision=self.coder_freq_precision)
            elif self.coder_type == "tans":
                self._encoder = TansEncoder(table_log=self.coder_freq_precision, max_symbol_value=self.categorical_dim-1)
                self._decoder = TansDecoder(table_log=self.coder_freq_precision, max_symbol_value=self.categorical_dim-1)
            else:
                raise NotImplementedError(f"Unknown coder_type {self.coder_type}!")

            prior_cnt = (prior_pmfs * (1<<self.coder_freq_precision)).clamp_min(1).reshape(-1, self.categorical_dim)
            prior_cnt = prior_cnt.detach().cpu().numpy().astype(np.int32)
            num_symbols = np.zeros(len(prior_cnt), dtype=np.int32) + self.categorical_dim
            offsets = np.zeros(len(prior_cnt), dtype=np.int32)

            self._encoder.init_params(prior_cnt, num_symbols, offsets)
            self._decoder.init_params(prior_cnt, num_symbols, offsets)

            # if self.use_autoregressive_prior and self.ar_method == "finitestate":
            #     ar_indexes = np.arange(len(prior_cnt), dtype=np.int32).reshape(1, *prior_pmfs.shape[:-1])

            #     self._encoder.init_ar_params(ar_indexes, [self.ar_offsets])
            #     self._decoder.init_ar_params(ar_indexes, [self.ar_offsets])


class SQVAEPriorCoder(NNPriorCoder):
    """Reimplemented SQVAE
    """    
    def __init__(self, param_var_q="gaussian_1", size_dict=512, dim_dict=64, log_param_q_init=3.0,
                 gs_temp=1.0, gs_temp_anneal=True):
        super(SQVAEPriorCoder, self).__init__()

        self.param_var_q = param_var_q
        self.size_dict = size_dict
        self.dim_dict = dim_dict

        self.gs_temp_anneal = gs_temp_anneal
        if gs_temp_anneal:
            self.gs_temp = nn.Parameter(torch.tensor(gs_temp), requires_grad=False)
        else:
            self.gs_temp = gs_temp

        # Codebook
        self.codebook = nn.Parameter(torch.randn(self.size_dict, self.dim_dict))
        self.log_param_q_scalar = nn.Parameter(torch.tensor(log_param_q_init))
        if self.param_var_q == "vmf":
            self.quantizer = VmfVectorQuantizer(
                self.size_dict, self.dim_dict, self.gs_temp)
        else:
            self.quantizer = GaussianVectorQuantizer(
                self.size_dict, self.dim_dict, self.gs_temp, self.param_var_q)
        
    
    def forward(self, z_from_encoder, flg_quant_det=True):
        # Encoding
        if self.param_var_q == "vmf":
            self.param_q = (self.log_param_q_scalar.exp() + torch.tensor([1.0], device="cuda"))
        else:
            self.param_q = (self.log_param_q_scalar.exp())
        
        # Quantization
        z_quantized, loss_latent, perplexity = self.quantizer(
            z_from_encoder, self.param_q, self.codebook, self.training, flg_quant_det)
        
        if self.training:
            self.update_cache("loss_dict",
                loss_rate = loss_latent,
            )
            self.update_cache("moniter_dict", 
                embedding_variance_mean=self.param_q.mean(),
            )

        self.update_cache("metric_dict", 
            perplexity=perplexity.sum() # / B
        )

        # TODO: add kl entropy metric?
        prior_entropy = math.log(self.size_dict) * (z_quantized.numel() / self.dim_dict) / z_quantized.shape[0]
        self.update_cache("metric_dict", 
            prior_entropy=prior_entropy,
        )

        return z_quantized
    
    def _calc_loss(self):
        raise NotImplementedError()
    