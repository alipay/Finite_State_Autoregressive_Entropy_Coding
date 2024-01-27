# try:
#     from compressai.ans import RansEncoder, RansDecoder
#     from compressai._CXX import pmf_to_quantized_cdf
# except:
#     print("Warning! compressai is not propoerly installed!")

import math
import struct
import itertools
from typing import Any, Dict, List, Tuple, Optional
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions

from .base import TorchQuantizedEntropyCoder
from cbench.nn.base import NNTrainableModule
from cbench.nn.utils import batched_cross_entropy

from cbench.nn.base import NNTrainableModule
from cbench.nn.layers import Upsample2DLayer, Downsample2DLayer, MaskedConv2d, MaskedConv3d
from cbench.modules.entropy_coder.rans import pmf_to_quantized_cdf_serial, pmf_to_quantized_cdf_batched
from cbench.utils.bytes_ops import encode_shape, decode_shape
from cbench.rans import BufferedRansEncoder, RansEncoder, RansDecoder, pmf_to_quantized_cdf


class AutoregressiveEntropyCoder(TorchQuantizedEntropyCoder, NNTrainableModule):
    """FSAR standalone implementation
    """    
    def __init__(self, *args,
        channel_dim=3, # categorical_dim=256,
        prior_trainable=False,
        use_autoregressive_prior=False, 
        ar_method="finitestate",
        ar_window_size=1, ar_offsets=None,
        ar_fs_method="table",
        ## for table based fsar
        ar_prior_decomp_method="sum", ar_prior_decomp_dim=None,
        ## for MLP based fsar
        ar_mlp_per_channel=False,
        # coding
        coder_type="rans", # current support "rans", "tans"
        fixed_input_shape : Optional[Tuple[int]] = None,
        **kwargs):
        super().__init__(*args, **kwargs)
        NNTrainableModule.__init__(self)

        self.channel_dim = channel_dim
        self.categorical_dim = self.data_precision
        self.total_channels = self.channel_dim * self.categorical_dim
        # total_channels = channel_dim * categorical_dim
        # super().__init__(in_channels, total_channels=total_channels, **kwargs)

        self.prior_trainable = prior_trainable

        self.use_autoregressive_prior = use_autoregressive_prior
        self.ar_method = ar_method
        self.ar_window_size = ar_window_size
        self.ar_offsets = ar_offsets
        self.ar_fs_method = ar_fs_method
        self.ar_prior_decomp_method = ar_prior_decomp_method
        self.ar_prior_decomp_dim = ar_prior_decomp_dim
        self.ar_mlp_per_channel = ar_mlp_per_channel
        # full ar
        if self.ar_window_size is None:
            self.ar_window_size = self.channel_dim - 1
        # custom ar offset setting
        if self.ar_offsets is None:
            self.ar_offsets = [(-offset,) for offset in range(1, self.ar_window_size+1)]
        else:
            self.ar_window_size = len(ar_offsets)


        self.coder_type = coder_type
        # TODO: temp fix for no rans fsar impl! Remove this after fsar-rans is done!
        if self.use_autoregressive_prior and self.ar_method == "finitestate":
            if self.coder_type == "rans":
                print("Warning! rans fsar is not implemented! switching to tans!")
                self.coder_type = "tans"
        self.fixed_input_shape = fixed_input_shape

        if use_autoregressive_prior and self.ar_method == "finitestate":
            if self.ar_fs_method == "table":
                if self.ar_prior_decomp_dim is not None:
                    # TODO: directly set ar_prior_categorical_dim?
                    self.ar_prior_categorical_dim = int(self.categorical_dim * self.ar_prior_decomp_dim)
                    # TODO: non-integer ar_prior_decomp_dim?
                    assert isinstance(self.ar_prior_decomp_dim, int)
                    ar_prior_dim = self.ar_prior_categorical_dim * self.ar_window_size
                    if self.ar_prior_decomp_method == "tucker":
                        self.ar_prior_tucker_core = nn.Parameter(torch.ones(self.ar_prior_decomp_dim ** self.ar_window_size) / self.ar_prior_decomp_dim)
                    elif self.ar_prior_decomp_method == "MLP3":
                        self.ar_mlps = nn.ModuleList(
                            [
                                nn.Sequential(
                                    nn.Linear(self.ar_prior_categorical_dim * self.ar_window_size, 2 * self.ar_prior_categorical_dim),
                                    nn.LeakyReLU(),
                                    nn.Linear(2 * self.ar_prior_categorical_dim, self.ar_prior_categorical_dim),
                                    nn.LeakyReLU(),
                                    nn.Linear(self.ar_prior_categorical_dim, self.categorical_dim),
                                )
                                for _ in range(self.channel_dim)
                            ]
                        )
                else:
                    self.ar_prior_categorical_dim = self.categorical_dim
                    # exponential lookup table
                    ar_prior_dim = self.categorical_dim
                    for _ in range(self.ar_window_size - 1):
                        ar_prior_dim *= self.categorical_dim
                prior_logprob = torch.zeros(channel_dim, ar_prior_dim, self.categorical_dim)
                # randomize to enable decomp matrix optimization
                if self.ar_prior_decomp_dim is not None:
                    prior_logprob.uniform_(-1, 1)
            # TODO: per-channel MLP may perform better
            elif self.ar_fs_method == "MLP3":
                if self.ar_mlp_per_channel:
                    self.fsar_mlps_per_channel = nn.ModuleList(
                        [
                            nn.Sequential(
                                nn.Linear((self.categorical_dim + 1) * self.ar_window_size, 2 * self.ar_window_size * (self.categorical_dim + 1)),
                                nn.LeakyReLU(),
                                nn.Linear(2 * self.ar_window_size * (self.categorical_dim + 1), 2 * self.categorical_dim),
                                nn.LeakyReLU(),
                                nn.Linear(2 * self.categorical_dim, self.categorical_dim),
                            )
                            for _ in range(self.channel_dim)
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

                # we do not really need this here, just a dummy for compability
                prior_logprob = torch.zeros(channel_dim, self.categorical_dim)
            else:
                raise NotImplementedError(f"Unknown self.ar_fs_method {self.ar_fs_method}")
        else:
            prior_logprob = torch.zeros(channel_dim, self.categorical_dim)

        # model based ar
        if self.use_autoregressive_prior:
            ar_model = None
            if self.ar_method == "maskconv3x3":
                ar_model = MaskedConv2d(self.total_channels, self.total_channels, 3, padding=1)
            elif self.ar_method == "maskconv5x5":
                ar_model = MaskedConv2d(self.total_channels, self.total_channels, 5, padding=2)
            elif self.ar_method == "maskconv3d3x3x3":
                ar_model = MaskedConv3d(self.categorical_dim, self.categorical_dim, 3, padding=1)
            elif self.ar_method == "maskconv3d5x5x5":
                ar_model = MaskedConv3d(self.categorical_dim, self.categorical_dim, 5, padding=2)
            elif self.ar_method == "checkerboard3x3":
                ar_model = MaskedConv2d(self.total_channels, self.total_channels, 3, padding=1, mask_type="Checkerboard")
            elif self.ar_method == "checkerboard5x5":
                ar_model = MaskedConv2d(self.total_channels, self.total_channels, 5, padding=2, mask_type="Checkerboard")

            if ar_model is not None:
                self.ar_model = nn.Sequential(
                    ar_model,
                    # nn.Conv2d(self.total_channels * 6 // 3, self.total_channels * 5 // 3, 1),
                    # nn.LeakyReLU(inplace=True),
                    # nn.Conv2d(self.total_channels * 5 // 3, self.total_channels * 4 // 3, 1),
                    # nn.LeakyReLU(inplace=True),
                    # nn.Conv2d(self.total_channels * 4 // 3, self.total_channels * 3 // 3, 1),
                )

        if prior_trainable:
            self.prior_logprob = nn.Parameter(prior_logprob)
        else:
            self.register_buffer("prior_logprob", prior_logprob, persistent=False)

        # initalize members for coding
        # self.update_state()

    def _merge_prior_logits_ar(self, prior_logits_ar):
        # prior_logits_ar : [batch_size, channel_dim, ar_dim, decomp_dim, categorical_dim]
        categorical_dim = self.categorical_dim # prior_logits_ar.shape[-1]
        # aggregate samples
        if self.ar_prior_decomp_method == "tucker":
            tucker_matrix = self.ar_prior_tucker_core.reshape(-1, self.ar_prior_decomp_dim).unsqueeze(0).unsqueeze(0)
            prior_logits_ar = prior_logits_ar.transpose(-2, -1)
            for w_offset in range(self.ar_window_size):
                tucker_matrix = torch.matmul(tucker_matrix, prior_logits_ar.select(2, w_offset).unsqueeze(-1)).squeeze(-1)
                if w_offset != self.ar_window_size -1:
                    tucker_matrix = tucker_matrix.reshape(*tucker_matrix.shape[:-1], -1, self.ar_prior_decomp_dim)
            prior_logits = tucker_matrix.squeeze(-1)
        elif self.ar_prior_decomp_method == "MLP3":
            prior_logits_ar = prior_logits_ar.reshape(-1, self.channel_dim, self.ar_window_size * self.ar_prior_decomp_dim * categorical_dim)
            prior_logits = torch.stack([mlp(logits.squeeze(1)) for mlp, logits in zip(self.ar_mlps, prior_logits_ar.split(1, dim=1))], dim=1)
        else:
            prior_logits = prior_logits_ar.sum(-2).sum(-2)
        return prior_logits

    def _normalize_prior_logits(self, prior_logits):
        prior_logits = torch.log_softmax(prior_logits, dim=-1)
        return prior_logits
    
    def forward(self, input : torch.Tensor, prior : torch.Tensor = None, *args, **kwargs):
        if self.prior_trainable:
            prior_logits = self.prior_logprob.unsqueeze(0)
        else:
            prior_logits = (torch.ones(1, self.channel_dim, self.categorical_dim) / self.categorical_dim).log()
        input_shape = input.shape
        # input_discrete = torch.as_tensor(self._data_preprocess(input), device=input.device, dtype=torch.long)
        input_discrete = ((input - self.data_range[0]) / self.data_step).long()
        ar_input = F.one_hot(input_discrete, self.categorical_dim).type_as(input)
        if self.use_autoregressive_prior:
            # if input_shape is None:
            #     input_shape = posterior_dist.logits.shape[:-1]
            batch_size = input_shape[0]
            spatial_shape = input_shape[2:]
            if self.ar_method == "finitestate":
                # find samples for ar
                # merge prior logits
                if self.ar_fs_method == "table":
                    autoregressive_samples = []
                    # for w_offset in range(self.ar_window_size):
                    for ar_offset in self.ar_offsets:
                        # take ar samples
                        ar_samples = ar_input
                        default_samples = torch.zeros_like(ar_input)
                        default_samples[..., 0] = 1
                        for data_dim, data_offset in enumerate(ar_offset):
                            if data_offset >= 0: continue
                            batched_data_dim = data_dim + 1
                            assert batched_data_dim != ar_samples.ndim - 1 # ar could not include categorical_dim
                            ar_samples = torch.cat((
                                # TODO: leave 0 as unknown sample, let total categories categorical_dim+1 (left for compability)
                                default_samples.narrow(batched_data_dim, 0, -data_offset),
                                ar_samples.narrow(batched_data_dim, 0, ar_input.shape[batched_data_dim]+data_offset)
                            ), dim=batched_data_dim)
                        # reshape back to sample format
                        ar_samples = ar_samples.movedim(1, -2).reshape_as(ar_input)
                        # ar_samples = torch.cat((
                        #     # first latent dim use default prior 0
                        #     F.one_hot(torch.zeros(posterior_dist.probs.shape[0], w_offset+1, dtype=torch.long).to(device=posterior_samples.device), self.categorical_dim),
                        #     # dims > 1 access prior according to the previous posterior
                        #     posterior_samples[:, :(self.channel_dim-1-w_offset)],
                        # ), dim=1)
                        autoregressive_samples.append(ar_samples)

                    if self.ar_prior_decomp_dim is not None:
                        # normalize logits to 0 mean
                        prior_logits = prior_logits - prior_logits.mean(-1, keepdim=True)
                        autoregressive_samples = torch.stack(autoregressive_samples, dim=-2)
                        prior_logits_ar = prior_logits.reshape(*prior_logits.shape[:-2], self.ar_window_size, self.ar_prior_decomp_dim, self.categorical_dim, self.categorical_dim)
                        prior_logits_ar = (prior_logits_ar * autoregressive_samples.unsqueeze(-2).unsqueeze(-1)).sum(-2)
                        # aggregate samples
                        prior_logits = self._merge_prior_logits_ar(prior_logits_ar)
                        # if self.ar_prior_decomp_method == "tucker":
                        #     tucker_matrix = self.ar_prior_tucker_core.reshape(-1, self.ar_prior_decomp_dim).unsqueeze(0).unsqueeze(0)
                        #     prior_logits_ar = prior_logits_ar.transpose(-2, -1)
                        #     for w_offset in range(self.ar_window_size):
                        #         tucker_matrix = torch.matmul(tucker_matrix, prior_logits_ar.select(2, w_offset).unsqueeze(-1)).squeeze(-1)
                        #         if w_offset != self.ar_window_size -1:
                        #             tucker_matrix = tucker_matrix.reshape(*tucker_matrix.shape[:-1], -1, self.ar_prior_decomp_dim)
                        #     prior_logits = tucker_matrix.squeeze(-1)
                        # elif self.ar_prior_decomp_method == "MLP3":
                        #     prior_logits_ar = prior_logits_ar.reshape(-1, self.channel_dim, self.ar_window_size * self.ar_prior_categorical_dim)
                        #     prior_logits = torch.stack([mlp(logits.squeeze(1)) for mlp, logits in zip(self.ar_mlps, prior_logits_ar.split(1, dim=1))], dim=1)
                        # else:
                        #     prior_logits = prior_logits_ar.sum(-2).sum(-2)
                    else:
                        aggregated_samples = autoregressive_samples[0]
                        for w_offset in range(self.ar_window_size-1):
                            cur_sample = autoregressive_samples[w_offset+1]
                            for _ in range(w_offset+1):
                                cur_sample = cur_sample.unsqueeze(-2)
                            aggregated_samples = aggregated_samples.unsqueeze(-1) * cur_sample
                        aggregated_samples = aggregated_samples.reshape(*ar_input.shape[:-1], -1)
                        prior_logits = torch.matmul(prior_logits.transpose(-2, -1), aggregated_samples.unsqueeze(-1)).squeeze(-1)
                elif self.ar_fs_method == "MLP3":
                    autoregressive_samples = []
                    for ar_offset in self.ar_offsets:
                        default_sample = torch.zeros_like(ar_input)[..., :1]
                        # take ar samples
                        ar_samples = torch.cat(
                            [
                                default_sample,
                                ar_input,
                            ], dim=-1
                        )
                        # leave 0 as unknown sample, let total categories categorical_dim+1
                        default_samples = torch.cat(
                            [
                                default_sample + 1,
                                torch.zeros_like(ar_input),
                            ], dim=-1
                        )
                        for data_dim, data_offset in enumerate(ar_offset):
                            if data_offset >= 0: continue
                            batched_data_dim = data_dim + 1
                            assert batched_data_dim != ar_samples.ndim - 1 # ar could not include categorical_dim
                            ar_samples = torch.cat((
                                default_samples.narrow(batched_data_dim, 0, -data_offset),
                                ar_samples.narrow(batched_data_dim, 0, ar_input.shape[batched_data_dim]+data_offset)
                            ), dim=batched_data_dim)
                        autoregressive_samples.append(ar_samples)
                    # [batch_size, self.channel_dim, *spatial_shape, self.ar_window_size*(self.categorical_dim+1)]
                    autoregressive_samples = torch.cat(autoregressive_samples, dim=-1)
                    if self.ar_mlp_per_channel:
                        autoregressive_samples_per_channel = autoregressive_samples.movedim(1, -2)\
                            .reshape(batch_size, *spatial_shape, self.channel_dim, self.ar_window_size*(self.categorical_dim+1))
                        ar_logits_reshape = torch.stack([mlp(sample.squeeze(-2)) for mlp, sample in zip(self.fsar_mlps_per_channel, autoregressive_samples_per_channel.split(1, dim=-2))], dim=-2)
                        prior_logits = ar_logits_reshape.reshape(-1, self.channel_dim, self.categorical_dim) + prior_logits
                    else:
                        autoregressive_samples_flat = autoregressive_samples.movedim(1, -2).reshape(-1, self.ar_window_size*(self.categorical_dim+1))
                        ar_logits_reshape = self.fsar_mlp(autoregressive_samples_flat)
                        # merge ar logits and prior logits
                        prior_logits = ar_logits_reshape.reshape(-1, self.channel_dim, self.categorical_dim) + prior_logits
            else:
                assert len(spatial_shape) == 2
                if self.ar_method.startswith("maskconv"):
                    if self.ar_method.startswith("maskconv3d"):
                        ar_input = ar_input.movedim(-1, 1)
                    else:
                        ar_input = ar_input.movedim(-1, 2).reshape(batch_size, self.total_channels, *spatial_shape)
                    prior_logits_reshape = self.ar_model(ar_input)
                    if self.ar_method.startswith("maskconv3d"):
                        prior_logits = prior_logits_reshape.movedim(1, -1).movedim(1, -2)\
                            .reshape(-1, self.channel_dim, self.categorical_dim)
                    else:
                        prior_logits = prior_logits_reshape.movedim(1, -1).reshape(-1, self.channel_dim, self.categorical_dim)
                
                elif self.ar_method.startswith("checkerboard"):
                    ar_input = ar_input.movedim(-1, 2).reshape(batch_size, self.channel_dim*self.categorical_dim, *spatial_shape)
                    prior_logits_reshape = self.ar_model(ar_input)
                    checkerboard_mask_h_0 = torch.arange(0, spatial_shape[-2], 2, dtype=torch.long, device=ar_input.device)
                    checkerboard_mask_h_1 = torch.arange(1, spatial_shape[-2], 2, dtype=torch.long, device=ar_input.device)
                    checkerboard_mask_w_0 = torch.arange(0, spatial_shape[-1], 2, dtype=torch.long, device=ar_input.device)
                    checkerboard_mask_w_1 = torch.arange(1, spatial_shape[-1], 2, dtype=torch.long, device=ar_input.device)
                    checkerboard_index_h_01, checkerboard_index_w_01 = torch.meshgrid(checkerboard_mask_h_0, checkerboard_mask_w_1)
                    checkerboard_index_h_10, checkerboard_index_w_10 = torch.meshgrid(checkerboard_mask_h_1, checkerboard_mask_w_0)
                    # multi-indexed tensor cannot be used as mutable left value
                    # prior_logits_reshape[..., checkerboard_mask_h_0, :][..., checkerboard_mask_w_1] = prior_dist.logits.reshape(1, self.total_channels, 1, 1)
                    # prior_logits_reshape[..., checkerboard_mask_h_1, :][..., checkerboard_mask_w_0] = prior_dist.logits.reshape(1, self.total_channels, 1, 1)
                    prior_logits_reshape[..., checkerboard_index_h_01, checkerboard_index_w_01] = prior_logits.reshape(1, self.total_channels, 1, 1)
                    prior_logits_reshape[..., checkerboard_index_h_10, checkerboard_index_w_10] = prior_logits.reshape(1, self.total_channels, 1, 1)
                    prior_logits = prior_logits_reshape.movedim(1, -1).reshape(-1, self.channel_dim, self.categorical_dim)
                else:
                    raise NotImplementedError(f"Unknown self.ar_method {self.ar_method}")
                # prior_logits = prior_logits_reshape.movedim(1, -1).reshape(-1, self.channel_dim, self.categorical_dim)
        
        ce = F.cross_entropy(prior_logits.reshape(-1, self.categorical_dim), 
                             input_discrete.movedim(1, -1).reshape(-1))
        if self.training:
            self.update_cache("loss_dict", 
                loss_distortion=ce,
            )
        self.update_cache("metric_dict", 
            prior_entropy=ce,
        )

        return input_discrete

    def encode(self, input : torch.Tensor, prior : torch.Tensor = None, **kwargs) -> bytes:
        batch_size = input.shape[0]
        channel_size = input.shape[1]
        spatial_shape = input.shape[2:]
        assert channel_size == self.channel_dim # * self.categorical_dim
        
        # posterior_dist = self.posterior_distribution(input.movedim(1, -1).reshape(-1, self.channel_dim, self.categorical_dim))
        # prior_dist = self.prior_distribution(prior=prior)

        # samples = self.sample_from_posterior(posterior_dist)

        # KLD = self.kl_divergence(prior_dist, posterior_dist, input_shape=(batch_size, self.channel_dim, *spatial_shape))

        samples = ((input - self.data_range[0]) / self.data_step).long()
        # input = input.view(batch_size, self.channel_dim, self.categorical_dim, *spatial_shape)
        
        # non-finite autoregressive
        data_bytes = b''
        if self.use_autoregressive_prior:
            # samples = torch.argmax(input, dim=2)
            input_one_hot = F.one_hot(samples, self.categorical_dim).type_as(input).movedim(-1, 2)\
                .reshape(batch_size, self.channel_dim*self.categorical_dim, *spatial_shape)
            if self.ar_method.startswith("maskconv"):
                if self.ar_method.startswith("maskconv3d"):
                    input_one_hot = input_one_hot.reshape(batch_size, self.channel_dim, self.categorical_dim, *spatial_shape)\
                        .permute(0, 2, 1, 3, 4)
                prior_logits_reshape = self.ar_model(input_one_hot)
                # move batched dimensions to last for correct decoding
                if self.ar_method.startswith("maskconv3d"):
                    prior_logits_reshape = prior_logits_reshape.movedim(0, -1)
                    samples = samples.movedim(0, -1)
                else:
                    prior_logits_reshape = prior_logits_reshape.reshape(batch_size, self.channel_dim, self.categorical_dim, *spatial_shape)
                    prior_logits_reshape = prior_logits_reshape.movedim(0, -1).movedim(0, -1)
                    samples = samples.movedim(0, -1).movedim(0, -1)
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
                prior_logits_reshape = self.ar_model(input_one_hot)
                checkerboard_mask_h_0 = torch.arange(0, spatial_shape[-2], 2, dtype=torch.long, device=input.device)
                checkerboard_mask_h_1 = torch.arange(1, spatial_shape[-2], 2, dtype=torch.long, device=input.device)
                checkerboard_mask_w_0 = torch.arange(0, spatial_shape[-1], 2, dtype=torch.long, device=input.device)
                checkerboard_mask_w_1 = torch.arange(1, spatial_shape[-1], 2, dtype=torch.long, device=input.device)
                # input_base = torch.cat([
                #     input_one_hot[..., checkerboard_mask_h_0, :][..., checkerboard_mask_w_1],
                #     input_one_hot[..., checkerboard_mask_h_1, :][..., checkerboard_mask_w_0],
                # ], dim=-1)
                # input_ar = torch.cat([
                #     input_one_hot[..., checkerboard_mask_h_0, :][..., checkerboard_mask_w_0],
                #     input_one_hot[..., checkerboard_mask_h_1, :][..., checkerboard_mask_w_1],
                # ], dim=-1)
                prior_logits_ar = torch.cat([
                    prior_logits_reshape[..., checkerboard_mask_h_0, :][..., checkerboard_mask_w_0],
                    prior_logits_reshape[..., checkerboard_mask_h_1, :][..., checkerboard_mask_w_1],
                ], dim=-1)
                prior_logits_ar = prior_logits_ar.reshape(batch_size, self.channel_dim, self.categorical_dim, *prior_logits_ar.shape[-2:]).movedim(2, -1)

                samples_base = torch.cat([
                    samples[..., checkerboard_mask_h_0, :][..., checkerboard_mask_w_1],
                    samples[..., checkerboard_mask_h_1, :][..., checkerboard_mask_w_0],
                ], dim=-1)
                data_base = samples_base.detach().cpu().numpy()
                indexes_base = torch.arange(self.channel_dim).unsqueeze(0).unsqueeze(-1)\
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

            # samples = torch.argmax(input, dim=2)
            data = samples.detach().cpu().numpy()
            # self._samples_cache = samples

            if self.coder_type == "rans":
                # TODO: autoregressive rans
                # if self.use_autoregressive_prior and self.ar_method == "finitestate":
                #     raise NotImplementedError()
                rans_encoder = RansEncoder()
                indexes = torch.arange(self.channel_dim).unsqueeze(0).unsqueeze(-1)\
                    .repeat(batch_size, 1, np.prod(spatial_shape)).reshape_as(samples).numpy()
                
                # prepare for coding
                data = data.astype(np.int32).reshape(-1)
                indexes = indexes.astype(np.int32).reshape(-1)
                cdfs = self._prior_cdfs
                cdf_sizes = np.array([len(cdf) for cdf in self._prior_cdfs])
                offsets = np.zeros(len(self._prior_cdfs))
                with self.profiler.start_time_profile("time_rans_encoder"):
                    data_bytes = rans_encoder.encode_with_indexes_np(
                        data, indexes, cdfs, cdf_sizes, offsets
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

    def decode(self, byte_string : bytes, prior : torch.Tensor = None, **kwargs) -> torch.Tensor:
        # if len(byte_string) == 0:
        #     return torch.zeros(1, self.channel_dim*self.categorical_dim, 8, 8, device=self.device)

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

        if self.use_autoregressive_prior:
            if self.ar_method.startswith("maskconv"):
                rans_decoder = RansDecoder()
                rans_decoder.set_stream(byte_string[byte_ptr:])
                samples = torch.zeros(batch_dim, self.channel_dim, *spatial_shape, dtype=torch.long, device=self.device)

                assert len(spatial_shape) == 2
                if self.ar_method.startswith("maskconv3d"):
                    c, h, w = (self.channel_dim, *spatial_shape)
                    for c_idx in range(c):
                        for h_idx in range(h):
                            for w_idx in range(w):
                                ar_input = F.one_hot(samples, self.categorical_dim).float().movedim(-1, 1)
                                prior_logits_ar = self.ar_model(ar_input).movedim(1, -1)[:, c_idx, h_idx, w_idx]
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
                                samples[:, c_idx, h_idx, w_idx] = samples_ar
                else:
                    h, w = spatial_shape
                    for h_idx in range(h):
                        for w_idx in range(w):
                                ar_input = F.one_hot(samples, self.categorical_dim).float().movedim(-1, 2).reshape(batch_dim, self.total_channels, *spatial_shape)
                                prior_logits_ar = self.ar_model(ar_input).reshape(batch_dim, self.channel_dim, self.categorical_dim, *spatial_shape).movedim(2, -1)[:, :, h_idx, w_idx]
                                prior_probs_ar = torch.softmax(prior_logits_ar, dim=-1)
                                cdfs_ar = pmf_to_quantized_cdf_batched(prior_probs_ar.reshape(-1, prior_probs_ar.shape[-1]))
                                cdfs_ar = cdfs_ar.detach().cpu().numpy().astype(np.int32)
                                indexes_ar = np.arange(len(cdfs_ar), dtype=np.int32)
                                cdf_sizes_ar = np.array([len(cdf) for cdf in cdfs_ar])
                                offsets_ar = np.zeros(len(indexes_ar)) # [0] * len(indexes)

                                samples_ar = rans_decoder.decode_stream_np(
                                    indexes_ar, cdfs_ar, cdf_sizes_ar, offsets_ar
                                )
                                samples_ar = torch.as_tensor(samples_ar, dtype=torch.long, device=self.device).reshape(-1, self.channel_dim)
                                samples[:, :, h_idx, w_idx] = samples_ar

                # warn about decoding error and fixit!
                if samples.max() >= self.categorical_dim or samples.min() < 0:
                    print("Decode error detected! The decompressed data may be corrupted!")
                    samples.clamp_max_(self.categorical_dim-1).clamp_min_(0)
                samples = F.one_hot(samples.movedim(1, -1), self.categorical_dim).float()
                samples = samples.reshape(batch_dim, *spatial_shape, self.channel_dim*self.categorical_dim)\
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
                indexes_base = torch.arange(self.channel_dim).unsqueeze(0).unsqueeze(-1)\
                    .repeat(batch_dim, 1, spatial_dim // 2).reshape(batch_dim, self.channel_dim, spatial_shape[0] // 2, spatial_shape[1])\
                    .numpy()

                # prepare for coding
                indexes_base = indexes_base.astype(np.int32).reshape(-1)
                cdfs_base = self._prior_cdfs
                cdf_sizes_base = np.array([len(cdf) for cdf in self._prior_cdfs])
                offsets_base = np.zeros(len(self._prior_cdfs))

                samples = torch.zeros(batch_dim, self.channel_dim, *spatial_shape, dtype=torch.long, device=self.device)
                with self.profiler.start_time_profile("time_rans_decoder"):
                    samples_base = rans_decoder.decode_stream_np(
                        indexes_base, cdfs_base, cdf_sizes_base, offsets_base
                    )
                    samples_base = torch.as_tensor(samples_base, dtype=torch.long, device=self.device)\
                        .reshape(batch_dim, self.channel_dim, spatial_shape[0] // 2, spatial_shape[1])
                    samples[..., checkerboard_index_h_01, checkerboard_index_w_01] = samples_base[..., :(spatial_shape[-1]//2)]
                    samples[..., checkerboard_index_h_10, checkerboard_index_w_10] = samples_base[..., (spatial_shape[-1]//2):]
                    ar_input = F.one_hot(samples.movedim(1, -1), self.categorical_dim).float()
                    ar_input = ar_input.reshape(batch_dim, *spatial_shape, self.channel_dim*self.categorical_dim)\
                        .movedim(-1, 1)
                    
                    prior_logits_reshape = self.ar_model(ar_input)
                    prior_logits_ar = torch.cat([
                        prior_logits_reshape[..., checkerboard_mask_h_0, :][..., checkerboard_mask_w_0],
                        prior_logits_reshape[..., checkerboard_mask_h_1, :][..., checkerboard_mask_w_1],
                    ], dim=-1)
                    prior_logits_ar = prior_logits_ar.reshape(batch_dim, self.channel_dim, self.categorical_dim, *prior_logits_ar.shape[-2:]).movedim(2, -1)
                    
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
                        .reshape(batch_dim, self.channel_dim, spatial_shape[0] // 2, spatial_shape[1])
                    samples[..., checkerboard_index_h_00, checkerboard_index_w_00] = samples_ar[..., :(spatial_shape[-1]//2)]
                    samples[..., checkerboard_index_h_11, checkerboard_index_w_11] = samples_ar[..., (spatial_shape[-1]//2):]

                # warn about decoding error and fixit!
                if samples.max() >= self.categorical_dim or samples.min() < 0:
                    print("Decode error detected! The decompressed data may be corrupted!")
                    samples.clamp_max_(self.categorical_dim-1).clamp_min_(0)
                samples = F.one_hot(samples.movedim(1, -1), self.categorical_dim).float()
                samples = samples.reshape(batch_dim, *spatial_shape, self.channel_dim*self.categorical_dim)\
                    .movedim(-1, 1)

                return samples

            else:
                pass

        # TODO: use iterative autoregressive for overwhelmed states
        if self.use_autoregressive_prior and self.ar_method == "finitestate" and len(self.ar_offsets) > 2:
            raise NotImplementedError("Overwhelmed states!")

        if self.coder_type == "rans":
            # TODO: autoregressive rans
            if self.use_autoregressive_prior and self.ar_method == "finitestate":
                raise NotImplementedError()
            rans_decoder = RansDecoder()
            indexes = torch.arange(self.channel_dim).unsqueeze(0).unsqueeze(-1)\
                .repeat(batch_dim, 1, spatial_dim).reshape(batch_dim, self.channel_dim, *spatial_shape)\
                .numpy()

            # prepare for coding
            indexes = indexes.astype(np.int32).reshape(-1)
            cdfs = self._prior_cdfs
            cdf_sizes = np.array([len(cdf) for cdf in self._prior_cdfs])
            offsets = np.zeros(len(self._prior_cdfs))
            with self.profiler.start_time_profile("time_rans_decoder"):
                samples = rans_decoder.decode_with_indexes_np(
                    byte_string[byte_ptr:], indexes, cdfs, cdf_sizes, offsets
                )

        samples = samples.reshape(batch_dim, self.channel_dim, *spatial_shape)
        # assert (samples == self._samples_cache).all()

        output = self._data_postprocess(samples)
        return output

    def update_state(self, *args, **kwargs) -> None:
        if self.prior_trainable:
            if self.use_autoregressive_prior and self.ar_method == "finitestate" and self.ar_fs_method == "table":
                prior_logits = self._normalize_prior_logits(self.prior_logprob.transpose(0,1)).transpose(0,1)
            else:
                prior_logits = self._normalize_prior_logits(self.prior_logprob)#.unsqueeze(-1)
        else:
            prior_logits = (torch.ones(self.channel_dim, self.categorical_dim) / self.categorical_dim).log()
        
        categorical_dim = self.categorical_dim # cat reduce moved after fsar
        if self.use_autoregressive_prior and self.ar_method == "finitestate":
            # TODO: this is a hard limit! may could be improved!
            if len(self.ar_offsets) > 2:
                pass
            else:
                if self.ar_fs_method == "table":
                    lookup_table_shape = [self.channel_dim] + [categorical_dim] * len(self.ar_offsets) + [categorical_dim]
                    if self.ar_prior_decomp_dim is None:
                        prior_logits = prior_logits.reshape(*lookup_table_shape)
                    else:
                        # normalize logits to 0 mean
                        prior_logits = prior_logits - prior_logits.mean(-1, keepdim=True)
                        # prior_logits = torch.log_softmax(prior_logits, dim=-1)#.unsqueeze(0)
                        prior_logits_ar = prior_logits.reshape(self.channel_dim, self.ar_window_size, self.ar_prior_decomp_dim, categorical_dim, categorical_dim)
                        cur_ar_idx = torch.zeros(len(self.ar_offsets), dtype=torch.long) # [0] * len(self.ar_offsets)
                        ar_dim_idx = torch.arange(len(self.ar_offsets))
                        # ar_idx_all = [cur_ar_idx]
                        prior_logits_ar_all = [prior_logits_ar[..., ar_dim_idx, :, cur_ar_idx, :]]
                        while True:
                            all_reset = True
                            for ar_idx in range(len(self.ar_offsets)):
                                cur_ar_idx[ar_idx] += 1
                                if cur_ar_idx[ar_idx] < categorical_dim:
                                    all_reset = False
                                    break
                                else:
                                    cur_ar_idx[ar_idx] = 0
                            if all_reset: break
                            # ar_idx_all.append(copy.deepcopy(cur_ar_idx))
                            prior_logits_ar_all.append(prior_logits_ar[..., ar_dim_idx, :, cur_ar_idx, :])
                        # ar_idx_all = torch.as_tensor(ar_idx_all, dtype=torch.long).unsqueeze(-1).unsqueeze(-1)
                        # prior_logits_ar = prior_logits_ar[ar_idx_all]
                        prior_logits_ar_all = torch.stack(prior_logits_ar_all).transpose(1, 2)
                        prior_logits = self._merge_prior_logits_ar(prior_logits_ar_all)
                        prior_logits = self._normalize_prior_logits(prior_logits)
                        prior_logits = prior_logits.reshape(*lookup_table_shape)
                elif self.ar_fs_method == "MLP3":
                    lookup_table_shape = [self.channel_dim] + [categorical_dim+1] * len(self.ar_offsets) + [categorical_dim]
                    ar_idx_all = list(itertools.product(range(self.categorical_dim+1), repeat=self.ar_window_size))
                    ar_idx_all = torch.tensor(ar_idx_all, device=self.device).reshape(-1)
                    ar_input_all = F.one_hot(ar_idx_all).type_as(prior_logits).reshape(-1, self.ar_window_size*(self.categorical_dim+1))
                    if self.ar_mlp_per_channel:
                        ar_logits_reshape = torch.stack([mlp(ar_input_all) for mlp in self.fsar_mlps_per_channel], dim=0)
                    else:
                        ar_logits_reshape = self.fsar_mlp(ar_input_all)
                    prior_logits = prior_logits.unsqueeze(-2) + ar_logits_reshape
                    prior_logits = self._normalize_prior_logits(prior_logits)
                    prior_logits = prior_logits.reshape(*lookup_table_shape)

        prior_pmfs = None

        if prior_pmfs is None:
            prior_pmfs = prior_logits.exp()
        if self.coder_type == "rans":
            self._prior_cdfs = pmf_to_quantized_cdf_serial(prior_pmfs.reshape(-1, categorical_dim))
            # self._prior_cdfs = np.array([pmf_to_quantized_cdf(pmf.tolist()) for pmf in prior_pmfs.reshape(-1, categorical_dim)])
            # TODO: rans fsar?
        else:
            raise NotImplementedError(f"Unknown coder_type {self.coder_type}!")

