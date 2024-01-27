# try:
#     from compressai.ans import RansEncoder, RansDecoder
#     from compressai._CXX import pmf_to_quantized_cdf
# except:
#     print("Warning! compressai is not propoerly installed!")

import math
import itertools
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions

from .base import TorchQuantizedEntropyCoder
from .ans import ANSEntropyCoder
from cbench.nn.base import NNTrainableModule
from cbench.nn.layers import Upsample2DLayer, Downsample2DLayer, MaskedConv2d, MaskedConv3d
from cbench.nn.utils import batched_cross_entropy
from cbench.nn.distributions.logistic import Logistic
from cbench.utils.ar_utils import create_ar_offsets, create_ar_offsets_multichannel
from cbench.ans import ar_linear_op, ar_limited_scaled_add_linear_op
from .rans import pmf_to_quantized_cdf_batched

import struct

class DistributionEntropyCoder(ANSEntropyCoder, NNTrainableModule):
    """ANS-based entropy coder with torch.distributions.Distribution implementation
    """
    def __init__(self, *args, 
        freq_precision=16,
        training_use_truncated_logits=False,
        dist_trainable=False,
        distortion_method="logprob",
        **kwargs):
        super().__init__(*args, **kwargs)
        NNTrainableModule.__init__(self)
        
        self.freq_precision = freq_precision
        self.freq_precision_total = 2**freq_precision

        self.training_use_truncated_logits = training_use_truncated_logits
        self.dist_trainable = dist_trainable
        self.distortion_method = distortion_method

        dist_params = self._init_dist_params()
        if dist_trainable:
            self.dist_params = nn.Parameter(dist_params)
        else:
            self.register_buffer("dist_params", dist_params, persistent=False)

        # self.update_state()

    def _param_to_dist(self, params) -> distributions.Distribution:
        raise NotImplementedError()

    def _init_dist_params(self) -> torch.Tensor:
        raise NotImplementedError()

    def _truncated_dist_to_logits(self, dist : distributions.Distribution) -> torch.Tensor:
        raise NotImplementedError()

    def _select_best_indexes(self, prior) -> torch.LongTensor:
        raise NotImplementedError()
    
    # def _data_preprocess_with_prior(self, data, prior):
    #     data = self._data_preprocess(data)
    #     # NOTE: by default prior is not used
    #     return data

    # def _data_postprocess_with_prior(self, data, prior):
    #     data = self._data_postprocess(data)
    #     # NOTE: by default prior is not used
    #     return data

    def _quantize_prior(self, prior) -> torch.Tensor:
        indexes = self._select_best_indexes(prior)
        dist_params = self.dist_params if self.dist_trainable else self._init_dist_params().type_as(self.dist_params)
        quantized_prior = dist_params[indexes.reshape(-1)]
        return quantized_prior.reshape_as(prior)

    def _get_ans_params(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.dist_trainable:
            dists = self._param_to_dist(self._init_dist_params())
        else:
            dists = self._param_to_dist(self.dist_params)
        logits = self._truncated_dist_to_logits(dists)
        prior_pmfs = torch.softmax(logits, dim=-1)#.unsqueeze(-1)

        prior_cnt = (prior_pmfs * (1<<self.freq_precision)).clamp_min(1).reshape(-1, self.data_precision)
        prior_cnt = prior_cnt.detach().cpu().contiguous().numpy().astype(np.int32)
        num_symbols = np.zeros(len(prior_cnt), dtype=np.int32) + self.data_precision
        offsets = np.zeros(len(prior_cnt), dtype=np.int32)

        return prior_cnt, num_symbols, offsets

    def forward(self, data, *args, prior=None, **kwargs):
        # prior = 1 / torch.zeros_like(prior)
        # NOTE: Distribution class will throw error with NaN prior! Skip if NaN detected!
        if torch.isnan(prior).any():
            print("Nan detected! Skipping")
            return data

        prior_dist = self._param_to_dist(prior)
        if self.training_use_truncated_logits:
            prior_logits = self._truncated_dist_to_logits(prior_dist).movedim(-1, 2)
            data_logprob = batched_cross_entropy(prior_logits.reshape(prior_logits.shape[0], -1, *prior_logits.shape[3:]),
                data, 
                min=self.data_range[0], 
                max=self.data_range[1]
            )
        else:
            if self.distortion_method == "logprob":
                data_logprob = -prior_dist.log_prob(data.unsqueeze(-1)) - math.log(self.data_step)
            elif self.distortion_method == "cdf_delta":
                data_prob = prior_dist.cdf(data.unsqueeze(-1) + self.data_step / 2) - prior_dist.cdf(data.unsqueeze(-1) - self.data_step / 2)
                data_logprob = -torch.log(data_prob + 1e-7)
            else:
                raise NotImplementedError()
        if self.training:
            self.update_cache("loss_dict",
                loss_distortion=data_logprob.mean(), # loss_distortion is just a name convention (maybe logprob is better?)
            )
        self.update_cache("metric_dict",
            estimated_x_epd=data_logprob.mean(),
            estimated_x_entropy=data_logprob.sum() / data.shape[0],
        )

        
        # TODO: maybe implement this in ans coder?
        # NOTE: maybe incorrect if dist is trainable!
        if not self.training:
            if not hasattr(self, "_ans_freqs"):
                freqs, nfreqs, offsets = self._get_ans_params()
                if self.coder_type == "rans" or self.coder_type == "rans64":
                    from .ans import Rans64Encoder
                    encoder = Rans64Encoder(freq_precision=self.freq_precision, bypass_coding=self.use_bypass_coding)
                    encoder.init_params(freqs, nfreqs, offsets)
                    cdfs = encoder.get_cdfs()
                    freqs = cdfs[:, 1:] - cdfs[:, :-1]
                self._ans_freqs = torch.as_tensor(freqs).type_as(prior)
            data_tmp = self._data_preprocess_with_prior(data, prior, to_numpy=False) % self.data_precision
            indexes = self._select_best_indexes(prior)
            # self._prior_pdfs_tensor = self._prior_pdfs_tensor.type_as(indexes)
            estimated_coding_entropy = -torch.log((self._ans_freqs[indexes.reshape(-1), data_tmp.reshape(-1)]) / self.freq_precision_total)
            self.update_cache("metric_dict",
                estimated_x_coding_epd=torch.mean(estimated_coding_entropy),
            )
        return data


class AutoregressiveDistributionEntropyCoder(DistributionEntropyCoder):
    """Added autoregressive entropy coding interface
    """
    def _autoregressive_prior(self, data : torch.Tensor, prior : torch.Tensor = None, **kwargs) -> torch.Tensor:
        raise NotImplementedError()
    
    def _merge_ar_prior(self, ar_prior : torch.Tensor, prior : torch.Tensor = None, **kwargs):
        return ar_prior

    def forward(self, data, *args, prior=None, **kwargs):
        if not self.training:
            prior = self._quantize_prior(prior)
        ar_prior = self._autoregressive_prior(data, prior)
        prior = self._merge_ar_prior(ar_prior, prior)
        return super().forward(data, *args, prior=prior, **kwargs)


class AutoregressiveImplDistributionEntropyCoder(AutoregressiveDistributionEntropyCoder):
    """Added autoregressive entropy coding implementation. Similar to prior_coders : AutoregressivePriorImplDistributionPriorCoder.
    Added ar_method linear/twar*.
    * https://openaccess.thecvf.com/content/CVPR2022/papers/Kang_PILC_Practical_Image_Lossless_Compression_With_an_End-to-End_GPU_Oriented_CVPR_2022_paper.pdf
    """
    def __init__(self, channel_dim=3, num_prior_params=2,
                 # ar prior
                 prior_trainable=False,
                 use_autoregressive_prior=False, 
                 # ar method config
                 ar_method="finitestate", ar_input_detach=False, # ar_input_sample=True, ar_input_straight_through=False,
                 # ar offset config
                 ar_window_size=1, ar_offsets=None, ar_offsets_per_channel=False,
                 # ar model config
                 ar_input_prior=False, ar_output_as_mean_offset=False,
                 ar_train_dropout=0.0, ar_default_sample=None,
                 ar_mlp_per_channel=False, ar_mlp_bottleneck_expansion=2,
                 # ar coding
                 ar_coding_use_default_bias=True,
                 **kwargs):
        self.channel_dim = channel_dim
        self.num_prior_params = num_prior_params
        self.ar_coding_use_default_bias = ar_coding_use_default_bias

        super().__init__(**kwargs)
        self.prior_trainable = prior_trainable
        prior_params = torch.zeros(self.channel_dim, self.num_prior_params)
        if prior_trainable:
            self.prior_params = nn.Parameter(prior_params)
        else:
            self.register_buffer("prior_params", prior_params, persistent=False)

        self.use_autoregressive_prior = use_autoregressive_prior
        self.ar_method = ar_method
        self.ar_input_detach = ar_input_detach
        self.ar_window_size = ar_window_size
        self.ar_offsets = ar_offsets
        self.ar_offsets_per_channel = ar_offsets_per_channel
        self.ar_input_prior = ar_input_prior
        self.ar_output_as_mean_offset = ar_output_as_mean_offset
        self.ar_train_dropout = ar_train_dropout
        self.ar_default_sample = ar_default_sample if ar_default_sample is not None else (self.data_range[0] - self.data_range[1])
        self.ar_mlp_per_channel = ar_mlp_per_channel
        # custom ar offset setting
        if self.ar_offsets is None:
            self.ar_offsets = np.array([(-offset,) for offset in range(1, self.ar_window_size+1)])
        else:
            self.ar_offsets = np.array(self.ar_offsets)
            self.ar_window_size = len(self.ar_offsets)
            if ar_offsets_per_channel:
                # if self.ar_offsets.ndim == 2:
                #     self.ar_offsets = np.repeat(self.ar_offsets, self.channel_dim)
                assert self.ar_offsets.ndim == 3
                assert self.ar_offsets.shape[1] == self.channel_dim

        if use_autoregressive_prior:
            # ar_offset buffer for faster training
            self.register_buffer("_ar_offsets_buffer", torch.zeros(len(self.ar_offsets), 1, self.channel_dim, 1, 1), persistent=False)

            # create ar models
            ar_input_channels = 1 
            mlp_input_channels = ar_input_channels * self.ar_window_size
            if self.ar_input_prior:
                mlp_input_channels += self.num_prior_params
            ar_output_channels = 1 if self.ar_output_as_mean_offset else self.num_prior_params
            self.ar_output_channels = ar_output_channels
            if self.ar_method == "finitestate":
                if self.ar_mlp_per_channel:
                    self.fsar_mlps_per_channel = nn.ModuleList(
                        [
                            nn.Sequential(
                                nn.Linear(mlp_input_channels, int(ar_mlp_bottleneck_expansion * self.ar_window_size * ar_input_channels)),
                                nn.LeakyReLU(),
                                nn.Linear(int(ar_mlp_bottleneck_expansion * self.ar_window_size * ar_input_channels), int(ar_mlp_bottleneck_expansion * ar_output_channels)),
                                nn.LeakyReLU(),
                                nn.Linear(int(ar_mlp_bottleneck_expansion * ar_output_channels), ar_output_channels),
                            )
                            for _ in range(self.channel_dim)
                        ]
                    )
                else:
                    self.fsar_mlp = nn.Sequential(
                        nn.Linear(mlp_input_channels, int(ar_mlp_bottleneck_expansion * self.ar_window_size * ar_input_channels)),
                        nn.LeakyReLU(),
                        nn.Linear(int(ar_mlp_bottleneck_expansion * self.ar_window_size * ar_input_channels), int(ar_mlp_bottleneck_expansion * ar_output_channels)),
                        nn.LeakyReLU(),
                        nn.Linear(int(ar_mlp_bottleneck_expansion * ar_output_channels), ar_output_channels),
                    )
            elif self.ar_method == "linear":
                self.ar_model = nn.Conv1d(self.channel_dim, self.channel_dim*ar_output_channels, mlp_input_channels, groups=self.channel_dim)
                # self.ar_model = nn.Linear(self.channel_dim, self.channel_dim*ar_output_channels)
            elif self.ar_method == "twar":
                self.predictor_offsets = [
                    [(0, -1, -1),
                    (0, 0, -1),
                    (0, -1, 0),],
                    [(-1, 0, 0),
                    (-1, 0, -1),
                    (0, 0, -1),],
                    [(-1, 0, 0),
                    (-1, 0, -1),
                    (0, 0, -1),],
                ]
                predictor_models = []
                for pred in self.predictor_offsets:
                    predictor_models.append(nn.Linear(len(pred), 1))
                self.predictor_models = nn.ModuleList(predictor_models)

        # model based ar
        if self.use_autoregressive_prior:
            ar_model = None
            if self.ar_method.startswith("maskconv3d"):
                input_channels = ar_input_channels
            else:
                input_channels = ar_input_channels * self.channel_dim
            if self.ar_input_prior:
                input_channels += ar_output_channels
            if self.ar_method == "maskconv3x3":
                ar_model = MaskedConv2d(input_channels, ar_output_channels * self.channel_dim, 3, padding=1)
            elif self.ar_method == "maskconv5x5":
                ar_model = MaskedConv2d(input_channels, ar_output_channels * self.channel_dim, 5, padding=2)
            elif self.ar_method == "maskconv3d3x3x3":
                ar_model = MaskedConv3d(input_channels, ar_output_channels, 3, padding=1)
            elif self.ar_method == "maskconv3d5x5x5":
                ar_model = MaskedConv3d(input_channels, ar_output_channels, 5, padding=2)
            elif self.ar_method == "checkerboard3x3":
                ar_model = MaskedConv2d(input_channels, ar_output_channels * self.channel_dim, 3, padding=1, mask_type="Checkerboard")
            elif self.ar_method == "checkerboard5x5":
                ar_model = MaskedConv2d(input_channels, ar_output_channels * self.channel_dim, 5, padding=2, mask_type="Checkerboard")

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


    def _get_ar_indices(self, prior) -> torch.Tensor:
        indices_shape = prior.shape[:-1]
        indices = torch.zeros_like(prior[..., 0]).reshape(prior.shape[0], prior.shape[1], -1)
        if self.ar_offsets_per_channel or self.ar_mlp_per_channel:
            indices = torch.arange(prior.shape[1], dtype=torch.int32).reshape(1, indices.shape[1], 1).expand_as(indices)
        return indices.reshape(*indices_shape)

    def _get_ar_params(self, prior) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if self.use_autoregressive_prior:
            ar_indices = self._get_ar_indices(prior).contiguous().numpy()
            if self.ar_offsets_per_channel:
                ar_offsets = create_ar_offsets_multichannel(prior.shape[:-1], self.ar_offsets)
            else:
                ar_offsets = create_ar_offsets(prior.shape[:-1], self.ar_offsets)
            return ar_indices, ar_offsets
        else:
            return super()._get_ar_params(prior)

    def _default_sample(self, samples : torch.Tensor = None) -> torch.Tensor:
        # take a sample outside data range
        return torch.zeros_like(samples).unsqueeze(-1) + self.ar_default_sample
    
    # def _finite_state_to_samples(self, states : torch.Tensor = None) -> torch.Tensor:
    #     raise NotImplementedError()

    def _get_prediction(self, data):
        pred = torch.zeros_like(data)
        data = F.pad(data, (1, 0, 1, 0), "constant", 0)
        for c in range(3):
            if c==0:
                pred_src = torch.stack([
                    # torch.zeros_like(data[..., c:(c+1), :-1, :-1]),
                    data[..., c:(c+1), :-1, :-1],
                    data[..., c:(c+1), :-1, 1:],
                    data[..., c:(c+1), 1:, :-1],
                ], dim=-1)
            else:
                pred_src = torch.stack([
                    data[..., (c-1):c, 1:, 1:],
                    data[..., (c-1):c, 1:, :-1],
                    # data[..., c:(c+1), :-1, 1:],
                    data[..., c:(c+1), 1:, :-1],
                ], dim=-1)
            pred[..., c, :, :] = self.predictor_models[c](pred_src.reshape(-1, 3)).reshape_as(pred[..., c, :, :])
        return pred
    
    # NOTE: should be overridden! index should be properly included in linear op
    def _get_linear_ops(self):
        weight, bias = self.ar_model.weight, self.ar_model.bias
        ar_ops = []
        for i in range(self.channel_dim):
            ar_op = ar_linear_op([1.0] + weight[i].tolist(), bias[i].item())
            ar_ops.append(ar_op)
        return ar_ops

    def _autoregressive_prior(self, data : torch.Tensor, prior : torch.Tensor = None, **kwargs) -> torch.Tensor:
        if prior is None:
            prior = torch.zeros(data.shape[0], data.shape[1]*self.ar_output_channels, *data.shape[2:]).type_as(self.prior_params)
            prior.index_fill_(1, torch.arange(data.shape[1]*self.ar_output_channels, dtype=torch.long, device=self.device), self.prior_params.reshape(-1))
            prior = prior.reshape(data.shape[0], data.shape[1], self.ar_output_channels, *data.shape[2:]).movedim(2, -1)
        # TODO: process prior parameter if exists!
        if self.use_autoregressive_prior:
            # if input_shape is None:
            #     input_shape = posterior_dist.logits.shape[:-1]
            batch_size = data.shape[0]
            spatial_shape = data.shape[2:]
            if self.ar_input_detach:
                data = data.detach()
            if self.ar_method == "finitestate" or self.ar_method == "linear":
                # find samples for ar
                autoregressive_samples = []
                if self.ar_input_prior:
                    autoregressive_samples.append(prior)
                # for ar_offset in self.ar_offsets:
                #     default_samples = self._default_sample(data)
                #     ar_samples = data.unsqueeze(-1)
                #     for data_dim, data_offset in enumerate(ar_offset):
                #         if data_offset >= 0: continue
                #         batched_data_dim = data_dim + 1
                #         assert batched_data_dim != ar_samples.ndim - 1 # ar could not include ar_input_channels
                #         ar_samples = torch.cat((
                #             default_samples.narrow(batched_data_dim, 0, -data_offset),
                #             ar_samples.narrow(batched_data_dim, 0, data.shape[batched_data_dim]+data_offset)
                #         ), dim=batched_data_dim)
                #     autoregressive_samples.append(ar_samples)
                # NOTE: this is more consistent with compression, but slower on gpu
                if self._ar_offsets_buffer.shape[2:] != prior.shape[1:-1]:
                    if self.ar_offsets_per_channel:
                        ar_offsets = create_ar_offsets_multichannel(prior.shape[:-1], self.ar_offsets)
                    else:
                        ar_offsets = create_ar_offsets(prior.shape[:-1], self.ar_offsets)
                    ar_offsets = torch.as_tensor(ar_offsets, device=prior.device)
                    self._ar_offsets_buffer = ar_offsets[:, :1]
                    ar_offsets = ar_offsets.reshape(ar_offsets.shape[0], -1)
                else:
                    ar_offsets = self._ar_offsets_buffer.reshape(self._ar_offsets_buffer.shape[0], -1).repeat(1, prior.shape[0])
                ar_ptr_base = torch.arange(ar_offsets.shape[-1], device=prior.device)
                default_samples = self._default_sample(data).reshape(-1)
                for ar_offset in ar_offsets:
                    ar_ptr = ar_ptr_base - ar_offset
                    ar_samples = torch.where(ar_offset > 0, data.reshape(-1)[ar_ptr], default_samples)
                    autoregressive_samples.append(ar_samples.reshape_as(data).unsqueeze(-1))
                
                # [batch_size, self.channel_dim, *spatial_shape, self.ar_window_size*self.ar_input_channels]
                autoregressive_samples = torch.cat(autoregressive_samples, dim=-1)
                if self.ar_method == "finitestate":
                    if self.ar_mlp_per_channel:
                        autoregressive_samples_per_channel = autoregressive_samples.movedim(1, -2)\
                            .reshape(-1, self.channel_dim, autoregressive_samples.shape[-1])
                        ar_logits = torch.stack([mlp(sample.squeeze(1)) for mlp, sample in zip(self.fsar_mlps_per_channel, autoregressive_samples_per_channel.split(1, dim=1))], dim=1)
                    else:
                        autoregressive_samples_flat = autoregressive_samples.movedim(1, -2).reshape(-1, autoregressive_samples.shape[-1])
                        ar_logits = self.fsar_mlp(autoregressive_samples_flat)
                elif self.ar_method == "linear":
                    autoregressive_samples_per_channel = autoregressive_samples.movedim(1, -2)\
                        .reshape(-1, self.channel_dim, autoregressive_samples.shape[-1])
                    ar_logits = self.ar_model(autoregressive_samples_per_channel)#.reshape(-1, self.channel_dim, self.ar_output_channels).movedim(-2, -1)
                ar_prior = ar_logits.reshape(batch_size, *spatial_shape, self.channel_dim, self.ar_output_channels).movedim(-2, 1)
                # if self.ar_output_as_mean_offset:
                #     # NOTE: by default we treat prior[..., 0] as mean!
                #     prior[..., 0] = prior[..., 0] - ar_logits[..., 0]
                # else:
                #     if self.ar_input_prior:
                #         prior = ar_logits
                #     else:
                #         if self.training and self.ar_train_dropout > 0.0:
                #             dropout_mask = torch.rand_like(ar_logits).mean(dim=-1, keepdim=True) > self.ar_train_dropout
                #             ar_logits = ar_logits * dropout_mask
                #         prior = prior + ar_logits
            elif self.ar_method == "twar":
                ar_prior = self._get_prediction(data)
                # prior[..., 0] = prior[..., 0] - pred
            else:
                # TODO: add prior to ar input
                assert len(spatial_shape) == 2
                if self.ar_method.startswith("maskconv"):
                    if self.ar_method.startswith("maskconv3d"):
                        posterior_samples_reshape = posterior_samples_reshape.reshape(batch_size, self.channel_dim, self.num_sample_params, *spatial_shape)\
                            .permute(0, 2, 1, 3, 4)
                    ar_logits_reshape = self.ar_model(posterior_samples_reshape)
                    if self.ar_method.startswith("maskconv3d"):
                        ar_logits_reshape = ar_logits_reshape.permute(0, 2, 1, 3, 4)\
                            .reshape(batch_size, self.channel_dim, *spatial_shape)
                elif self.ar_method.startswith("checkerboard"):
                    ar_logits_reshape = self.ar_model(posterior_samples_reshape)
                    checkerboard_mask_h_0 = torch.arange(0, spatial_shape[-2], 2, dtype=torch.long, device=posterior_samples_reshape.device)
                    checkerboard_mask_h_1 = torch.arange(1, spatial_shape[-2], 2, dtype=torch.long, device=posterior_samples_reshape.device)
                    checkerboard_mask_w_0 = torch.arange(0, spatial_shape[-1], 2, dtype=torch.long, device=posterior_samples_reshape.device)
                    checkerboard_mask_w_1 = torch.arange(1, spatial_shape[-1], 2, dtype=torch.long, device=posterior_samples_reshape.device)
                    checkerboard_index_h_01, checkerboard_index_w_01 = torch.meshgrid(checkerboard_mask_h_0, checkerboard_mask_w_1)
                    checkerboard_index_h_10, checkerboard_index_w_10 = torch.meshgrid(checkerboard_mask_h_1, checkerboard_mask_w_0)
                    # multi-indexed tensor cannot be used as mutable left value
                    # ar_logits_reshape[..., checkerboard_mask_h_0, :][..., checkerboard_mask_w_1] = prior_dist.logits.reshape(1, self.channel_dim, 1, 1)
                    # ar_logits_reshape[..., checkerboard_mask_h_1, :][..., checkerboard_mask_w_0] = prior_dist.logits.reshape(1, self.channel_dim, 1, 1)
                    # TODO: default prior params?
                    ar_logits_reshape[..., checkerboard_index_h_01, checkerboard_index_w_01] = self.prior_params.reshape(1, -1, 1, 1).repeat(ar_logits_reshape.shape[0], 1, ar_logits_reshape.shape[2]//2, ar_logits_reshape.shape[3]//2)
                    ar_logits_reshape[..., checkerboard_index_h_10, checkerboard_index_w_10] = self.prior_params.reshape(1, -1, 1, 1).repeat(ar_logits_reshape.shape[0], 1, ar_logits_reshape.shape[2]//2, ar_logits_reshape.shape[3]//2)
                else:
                    raise NotImplementedError(f"Unknown self.ar_method {self.ar_method}")
                ar_prior = ar_logits_reshape.movedim(1, -1).reshape(data.shape[0], self.channel_dim, self.ar_output_channels)

            return ar_prior
        else:
            return prior
    
    def _merge_ar_prior(self, ar_prior: torch.Tensor, prior: torch.Tensor = None, **kwargs):
        if self.ar_method == "twar":
            # NOTE: twar only allows self.ar_output_as_mean_offset==True, by default we treat prior[..., 0] as mean!
            prior[..., 0] = prior[..., 0] + ar_prior[..., 0]
        else:
            if self.ar_output_as_mean_offset:
                # NOTE: by default we treat prior[..., 0] as mean!
                prior[..., 0] = prior[..., 0] + ar_logits[..., 0]
            else:
                if self.ar_method == "finitestate" and not self.ar_input_prior:
                    if self.training and self.ar_train_dropout > 0.0:
                        dropout_mask = torch.rand_like(ar_logits).mean(dim=-1, keepdim=True) > self.ar_train_dropout
                        ar_logits = ar_logits * dropout_mask
                    prior = prior + ar_logits
                else:
                    prior = ar_logits
        return prior
    
    def encode(self, data, *args, prior=None, **kwargs):
        if self.use_autoregressive_prior:
            if self.ar_method == "finitestate":
                # NOTE: get the true default prior
                if self.ar_coding_use_default_bias:
                    prior_shape = prior.shape
                    prior = prior.movedim(1, -2).reshape(prior_shape[0], -1, self.channel_dim, prior_shape[-1]) + self.default_ar_prior.unsqueeze(0).unsqueeze(0)
                    prior = prior.movedim(-2, 1).reshape(*prior_shape)
            elif self.ar_method == "linear":
                pass
            else:
                raise NotImplementedError("Non-FSAR model based coding are not supported!")
        return super().encode(data, *args, prior=prior, **kwargs)
    
    def decode(self, byte_string: bytes, *args, prior=None, data_length=None, **kwargs):
        if self.use_autoregressive_prior:
            if self.ar_method == "finitestate":
                # NOTE: get the true default prior
                if self.ar_coding_use_default_bias:
                    prior_shape = prior.shape
                    prior = prior.movedim(1, -2).reshape(prior_shape[0], -1, self.channel_dim, prior_shape[-1]) + self.default_ar_prior.unsqueeze(0).unsqueeze(0)
                    prior = prior.movedim(-2, 1).reshape(*prior_shape)
            elif self.ar_method == "linear":
                pass
            else:
                raise NotImplementedError("Non-FSAR model based coding are not supported!")
        return super().decode(byte_string, *args, prior=prior, data_length=data_length, **kwargs)

    def update_state(self, *args, **kwargs) -> None:
        super().update_state(*args, **kwargs)
        with torch.no_grad():
            if not self.dist_trainable:
                dist_params = self._init_dist_params()
            else:
                dist_params = self.dist_params
            # logits = self._truncated_dist_to_logits(dists)
            # prior_pmfs = torch.softmax(logits, dim=-1)#.unsqueeze(-1)

            categorical_dim = self.data_precision
            num_dists = len(dist_params)
            num_sample_params = 1
            if self.use_autoregressive_prior:
                if self.ar_method == "finitestate":
                    assert self.ar_input_prior == False
                    # TODO: this is a hard limit! may could be improved!
                    if len(self.ar_offsets) > 2:
                        pass
                    else:
                        lookup_table_shape = [self.channel_dim, num_dists] + [categorical_dim+1] * len(self.ar_offsets)
                        ar_states = self._data_postprocess(np.arange(categorical_dim))
                        ar_states = torch.cat([self._default_sample(ar_states)[0], ar_states], dim=0)
                        ar_input_all = list(itertools.product(ar_states.tolist(), repeat=self.ar_window_size))
                        ar_input_all = torch.tensor(ar_input_all, dtype=ar_states.dtype, device=self.device)
                        # ar_input_all = ar_input_all.repeat(1, self.channel_dim)\
                        #     .reshape(-1, self.ar_window_size, self.channel_dim, num_sample_params).movedim(1, -2).movedim(1, 0)\
                        #     .reshape(self.channel_dim, -1, self.ar_window_size*num_sample_params)
                        ar_input_all = ar_input_all.unsqueeze(0).repeat(self.channel_dim, 1, 1)
                        if self.ar_mlp_per_channel:
                            ar_prior_reshape = torch.stack([mlp(ar_input) for (mlp, ar_input) in zip(self.fsar_mlps_per_channel, ar_input_all)], dim=0)
                        else:
                            ar_prior_reshape = self.fsar_mlp(ar_input_all)
                        if self.ar_coding_use_default_bias:
                            default_ar_input = self._default_sample(ar_input_all).squeeze(-1)
                            if self.ar_mlp_per_channel:
                                default_ar_prior_reshape = torch.stack([mlp(ar_input) for (mlp, ar_input) in zip(self.fsar_mlps_per_channel, default_ar_input)], dim=0)
                            else:
                                default_ar_prior_reshape = self.fsar_mlp(default_ar_input)
                            self.default_ar_prior = default_ar_prior_reshape[:, 0, :]
                            ar_prior_reshape = ar_prior_reshape - default_ar_prior_reshape

                        prior_params = dist_params.unsqueeze(1).unsqueeze(0).type_as(ar_prior_reshape) + ar_prior_reshape.unsqueeze(1) #- default_ar_prior_reshape.unsqueeze(1)
                        ar_indexes = self._select_best_indexes(prior_params)
                        ar_indexes = ar_indexes.reshape(*lookup_table_shape).detach().cpu().contiguous().numpy()

                        ar_offsets = np.array(self.ar_offsets)
                        if ar_offsets.ndim == 2:
                            ar_offsets = ar_offsets[None].repeat(self.channel_dim, axis=0)

                        self.encoder.init_ar_params(ar_indexes, ar_offsets)
                        self.decoder.init_ar_params(ar_indexes, ar_offsets)
                elif self.ar_method == "linear":
                    ar_ops = self._get_linear_ops()
                    self.encoder.init_custom_ar_ops(ar_ops)
                    self.decoder.init_custom_ar_ops(ar_ops)


class GaussianDistributionEntropyCoder(AutoregressiveImplDistributionEntropyCoder):
    """Efficient Lookup-table based* Guassian/Logistic distribution entropy coding implementation.
        Args:
            logvar_min (float, optional): [description]. Defaults to -7.0.
            logvar_max (float, optional): [description]. Defaults to 0.0.
            logvar_step (float, optional): [description]. Defaults to 0.5.
            mean_min ([type], optional): [description]. Defaults to None.
            mean_max ([type], optional): [description]. Defaults to None.
            mean_step ([type], optional): [description]. Defaults to None.
            sigmoid_mean (bool, optional): Sigmoid mean value to fix it [0, 1]*. Defaults to False.
            use_logistic_dist (bool, optional): Use Logistic distribution instead of Gaussian*. Defaults to False.
            mean_as_offset (bool, optional): Use mean value as data offset*. Reduces lookup table size but may harm compression. Defaults to False.

        * AI-ANS in https://openaccess.thecvf.com/content/CVPR2022/papers/Kang_PILC_Practical_Image_Lossless_Compression_With_an_End-to-End_GPU_Oriented_CVPR_2022_paper.pdf
    """    
    def __init__(self, *args, 
        # lookup table configs
        logvar_min=-7.0, logvar_max=0.0, logvar_step=0.5,
        mean_min=None, mean_max=None, mean_step=None,
        # PILC tricks
        sigmoid_mean=False,
        use_logistic_dist=False,
        mean_as_offset=False,
        **kwargs):
        self.logvar_min = logvar_min
        self.logvar_max = logvar_max
        self.logvar_step = logvar_step
        self.logvar_levels = len(np.arange(self.logvar_min, self.logvar_max+self.logvar_step, self.logvar_step))
        self.mean_min = mean_min
        self.mean_max = mean_max
        self.mean_step = mean_step

        self.sigmoid_mean = sigmoid_mean
        self.use_logistic_dist = use_logistic_dist

        self.mean_as_offset = mean_as_offset

        # self.mean_table = mean_table
        # self.scale_table = scale_table
        # if self.mean_table is None:
        #     self.mean_table = [i for i in range(self.data_range[0], self.data_range[1], self.data_step)]
        # if self.scale_table is None:
        #     self.scale_table = [-math.log(i) for i in range(1, 5)]

        super().__init__(*args, **kwargs)

    def _param_to_dist(self, params) -> distributions.Distribution:
        prior_mean, prior_logvar = params.chunk(2, dim=-1)
        assert prior_mean.shape[-1] == 1 and prior_logvar.shape[-1] == 1
        if self.sigmoid_mean:
            prior_mean = (torch.sigmoid(prior_mean) + self.data_range[0]) * (self.data_range[1] - self.data_range[0])
        if self.use_logistic_dist:
            prior_dist = Logistic(prior_mean, torch.exp(prior_logvar))
        else:
            prior_dist = distributions.Normal(prior_mean, torch.exp(prior_logvar))
        return prior_dist

    def _init_dist_params(self) -> torch.Tensor:
        self.mean_min = self.data_range[0] if self.mean_min is None else self.mean_min
        self.mean_max = self.data_range[1] if self.mean_max is None else self.mean_max
        self.mean_step = self.data_step if self.mean_step is None else self.mean_step
        self.mean_levels = len(np.arange(self.mean_min, self.mean_max+self.mean_step, self.mean_step))

        params = []
        if self.mean_as_offset:
            for logvar in np.arange(self.logvar_min, self.logvar_max+self.logvar_step, self.logvar_step):
                params.append((0 if self.sigmoid_mean else self.data_mid, logvar))
        else:
            for mean in np.arange(self.mean_min, self.mean_max+self.mean_step, self.mean_step):
                if self.sigmoid_mean:
                    # inverse sigmoid
                    mean = mean / (self.data_range[1] - self.data_range[0]) - self.data_range[0]
                    mean = mean.clip(1e-7, 1-1e-7)
                    mean = np.log(mean) - np.log1p(-mean)
                for logvar in np.arange(self.logvar_min, self.logvar_max+self.logvar_step, self.logvar_step):
                    params.append((mean, logvar))
        return torch.as_tensor(params)

    def _truncated_dist_to_logits(self, dist : distributions.Distribution) -> torch.Tensor:
        if self.distortion_method == "logprob":
            pts = torch.arange(self.data_range[0], self.data_range[1]+self.data_step, step=self.data_step).type_as(dist.mean)
            prior = torch.zeros(*dist.mean.shape[:-1], self.data_precision, device=dist.mean.device)
            prior[..., :] = pts
            prior_logprob = dist.log_prob(prior)
            return torch.log_softmax(prior_logprob, dim=-1)
        elif self.distortion_method == "cdf_delta":
            pts = [self.data_range[0]-self.data_step/2] + np.arange(self.data_range[0]+self.data_step/2, self.data_range[1]+self.data_step/2, self.data_step).tolist() + [self.data_range[1]+self.data_step/2]
            pts = torch.tensor(pts).type_as(dist.mean)
            # pts_step = (self.data_range[1] - self.data_range[0]) / self.data_precision
            # pts = torch.arange(self.data_range[0], self.data_range[1]+pts_step, step=pts_step).type_as(dist.mean)
            prior = torch.zeros(*dist.mean.shape[:-1], self.data_precision+1, device=dist.mean.device)
            prior[..., :] = pts
            prior_logprob = (dist.cdf(prior[..., 1:]) - dist.cdf(prior[..., :-1])).log()
            return torch.log_softmax(prior_logprob, dim=-1)# .clamp_min(np.log(1./self.freq_precision_total))
            # NOTE: try modspace prior for PILC
            # pts_step = (self.data_range[1] - self.data_range[0]) / self.data_precision
            # pts = torch.arange(self.data_range[0]-self.data_mid, self.data_range[1]+self.data_mid+pts_step, step=pts_step).type_as(dist.mean)
            # prior_pts = torch.zeros(*dist.mean.shape[:-1], self.data_precision*2+1, device=dist.mean.device)
            # prior_pts[..., :] = pts
            # prior_prob = dist.cdf(prior_pts[..., 1:]) - dist.cdf(prior_pts[..., :-1])
            # prior_prob_modspace = prior_prob[..., (self.data_precision // 2):(self.data_precision // 2 * 3)] + \
            #     torch.cat([prior_prob[..., :(self.data_precision // 2)], prior_prob[..., (self.data_precision // 2 * 3):]], dim=-1)
            # return prior_prob_modspace.log() / prior_prob_modspace.sum(dim=-1, keepdim=True)
        else:
            raise NotImplementedError()

    def _merge_ar_prior(self, ar_prior: torch.Tensor, prior: torch.Tensor = None, **kwargs):
        if self.use_autoregressive_prior:
            prior_mean, prior_logvar = prior.chunk(2, dim=-1)
            if self.ar_method == "twar":
                # NOTE: twar only allows self.ar_output_as_mean_offset==True
                if self.sigmoid_mean:
                    prior_mean = (torch.sigmoid(prior_mean) + self.data_range[0]) * (self.data_range[1] - self.data_range[0])
                    sigmoid_min = min(1e-6, 0.49 / self.data_precision)
                    prior_mean = (prior_mean + ar_prior[..., 0:1]).clamp(min=sigmoid_min, max=(1-sigmoid_min))
                    prior_mean = torch.log(prior_mean) - torch.log1p(-prior_mean)
                else:
                    prior_mean = prior_mean + ar_logits[..., 0:1]
                prior = torch.cat([prior_mean, prior_logvar], dim=-1)
            else:
                if self.ar_output_as_mean_offset:
                    if self.sigmoid_mean:
                        prior_mean = (torch.sigmoid(prior_mean) + self.data_range[0]) * (self.data_range[1] - self.data_range[0])
                        sigmoid_min = min(1e-6, 0.49 / self.data_precision)
                        prior_mean = (prior_mean + ar_prior[..., 0:1]).clamp(min=sigmoid_min, max=(1-sigmoid_min))
                        prior_mean = torch.log(prior_mean) - torch.log1p(-prior_mean)
                    else:
                        prior_mean = prior_mean + ar_logits[..., 0:1]
                    prior = torch.cat([prior_mean, prior_logvar], dim=-1)
                else:
                    if self.ar_method == "finitestate" and not self.ar_input_prior:
                        if self.training and self.ar_train_dropout > 0.0:
                            dropout_mask = torch.rand_like(ar_logits).mean(dim=-1, keepdim=True) > self.ar_train_dropout
                            ar_logits = ar_logits * dropout_mask
                        prior = prior + ar_logits
                    else:
                        prior = ar_logits
        return prior

    def _get_linear_ops(self):
        if not self.ar_output_as_mean_offset:
            print("Warning! ar_output_as_mean_offset is required for coding!")
            # pass
        if self.mean_as_offset:
            print("Warning! mean_as_offset is not supported for ar linear op")
        weight, bias = self.ar_model.weight, self.ar_model.bias
        ar_ops = []
        for i in range(self.channel_dim):
            ar_op = ar_limited_scaled_add_linear_op(
                (weight[i].reshape(-1) * (self.mean_levels-1) / (self.data_precision-1)).tolist(), 
                bias[i].item() * (self.mean_levels-1), 
                self.logvar_levels,
                0, self.mean_levels-1
            )
            ar_ops.append(ar_op)
        return ar_ops

    def _quantize_prior(self, prior) -> torch.Tensor:
        quantized_prior = super()._quantize_prior(prior)
        if self.mean_as_offset:
            quantized_prior[..., 0] = prior[..., 0]
        return quantized_prior

    def _select_best_indexes(self, prior) -> torch.LongTensor:
        prior_mean, prior_logvar = prior.chunk(2, dim=-1)
        assert prior_mean.shape[-1] == 1 and prior_logvar.shape[-1] == 1
        prior_mean = prior_mean.squeeze(-1)
        prior_logvar = prior_logvar.squeeze(-1)
        # logvar_num = (self.logvar_max - self.logvar_min) / self.logvar_step
        if self.mean_as_offset:
            scale_idx = ((prior_logvar - self.logvar_min) / self.logvar_step).round()\
                .clamp(min=0, max=self.logvar_levels-1).long()
            return scale_idx

        if self.sigmoid_mean:
            prior_mean = (torch.sigmoid(prior_mean) + self.data_range[0]) * (self.data_range[1] - self.data_range[0])
        if self.dist_trainable:
            # vq like selection
            dist_mean, dist_logvar = self.dist_params.chunk(2, dim=-1)
            if self.sigmoid_mean:
                dist_mean = (torch.sigmoid(dist_mean) + self.data_range[0]) * (self.data_range[1] - self.data_range[0])
            best_idx = ((prior_mean - dist_mean).abs() + (prior_logvar - dist_logvar).abs()).argmin(-1)
            return best_idx
        else:
            # TODO: optimize this
            # fast split selection
            mean_idx = ((prior_mean - self.mean_min) / self.mean_step).round()\
                .clamp(min=0, max=self.mean_levels-1).long()
            scale_idx = ((prior_logvar - self.logvar_min) / self.logvar_step).round()\
                .clamp(min=0, max=self.logvar_levels-1).long()
            return mean_idx * self.logvar_levels + scale_idx

    def _data_preprocess_with_prior(self, data, prior, **kwargs):
        if self.mean_as_offset:
            data = self._data_preprocess(data, **kwargs)
            prior_mean, prior_logvar = prior.chunk(2, dim=-1)
            if self.sigmoid_mean:
                prior_mean = (torch.sigmoid(prior_mean) + self.data_range[0]) * (self.data_range[1] - self.data_range[0])
            mean_offset = self._data_preprocess(prior_mean.squeeze(-1), **kwargs)
            data = (data - mean_offset + int(self.data_mid * self.data_precision)) % self.data_precision
        else:
            data = super()._data_preprocess_with_prior(data, prior, **kwargs)
        return data

    def _data_postprocess_with_prior(self, data, prior, **kwargs):
        if self.mean_as_offset:
            prior_mean, prior_logvar = prior.chunk(2, dim=-1)
            if self.sigmoid_mean:
                prior_mean = (torch.sigmoid(prior_mean) + self.data_range[0]) * (self.data_range[1] - self.data_range[0])
            mean_offset = self._data_preprocess(prior_mean.squeeze(-1), **kwargs)
            data = (data + mean_offset - int(self.data_mid * self.data_precision)) % self.data_precision
            data = self._data_postprocess(data, **kwargs)
        else:
            data = super()._data_postprocess_with_prior(data, prior, **kwargs)
        return data

