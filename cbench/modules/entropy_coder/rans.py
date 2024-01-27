# try:
#     from compressai.ans import RansEncoder, RansDecoder
#     from compressai._CXX import pmf_to_quantized_cdf
# except:
#     print("Warning! compressai is not propoerly installed!")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import EntropyCoderInterface, EntropyCoder, TorchQuantizedEntropyCoder
from cbench.rans import BufferedRansEncoder, RansEncoder, RansDecoder, pmf_to_quantized_cdf, pmf_to_quantized_cdf_np

def pmf_to_quantized_cdf_serial(pmf : torch.Tensor, add_tail=True, tail_mass=1e-10, freq_precision=16):
    max_index = 1 << freq_precision
    assert pmf.shape[-1] * 2 < max_index # NOTE: is this needed?
    if add_tail:
        pmf = torch.cat([pmf, torch.zeros(pmf.shape[0], 1).type_as(pmf) + tail_mass], dim=1)
    pmf = pmf.detach().cpu()
    return torch.as_tensor(pmf_to_quantized_cdf_np(pmf.contiguous().numpy(), freq_precision).astype(np.int32))


def pmf_to_quantized_cdf_batched(pmf : torch.Tensor, add_tail=True, tail_mass=1e-10, freq_precision=16):
    max_index = 1 << freq_precision
    assert pmf.shape[-1] * 2 < max_index # NOTE: is this needed?
    if add_tail:
        pmf = torch.cat([pmf.clone(), torch.zeros(pmf.shape[0], 1).type_as(pmf) + tail_mass], dim=1)
    # pmf[:, pmf_length] = tail_mass
    # for i, p in enumerate(pmf):
    #     p[(pmf_length[i]+1):] = 0
    pmf = pmf / pmf.sum(1, keepdim=True)
    # make sure all element in pmf is at least 1
    pmf_norm = pmf * max_index + 1.0
    # for i, p in enumerate(pmf_norm):
    #     p[(pmf_length[i]+1):] = 0
    pmf_sum_step = pmf.shape[-1] // 2 # NOTE: just a good value in practice. Why is that?
    pmf_norm_int = (pmf_norm * max_index / (pmf_norm.sum(1, keepdim=True) + pmf_sum_step)).round()
    # reduce some pdf to limit max cdf
    cdf_max = pmf_norm_int.sum(dim=1, keepdim=True)
    pmf_sum = cdf_max.clone()
    while (cdf_max > max_index).any():
        # further reducing (needed?)
        # idx = cdf_max > max_index
        pmf_sum[cdf_max > max_index] += pmf_sum_step
        pmf_norm_int = (pmf_norm_int * max_index / pmf_sum).round()
        cdf_max = pmf_norm_int.sum(dim=1, keepdim=True)
        # using steal technique (not deterministic for all elements)
        # max_pdf_steals = cdf_max.max().int().item() - max_index
        # steal_pdfs, steal_indices = torch.topk(pmf_norm_int, max_pdf_steals, dim=1, sorted=False)
        # # steal_indices = steal_indices[steal_pdfs > 1]
        # steal_values = ((steal_pdfs - 1) / (steal_pdfs-1).sum(1, keepdim=True) * max_pdf_steals).ceil().int()
        # pmf_norm_int[torch.arange(len(pmf_norm_int)).type_as(steal_indices).unsqueeze(-1), steal_indices] -= steal_values
    
    # Alt: use random sampling to generate integer to keep sum below max_index
    # num_samples = pmf_norm_int.sum(1) - max_index - 1
    # sample_distribution = pmf.cumsum(dim=1)
    # sample_prob = torch.rand(num_samples)

    # convert pmf to cdf
    cdf_float = pmf_norm_int.cumsum(dim=1)
    # cdf_float = pmf_norm.cumsum(dim=1)
    # cdf_float = cdf_float / cdf_float[:, -2:-1] * max_index # renormalize cdf
    cdf_float = torch.cat([torch.zeros(pmf.shape[0], 1).type_as(cdf_float), cdf_float], dim=1)
    cdf = cdf_float.int()
    # cdf[:, pmf_length] = max_index
    return cdf


class RansEntropyCoder(TorchQuantizedEntropyCoder):
    # FREQ_PRECISION = 16

    def __init__(self, *args, 
        fast_mode=True, 
        freq_precision=16, 
        **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = RansEncoder()
        self.decoder = RansDecoder()

        self.fast_mode = fast_mode
        self.freq_precision = freq_precision
    
    # TODO: add tail mass
    # def _pmf_to_cdf(self, pmf, tail_mass, pmf_length, max_length):
    #     cdf = torch.zeros(
    #         (len(pmf_length), max_length + 2), dtype=torch.int32, device=pmf.device
    #     )
    #     for i, p in enumerate(pmf):
    #         prob = torch.cat((p[: pmf_length[i]], tail_mass[i]), dim=0)
    #         _cdf = pmf_to_quantized_cdf(prob, self.entropy_coder_precision)
    #         cdf[i, : _cdf.size(0)] = _cdf
    #     return cdf

    def _pmf_to_cdf(self, pmf, tail_mass=1e-8):
        # A fast implementation using batch cumsum. May be a bit less accurate than super()._pmf_to_cdf
        if self.fast_mode:
            return pmf_to_quantized_cdf_batched(pmf, tail_mass=tail_mass, freq_precision=self.freq_precision)
        else:
            # return super()._pmf_to_cdf(pmf, tail_mass, pmf_length, max_length)
            # return torch.as_tensor([pmf_to_quantized_cdf(p + [tail_mass, ], self.freq_precision) for p in pmf.tolist()])
            pmf = pmf.detach().cpu()
            pmf = torch.cat([pmf, torch.zeros(pmf.shape[0], 1).type_as(pmf) + tail_mass], dim=1)
            return torch.as_tensor(pmf_to_quantized_cdf_np(pmf.contiguous().numpy(), self.freq_precision).astype(np.int32))
            return pmf_to_quantized_cdf_serial(pmf, tail_mass=tail_mass, freq_precision=self.freq_precision)

    def encode(self, data, *args, prior=None, **kwargs):
        # trim or pad prior spatial shape to be same as input
        # if data.shape[2:] != prior.shape[2:-1]:
        #     assert(len(data.shape[2:]) == len(prior.shape[2:-1]))
        #     padding = []
        #     cur_dim = 2
        #     for data_dim, prior_dim in zip(data.shape[2:], prior.shape[2:-1]):
        #         if data_dim < prior_dim:
        #             narrow_start = (prior_dim - data_dim) // 2
        #             prior = prior.narrow(cur_dim, narrow_start, data_dim)
        #             padding.extend([0, 0])
        #         else:
        #             padding_front = (data_dim - prior_dim) // 2
        #             padding_back = data_dim - prior_dim - padding_front
        #             padding.extend([padding_front, padding_back])
        #         cur_dim += 1
        #     # preserve prob dim
        #     prob_dim = prior.shape[-1]
        #     prior = prior.movedim(-1, 2)
        #     prior = prior.reshape(prior.shape[0], prior.shape[1] * prob_dim, *prior.shape[3:])
        #     prior = F.pad(prior, padding, mode="reflect")
        #     prior = prior.reshape(prior.shape[0], prior.shape[1] // prob_dim, prob_dim, *prior.shape[2:]).movedim(2, -1)
        data = self._data_preprocess(data)
        data = data.reshape(-1)
        with self.profiler.start_time_profile("time_prior_preprocess_encode"):
            if prior is not None:
                prior = self._prior_preprocess(prior)
                prior = prior.reshape(-1, prior.shape[-1])
            if len(prior) == 1:
                indexes = np.zeros_like(data, dtype=np.int32)
                cdfs = np.array([pmf_to_quantized_cdf(prior[0], self.freq_precision)])
            elif len(data) == len(prior):
                indexes = np.arange(len(data), dtype=np.int32)
                cdfs = self._pmf_to_cdf(prior).detach().cpu().numpy()
            else:
                raise ValueError("prior should be length 1 or per-data probability list!")
        cdf_lengths = np.array([len(cdf) for cdf in cdfs])
        offsets = np.zeros(len(indexes), dtype=np.int32) # [0] * len(indexes)

        with self.profiler.start_time_profile("time_rans_encoder"):
            byte_string = self.encoder.encode_with_indexes_np(
                data, indexes,
                cdfs, cdf_lengths, offsets
            )
        return byte_string

    def decode(self, byte_string: bytes, *args, prior=None, data_length=None, **kwargs):
        assert(prior is not None) # TODO: default prior?
        data_shape = prior.shape[:-1]
        with self.profiler.start_time_profile("time_prior_preprocess_decode"):
            prior = self._prior_preprocess(prior)
            prior = prior.reshape(-1, prior.shape[-1])
            if len(prior) == 1:
                assert(data_length is not None)
                indexes = np.zeros_like(data_length, dtype=np.int32)
                cdfs = np.array([pmf_to_quantized_cdf(prior[0], self.freq_precision)])
            else:
                indexes = np.arange(len(prior), dtype=np.int32)
                cdfs = self._pmf_to_cdf(prior).detach().cpu().numpy()
        cdf_lengths = np.array([len(cdf) for cdf in cdfs])
        offsets = np.zeros(len(indexes), dtype=np.int32) # [0] * len(indexes)

        with self.profiler.start_time_profile("time_rans_decoder"):
            symbols = self.decoder.decode_with_indexes_np(
                byte_string, indexes,
                cdfs, cdf_lengths, offsets
            )

        data = self._data_postprocess(np.array(symbols)).reshape(*data_shape)
        return data

