import math
import struct
import io
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.models.utils import update_registered_buffers
from compressai.layers import GDN, MaskedConv2d
from compressai.models.utils import conv, deconv

from .base import PriorCoder
from cbench.nn.base import NNTrainableModule
from cbench.nn.models.google import HyperpriorHyperSynthesisModel, HyperpriorHyperAnalysisModel

# From Balle's tensorflow compression examples
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64


def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    """Returns table of logarithmically scales."""
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))

def write_uints(fd, values, fmt=">{:d}I"):
    fd.write(struct.pack(fmt.format(len(values)), *values))
    return len(values) * 4


def write_uchars(fd, values, fmt=">{:d}B"):
    fd.write(struct.pack(fmt.format(len(values)), *values))
    return len(values) * 1


def read_uints(fd, n, fmt=">{:d}I"):
    sz = struct.calcsize("I")
    return struct.unpack(fmt.format(n), fd.read(n * sz))


def read_uchars(fd, n, fmt=">{:d}B"):
    sz = struct.calcsize("B")
    return struct.unpack(fmt.format(n), fd.read(n * sz))


def write_bytes(fd, values, fmt=">{:d}s"):
    if len(values) == 0:
        return
    fd.write(struct.pack(fmt.format(len(values)), values))
    return len(values) * 1


def read_bytes(fd, n, fmt=">{:d}s"):
    sz = struct.calcsize("s")
    return struct.unpack(fmt.format(n), fd.read(n * sz))[0]

def read_body(fd, segments=1):
    lstrings = []
    shape = read_uints(fd, 2)
    n_strings = read_uints(fd, 1)[0]
    for _ in range(n_strings):
        batch = []
        for seglen in read_uints(fd, segments):
            batch.append(read_bytes(fd, seglen))
        lstrings.append(batch)

    return lstrings, shape


def write_body(fd, shape, out_strings, segments=1):
    bytes_cnt = 0
    bytes_cnt = write_uints(fd, (shape[0], shape[1], len(out_strings)))
    for s in out_strings:
        assert len(s) == segments
        bytes_cnt += write_uints(fd, [len(seg) for seg in s])
        for segidx in range(segments):
            bytes_cnt += write_bytes(fd, s[segidx])
    return bytes_cnt


class CompressAIEntropyBottleneckPriorCoder(PriorCoder, NNTrainableModule):
    def __init__(self, entropy_bottleneck_channels=256, 
                 eps=1e-7, 
                 use_inner_aux_opt=False, 
                 use_bit_rate_loss=True,
                 **kwargs):
        super().__init__()
        NNTrainableModule.__init__(self)

        self.entropy_bottleneck = EntropyBottleneck(entropy_bottleneck_channels)
        self.eps = eps
        self.use_inner_aux_opt = use_inner_aux_opt
        self.use_bit_rate_loss = use_bit_rate_loss
        # self.update_state()

        # use aux optimizer for quantiles
        aux_params = []
        for name, param in self.entropy_bottleneck.named_parameters():
            if param.requires_grad and name.endswith("quantiles"):
                aux_params.append(param)
        if self.use_inner_aux_opt:
            self.aux_opt = optim.Adam(aux_params, lr=1e-3)
        else:
            # mark params for external aux optimizer
            for param in aux_params:
                param.aux_id = 0

    def aux_loss(self):
        """Return the aggregated loss over the auxiliary entropy bottleneck
        module(s).
        """
        aux_loss = sum(
            m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck)
        )
        return aux_loss

    def update(self, force=False):
        """Updates the entropy bottleneck(s) CDF values.

        Needs to be called once after training to be able to later perform the
        evaluation with an actual entropy coder.

        Args:
            force (bool): overwrite previous values (default: False)

        Returns:
            updated (bool): True if one of the EntropyBottlenecks was updated.

        """
        updated = False
        for m in self.children():
            if not isinstance(m, EntropyBottleneck):
                continue
            rv = m.update(force=force)
            updated |= rv
        return updated

    # def load_state_dict(self, state_dict):
    #     # Dynamically update the entropy bottleneck buffers related to the CDFs
    #     update_registered_buffers(
    #         self.entropy_bottleneck,
    #         "entropy_bottleneck",
    #         ["_quantized_cdf", "_offset", "_cdf_length"],
    #         state_dict,
    #     )
    #     super().load_state_dict(state_dict)

    def load_state_dict(self, state_dict, strict=True):
        for name, module in self.named_modules():
            if not any(x.startswith(name) for x in state_dict.keys()):
                continue

            if isinstance(module, EntropyBottleneck):
                update_registered_buffers(
                    module,
                    name,
                    ["_quantized_cdf", "_offset", "_cdf_length"],
                    state_dict,
                )

            if isinstance(module, GaussianConditional):
                update_registered_buffers(
                    module,
                    name,
                    ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
                    state_dict,
                )

        return nn.Module.load_state_dict(self, state_dict, strict=strict)

    
    def forward(self, input, *args, **kwargs):
        y_hat, y_likelihoods = self.entropy_bottleneck(input)

        entropy = -torch.log(y_likelihoods).sum() 
        if self.training:
            # NOTE: we follow most works using bits as rate loss
            loss_rate = (entropy / math.log(2)) if self.use_bit_rate_loss else entropy
            self.update_cache("loss_dict", 
                loss_rate = loss_rate / input.shape[0] # normalize by batch size
            )
            loss_aux = self.aux_loss()
            if self.use_inner_aux_opt:
                self.aux_opt.zero_grad()
                loss_aux.backward()
                self.aux_opt.step()
                self.update_cache("moniter_dict",
                    loss_aux = loss_aux,
                )
            else:
                self.update_cache("loss_dict", 
                    loss_aux = loss_aux,
                )
        self.update_cache("metric_dict",
            prior_entropy = entropy / input.shape[0], # normalize by batch size
        )

        return y_hat

    def encode(self, input, *args, **kwargs) -> bytes:
        y_strings = self.entropy_bottleneck.compress(input)
        with io.BytesIO() as bio:
            write_body(bio, input.size()[-2:], [[string] for string in y_strings])
            return bio.getvalue()

    def decode(self, byte_string, *args, **kwargs):
        with io.BytesIO(byte_string) as bio:
            strings, shape = read_body(bio)
            # assert isinstance(strings, list) and len(strings) == 1
            y_hat = self.entropy_bottleneck.decompress([string[0] for string in strings], shape)
            return y_hat

    def update_state(self, *args, **kwargs) -> None:
        return self.update(*args, **kwargs)


class CompressAIGaussianConditionalCoder(PriorCoder, NNTrainableModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        NNTrainableModule.__init__(self)
        self.gaussian_conditional = GaussianConditional(None)

    def forward(self, y, *args, prior=None, **kwargs):
        y_hat, y_likelihoods = self.gaussian_conditional(y, prior)

        entropy = -torch.log(y_likelihoods).sum()
        if self.training:
            # NOTE: we follow most works using bits as rate loss
            loss_rate = (entropy / math.log(2)) # if self.use_bit_rate_loss else entropy
            self.update_cache("loss_dict", 
                loss_rate = loss_rate / y.shape[0] # normalize by batch size
            )
        self.update_cache("metric_dict",
            prior_entropy = entropy / y.shape[0], # normalize by batch size
        )
        return y_hat

    def encode(self, y, *args, prior=None, **kwargs):
        indexes = self.gaussian_conditional.build_indexes(prior)
        y_strings = self.gaussian_conditional.compress(y, indexes)
        with io.BytesIO() as bio:
            write_body(bio, input.size()[-2:], [[string] for string in y_strings])
            return bio.getvalue()

    def decode(self, byte_string, *args, prior=None, **kwargs):
        with io.BytesIO(byte_string) as bio:
            strings, shape = read_body(bio)
            indexes = self.gaussian_conditional.build_indexes(prior)
            y_hat = self.gaussian_conditional.decompress([string[0] for string in strings], indexes)
            return y_hat

    def update_state(self, *args, **kwargs) -> None:
        self.gaussian_conditional.update_scale_table(get_scale_table())
        return super().update_state(*args, **kwargs)