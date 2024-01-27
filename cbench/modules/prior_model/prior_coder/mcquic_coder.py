# Ported from McQuic (https://github.com/xiaosu-zhu/McQuic)
import math
import numpy as np
from typing import Callable, Dict, List, Tuple, Union
from dataclasses import dataclass

import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F

# from mcquic.rans import pmfToQuantizedCDF
# from mcquic.rans import RansEncoder, RansDecoder
from cbench.rans import BufferedRansEncoder, RansEncoder, RansDecoder, pmf_to_quantized_cdf

from .base import PriorCoder
from cbench.nn.base import NNTrainableModule
from cbench.nn.layers import Upsample2DLayer, Downsample2DLayer, ResidualLayer
from cbench.utils.bytes_ops import encode_shape, decode_shape

class Consts:
    Name = "mcquic"
    # lazy load
    # TempDir = "/tmp/mcquic/"
    Eps = 1e-6
    CDot = "·"
    TimeOut = 15

@dataclass
class CodeSize:
    """Latent code specification.
           Code in this paper is of shape: `[[1, m, h, w], [1, m, h, w] ... ]`
                                                            `↑ total length = L`

    Args:
        heights (List[int]): Latent height for each stage.
        widths (List[int]): Latent width for each stage.
        k (List[int]): [k1, k2, ...], codewords amount for each stage.
        m (int): M, multi-codebook amount.
    """
    m: int
    heights: List[int]
    widths: List[int]
    k: List[int]

    def __str__(self) -> str:
        sequence = ", ".join(f"[{w}x{h}, {k}]" for h, w, k in zip(self.heights, self.widths, self.k))
        return f"""
        {self.m} code-groups: {sequence}"""


class _lowerBound(torch.autograd.Function):
    """Autograd function for the `LowerBound` operator."""

    @staticmethod
    def forward(ctx, input_, bound):
        ctx.save_for_backward(input_, bound)
        return torch.max(input_, bound)

    @staticmethod
    def backward(ctx, grad_output):
        input_, bound = ctx.saved_tensors
        pass_through_if = (input_ >= bound) | (grad_output < 0)
        return pass_through_if.type(grad_output.dtype) * grad_output, None

class LowerBound(nn.Module):
    """Lower bound operator, computes `torch.max(x, bound)` with a custom
    gradient.
    The derivative is replaced by the identity function when `x` is moved
    towards the `bound`, otherwise the gradient is kept to zero.
    """

    def __init__(self, bound: float):
        """Lower bound operator.

        Args:
            bound (float): The lower bound.
        """
        super().__init__()
        self.register_buffer("bound", torch.Tensor([float(bound)]))

    @torch.jit.unused
    def lower_bound(self, x):
        return _lowerBound.apply(x, self.bound)

    def forward(self, x):
        if torch.jit.is_scripting():
            return torch.max(x, self.bound)
        return self.lower_bound(x)

def gumbelSoftmax(logits: torch.Tensor, temperature: float = 1.0, hard: bool = True, dim: int = -1):
    eps = torch.finfo(logits.dtype).eps
    uniforms = torch.rand_like(logits).clamp_(eps, 1 - eps)
    gumbels = -((-(uniforms.log())).log())

    y_soft = ((logits + gumbels) / temperature).softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret

class EntropyCoder(nn.Module):
    def __init__(self, m: int, k: List[int], ema: float = 0.9):
        super().__init__()
        self.encoder = RansEncoder()
        self.decoder = RansDecoder()
        # initial value is uniform
        self._freqEMA = nn.ParameterList(nn.Parameter(torch.ones(m, ki) / ki, requires_grad=False) for ki in k)
        self._m = m
        self._k = k
        self._ema = ema
        self._cdfs = None
        self._normalizedFreq = None

    @torch.no_grad()
    def forward(self, oneHotCodes: List[torch.Tensor]):
        # Update freq by EMA
        # [n, m, h, w, k]
        for lv, code in enumerate(oneHotCodes):
            # [m, k]
            totalCount = code.sum((0, 2, 3))
            # sum over all gpus
            if dist.is_initialized():
                dist.all_reduce(totalCount)

            # normalize to probability.
            normalized = totalCount / totalCount.sum(-1, keepdim=True)

            # ema update
            ema = (1 - self._ema) * normalized + self._ema * self._freqEMA[lv]
            self._freqEMA[lv].copy_(ema)
        self.resetFreqAndCDF()

    def resetFreqAndCDF(self):
        self._normalizedFreq = None
        self._cdfs = None

    def updateFreqAndCDF(self):
        freq = list()
        for freqEMA in self._freqEMA:
            # normalized probs.
            freq.append((freqEMA / freqEMA.sum(-1, keepdim=True)).clone().detach())
        cdfs = list()
        for fr in freq:
            cdfAtLv = list()
            for frAtM in fr:
                cdf = pmf_to_quantized_cdf(frAtM.tolist(), 16)
                cdfAtLv.append(cdf)
            cdfs.append(cdfAtLv)
        self._normalizedFreq = freq
        self._cdfs = cdfs

    @property
    def CDFs(self) -> List[List[List[int]]]:
        if self._cdfs is None or self._normalizedFreq is None:
            self.updateFreqAndCDF()
        return self._cdfs

    @property
    def NormalizedFreq(self) -> List[torch.Tensor]:
        """Return list of `[m, k]` frequency tensors.
        """
        if self._cdfs is None or self._normalizedFreq is None:
            self.updateFreqAndCDF()
        return self._normalizedFreq

    def _checkShape(self, codes: List[torch.Tensor]):
        info = "Please give codes with correct shape, for example, [[1, 2, 24, 24], [1, 2, 12, 12], ...], which is a `level` length list. each code has shape [n, m, h, w]. "
        if len(codes) < 1:
            raise RuntimeError("Length of codes is 0.")
        n = codes[0].shape[0]
        m = codes[0].shape[1]
        for code in codes:
            newN, newM = code.shape[0], code.shape[1]
            if n < 1:
                raise RuntimeError(info + "Now `n` = 0.")
            if m != newM:
                raise RuntimeError(info + "Now `m` is inconsisitent.")
            if n != newN:
                raise RuntimeError(info + "Now `n` is inconsisitent.")
        return n, m

    # @torch.inference_mode()
    def compress(self, codes: List[torch.Tensor]) -> Tuple[List[List[bytes]], List[CodeSize]]:
        """Compress codes to binary.

        Args:
            codes (List[torch.Tensor]): List of tensor, len = level, code.shape = [n, m, h, w]
            cdfs (List[List[List[int]]]): cdfs for entropy coder, len = level, len(cdfs[0]) = m

        Returns:
            List[List[bytes]]: List of binary, len = n, len(binary[0]) = level
            List[CodeSize]]: List of code size, len = n
        """
        n, m = self._checkShape(codes)
        compressed = list(list() for _ in range(n))
        heights = list()
        widths = list()
        # [n, m, h, w]
        for code, ki, cdf in zip(codes, self._k, self.CDFs):
            _, _, h, w = code.shape
            heights.append(h)
            widths.append(w)
            for i, codePerImage in enumerate(code):
                indices = torch.arange(m)[:, None, None]
                # [m, h, w]
                idx = indices.expand_as(codePerImage).flatten().int().tolist()
                cdfSizes = [ki + 2] * m
                # [m, h, w]
                offsets = torch.zeros_like(codePerImage).flatten().int().tolist()
                binary: bytes = self.encoder.encode_with_indexes(codePerImage.flatten().int().tolist(), idx, cdf, cdfSizes, offsets)
                compressed[i].append(binary)
        return compressed, [CodeSize(m, heights, widths, self._k) for _ in range(n)]

    # @torch.inference_mode()
    def decompress(self, binaries: List[List[bytes]], codeSizes: List[CodeSize]) -> List[torch.Tensor]:
        """Restore codes from binary

        Args:
            binaries (List[List[bytes]]): len = n, len(binary[0]) = level
            codeSizes (List[CodeSize]): len = n
            cdfs (List[List[List[int]]]): len = level, len(cdfs[0]) = m

        Returns:
            List[List[torch.Tensor]]: len = level, each code.shape = [n, m, h, w]
        """
        lv = len(binaries[0])
        m = codeSizes[0].m
        codes = list(list() for _ in range(lv))
        indices = torch.arange(m)[:, None, None]
        for binary, codeSize in zip(binaries, codeSizes):
            for lv, (binaryAtLv, cdf, ki, h, w) in enumerate(zip(binary, self.CDFs, self._k, codeSize.heights, codeSize.widths)):
                idx = indices.expand(codeSize.m, h, w).flatten().int().tolist()
                cdfSizes = [ki + 2] * codeSize.m
                offsets = torch.zeros(codeSize.m, h, w, dtype=torch.int).flatten().int().tolist()
                restored: List[int] = self.decoder.decode_with_indexes(binaryAtLv, idx, cdf, cdfSizes, offsets)
                # [m, h, w]
                code = torch.tensor(restored).reshape(codeSize.m, h, w)
                codes[lv].append(code)
        return [torch.stack(c, 0).to(self._freqEMA[0].device) for c in codes]

    def compress_bytes(self, codes: List[torch.Tensor]) -> bytes:
        n, m = self._checkShape(codes)
        heights = list()
        widths = list()
        # [n, m, h, w]
        encoder = BufferedRansEncoder()
        for code, ki, cdf in zip(codes, self._k, self.CDFs):
            _, _, h, w = code.shape
            heights.append(h)
            widths.append(w)
            for i, codePerImage in enumerate(code):
                indices = torch.arange(m)[:, None, None]
                # [m, h, w]
                idx = indices.expand_as(codePerImage).flatten().int().detach().cpu().numpy()
                cdfSizes = np.array([ki + 2] * m)
                # [m, h, w]
                offsets = torch.zeros_like(codePerImage).flatten().int().detach().cpu().numpy()
                encoder.encode_with_indexes_np(codePerImage.flatten().int().detach().cpu().numpy(), idx, cdf, cdfSizes, offsets)
        data_bytes = encoder.flush()

        bytes_shape = encode_shape((n,) + tuple(heights) + tuple(widths))
        data_bytes = b''.join([bytes_shape, data_bytes])

        return data_bytes

    def decompress_bytes(self, byte_string: bytes) -> List[torch.Tensor]:
        full_shape, byte_ptr = decode_shape(byte_string)
        decoder = RansDecoder()
        decoder.set_stream(byte_string[byte_ptr:])

        lv = (len(full_shape)-1) // 2
        batch_size = full_shape[0]
        widths = full_shape[1:(lv+1)]
        heights = full_shape[(-lv):]
        m = self._m
        codes = list(list() for _ in range(lv))
        indices = torch.arange(m)[:, None, None]
        for lv, (cdf, ki, h, w) in enumerate(zip(self.CDFs, self._k, heights, widths)):
            idx = indices.expand(m, h, w).flatten().int().detach().cpu().numpy()
            cdf = np.array(cdf)
            cdfSizes = np.array([ki + 2] * m)
            offsets = torch.zeros(m, h, w, dtype=torch.int).flatten().int().detach().cpu().numpy()
            for batch_idx in range(batch_size):
                restored: List[int] = decoder.decode_stream_np(idx, cdf, cdfSizes, offsets)
                # [m, h, w]
                code = torch.as_tensor(restored, dtype=torch.long).reshape(m, h, w)
                codes[lv].append(code)
        return [torch.stack(c, 0).to(self._freqEMA[0].device) for c in codes]

class BaseQuantizer(nn.Module):
    def __init__(self, m: int, k: List[int]):
        super().__init__()
        self._entropyCoder = EntropyCoder(m, k)
        self._m = m
        self._k = k

    def encode(self, x: torch.Tensor) -> List[torch.Tensor]:
        raise NotImplementedError

    def decode(self, codes: List[torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

    @property
    def Codebooks(self) -> List[torch.Tensor]:
        raise NotImplementedError

    @property
    def CDFs(self):
        return self._entropyCoder.CDFs

    def reAssignCodebook(self) -> torch.Tensor:
        raise NotImplementedError

    def syncCodebook(self):
        raise NotImplementedError

    @property
    def NormalizedFreq(self):
        return self._entropyCoder.NormalizedFreq

    def compress(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[List[bytes]], List[CodeSize]]:
        codes = self.encode(x)

        # List of binary, len = n, len(binaries[0]) = level
        binaries, codeSize = self._entropyCoder.compress(codes)
        return codes, binaries, codeSize

    def _validateCode(self, refCodes: List[torch.Tensor], decompressed: List[torch.Tensor]):
        for code, restored in zip(refCodes, decompressed):
            if torch.any(code != restored):
                raise RuntimeError("Got wrong decompressed result from entropy coder.")

    def decompress(self, binaries: List[List[bytes]], codeSize: List[CodeSize]) -> torch.Tensor:
        decompressed = self._entropyCoder.decompress(binaries, codeSize)
        # self._validateCode(codes, decompressed)
        return self.decode(decompressed)

    def compress_bytes(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[List[bytes]], List[CodeSize]]:
        codes = self.encode(x)
        return self._entropyCoder.compress_bytes(codes)

    def decompress_bytes(self, byte_string: bytes) -> torch.Tensor:
        decompressed = self._entropyCoder.decompress_bytes(byte_string)
        # self._validateCode(codes, decompressed)
        return self.decode(decompressed)

# NOTE: You may notice the quantizer implemented here is different with README.md
#       After some tests, I find some strange behavior if `k` is not placed in the last dim.
#       Generally, although code is neat and output is same as here,
#         training with README's implementation will cause loss become suddenly NAN after a few epoches.
class _multiCodebookQuantization(nn.Module):
    def __init__(self, codebook: nn.Parameter, permutationRate: float = 0.0):
        super().__init__()
        self._m, self._k, self._d = codebook.shape
        self._codebook = codebook
        self._scale = math.sqrt(self._k)
        self._temperature = nn.Parameter(torch.ones((self._m, 1, 1, 1)))
        self._bound = LowerBound(Consts.Eps)
        self._permutationRate = permutationRate

    def reAssignCodebook(self, freq: torch.Tensor)-> torch.Tensor:
        codebook = self._codebook.clone().detach()
        freq = freq.to(self._codebook.device).clone().detach()
        #       [k, d],        [k]
        for m, (codebookGroup, freqGroup) in enumerate(zip(self._codebook, freq)):
            neverAssignedLoc = freqGroup < Consts.Eps
            totalNeverAssigned = int(neverAssignedLoc.sum())
            # More than half are never assigned
            if totalNeverAssigned > self._k // 2:
                mask = torch.zeros((totalNeverAssigned, ), device=self._codebook.device)
                maskIdx = torch.randperm(len(mask))[self._k // 2:]
                # Random pick some never assigned loc and drop them.
                mask[maskIdx] = 1.
                freqGroup[neverAssignedLoc] = mask
                # Update
                neverAssignedLoc = freqGroup < Consts.Eps
                totalNeverAssigned = int(neverAssignedLoc.sum())
            argIdx = torch.argsort(freqGroup, descending=True)[:(self._k - totalNeverAssigned)]
            mostAssigned = codebookGroup[argIdx]
            selectedIdx = torch.randperm(len(mostAssigned))[:totalNeverAssigned]
            codebook.data[m, neverAssignedLoc] = mostAssigned[selectedIdx]
        # [m, k] bool
        diff = ((codebook - self._codebook) ** 2).sum(-1) > 1e-6
        proportion = diff.flatten()
        self._codebook.data.copy_(codebook)
        return proportion

    def syncCodebook(self):
        # codebook = self._codebook.clone().detach()
        dist.broadcast(self._codebook, 0)

    def encode(self, x: torch.Tensor):
        # [n, m, h, w, k]
        distance = self._distance(x)
        # [n, m, h, w, k] -> [n, m, h, w]
        code = distance.argmin(-1)
        #      [n, m, h, w]
        return code

    # NOTE: ALREADY CHECKED CONSISTENCY WITH NAIVE IMPL.
    def _distance(self, x: torch.Tensor) -> torch.Tensor:
        n, _, h, w = x.shape
        # [n, m, d, h, w]
        x = x.reshape(n, self._m, self._d, h, w)

        # [n, m, 1, h, w]
        x2 = (x ** 2).sum(2, keepdim=True)

        # [m, k, 1, 1]
        c2 = (self._codebook ** 2).sum(-1, keepdim=True)[..., None]
        # [n, m, d, h, w] * [m, k, d] -sum-> [n, m, k, h, w]
        inter = torch.einsum("nmdhw,mkd->nmkhw", x, self._codebook)
        # [n, m, k, h, w]
        distance = x2 + c2 - 2 * inter
        # IMPORTANT to move k to last dim --- PLEASE SEE NOTE.
        # [n, m, h, w, k]
        return distance.permute(0, 1, 3, 4, 2)

    def _logit(self, x: torch.Tensor) -> torch.Tensor:
        logit = -1 * self._distance(x)
        return logit / self._scale

    def _permute(self, sample: torch.Tensor) -> torch.Tensor:
        if self._permutationRate < Consts.Eps:
            return sample
        # [n, h, w, m]
        needPerm = torch.rand_like(sample[..., 0]) < self._permutationRate
        randomed = F.one_hot(torch.randint(self._k, (needPerm.sum(), ), device=sample.device), num_classes=self._k).float()
        sample[needPerm] = randomed
        return sample

    def _sample(self, x: torch.Tensor, temperature: float):
        # [n, m, h, w, k] * [m, 1, 1, 1]
        logit = self._logit(x) * self._bound(self._temperature)

        # It causes training unstable
        # leave to future tests.
        # add random mask to pick a different index.
        # [n, m, h, w]
        # needPerm = torch.rand_like(logit[..., 0]) < self._permutationRate * rateScale
        # target will set to zero (one of k) but don't break gradient
        # mask = F.one_hot(torch.randint(self._k, (needPerm.sum(), ), device=logit.device), num_classes=self._k).float() * logit[needPerm]
        # logit[needPerm] -= mask.detach()

        # NOTE: STE: code usage is very low; RelaxedOneHotCat: Doesn't have STE trick
        # So reverse back to F.gumbel_softmax
        # posterior = OneHotCategoricalStraightThrough(logits=logit / temperature)
        # [n, m, k, h, w]
        # sampled = posterior.rsample(())

        sampled = gumbelSoftmax(logit, temperature, True)

        sampled = self._permute(sampled)

        # It causes training unstable
        # leave to future tests.
        # sampled = gumbelArgmaxRandomPerturb(logit, self._permutationRate * rateScale, temperature)
        return sampled, logit

    def forward(self, x: torch.Tensor):
        sample, logit = self._sample(x, 1.0)
        # [n, m, h, w, 1]
        code = logit.argmax(-1, keepdim=True)
        # [n, m, h, w, k]
        oneHot = torch.zeros_like(logit).scatter_(-1, code, 1)
        # [n, m, h, w, k]
        return sample, code[..., 0], oneHot, logit


class _multiCodebookDeQuantization(nn.Module):
    def __init__(self, codebook: nn.Parameter):
        super().__init__()
        self._m, self._k, self._d = codebook.shape
        self._codebook = codebook
        self.register_buffer("_ix", torch.arange(self._m), persistent=False)

    def decode(self, code: torch.Tensor):
        # codes: [n, m, h, w]
        n, _, h, w = code.shape
        # [n, h, w, m]
        code = code.permute(0, 2, 3, 1)
        # use codes to index codebook (m, k, d) ==> [n, h, w, m, k] -> [n, c, h, w]
        ix = self._ix.expand_as(code)
        # [n, h, w, m, d]
        indexed = self._codebook[ix, code]
        # [n, c, h, w]
        return indexed.reshape(n, h, w, -1).permute(0, 3, 1, 2)

    # NOTE: ALREADY CHECKED CONSISTENCY WITH NAIVE IMPL.
    def forward(self, sample: torch.Tensor):
        n, _, h, w, _ = sample.shape
        # [n, m, h, w, k, 1], [m, 1, 1, k, d] -sum-> [n, m, h, w, d] -> [n, m, d, h, w] -> [n, c, h, w]
        return torch.einsum("nmhwk,mkd->nmhwd", sample, self._codebook).permute(0, 1, 4, 2, 3).reshape(n, -1, h, w)


class _quantizerEncoder(nn.Module):
    """
    Default structure:
    ```plain
        x [H, W]
        | `latentStageEncoder`
        z [H/2 , W/2] -------╮
        | `quantizationHead` | `latentHead`
        q [H/2, W/2]         z [H/2, w/2]
        |                    |
        ├-`subtract` --------╯
        residual for next level
    ```
    """

    def __init__(self, quantizer: _multiCodebookQuantization, dequantizer: _multiCodebookDeQuantization, latentStageEncoder: nn.Module, quantizationHead: nn.Module, latentHead: Union[None, nn.Module]):
        super().__init__()
        self._quantizer = quantizer
        self._dequantizer = dequantizer
        self._latentStageEncoder = latentStageEncoder
        self._quantizationHead = quantizationHead
        self._latentHead = latentHead

    @property
    def Codebook(self):
        return self._quantizer._codebook

    def syncCodebook(self):
        self._quantizer.syncCodebook()

    def reAssignCodebook(self, freq: torch.Tensor) -> torch.Tensor:
        return self._quantizer.reAssignCodebook(freq)

    def encode(self, x: torch.Tensor):
        # [h, w] -> [h/2, w/2]
        z = self._latentStageEncoder(x)
        code = self._quantizer.encode(self._quantizationHead(z))
        if self._latentHead is None:
            return None, code
        z = self._latentHead(z)
        #      ↓ residual,                         [n, m, h, w]
        return z - self._dequantizer.decode(code), code

    def forward(self, x: torch.Tensor):
        # [h, w] -> [h/2, w/2]
        z = self._latentStageEncoder(x)
        q, code, oneHot, logit = self._quantizer(self._quantizationHead(z))
        if self._latentHead is None:
            return q, None, code, oneHot, logit
        z = self._latentHead(z)
        #         ↓ residual
        return q, z - self._dequantizer(q), code, oneHot, logit

class _quantizerDecoder(nn.Module):
    """
    Default structure:
    ```plain
        q [H/2, W/2]            formerLevelRestored [H/2, W/2]
        | `dequantizaitonHead`  | `sideHead`
        ├-`add` ----------------╯
        xHat [H/2, W/2]
        | `restoreHead`
        nextLevelRestored [H, W]
    ```
    """

    def __init__(self, dequantizer: _multiCodebookDeQuantization, dequantizationHead: nn.Module, sideHead: Union[None, nn.Module], restoreHead: nn.Module):
        super().__init__()
        self._dequantizer =  dequantizer
        self._dequantizationHead =  dequantizationHead
        self._sideHead =  sideHead
        self._restoreHead =  restoreHead

    #                [n, m, h, w]
    def decode(self, code: torch.Tensor, formerLevel: Union[None, torch.Tensor]):
        q = self._dequantizationHead(self._dequantizer.decode(code))
        if self._sideHead is not None:
            xHat = q + self._sideHead(formerLevel)
        else:
            xHat = q
        return self._restoreHead(xHat)

    def forward(self, q: torch.Tensor, formerLevel: Union[None, torch.Tensor]):
        q = self._dequantizationHead(self._dequantizer(q))
        if self._sideHead is not None:
            xHat = q + self._sideHead(formerLevel)
        else:
            xHat = q
        return self._restoreHead(xHat)


class UMGMQuantizer(BaseQuantizer):
    _components = [
        "latentStageEncoder",
        "quantizationHead",
        "latentHead",
        "dequantizationHead",
        "sideHead",
        "restoreHead"
    ]
    def __init__(self, channel: int, m: int, k: Union[int, List[int]], permutationRate: float, components: Dict[str, Callable[[], nn.Module]]):
        if isinstance(k, int):
            k = [k]
        super().__init__(m, k)
        componentFns = [components[key] for key in self._components]
        latentStageEncoderFn, quantizationHeadFn, latentHeadFn, dequantizationHeadFn, sideHeadFn, restoreHeadFn = componentFns

        encoders = list()
        decoders = list()

        for i, ki in enumerate(k):
            latentStageEncoder = latentStageEncoderFn()
            quantizationHead = quantizationHeadFn()
            latentHead = latentHeadFn() if i < len(k) - 1 else None
            dequantizationHead = dequantizationHeadFn()
            sideHead = sideHeadFn() if i < len(k) - 1 else None
            restoreHead = restoreHeadFn()
            # This magic is called SmallInit, from paper
            # "Transformers without Tears: Improving the Normalization of Self-Attention",
            # https://arxiv.org/pdf/1910.05895.pdf
            # I've tried a series of initilizations, but found this works the best.
            codebook = nn.Parameter(nn.init.normal_(torch.empty(m, ki, channel // m), std=math.sqrt(2 / (5 * channel / m))))
            quantizer = _multiCodebookQuantization(codebook, permutationRate)
            dequantizer = _multiCodebookDeQuantization(codebook)
            encoders.append(_quantizerEncoder(quantizer, dequantizer, latentStageEncoder, quantizationHead, latentHead))
            decoders.append(_quantizerDecoder(dequantizer, dequantizationHead, sideHead, restoreHead))

        self._encoders: nn.ModuleList[_quantizerEncoder] = nn.ModuleList(encoders)
        self._decoders: nn.ModuleList[_quantizerDecoder] = nn.ModuleList(decoders)

    @property
    def Codebooks(self):
        return list(encoder.Codebook for encoder in self._encoders)

    def encode(self, x: torch.Tensor) -> List[torch.Tensor]:
        codes = list()
        for encoder in self._encoders:
            x, code = encoder.encode(x)
            #            [n, m, h, w]
            codes.append(code)
        # lv * [n, m, h, w]
        return codes

    def decode(self, codes: List[torch.Tensor]) -> Union[torch.Tensor, None]:
        formerLevel = None
        for decoder, code in zip(self._decoders[::-1], codes[::-1]):
            formerLevel = decoder.decode(code, formerLevel)
        return formerLevel

    def reAssignCodebook(self) -> torch.Tensor:
        freqs = self.NormalizedFreq
        reassigned: List[torch.Tensor] = list()
        for encoder, freq in zip(self._encoders, freqs):
            # freq: [m, ki]
            reassigned.append(encoder.reAssignCodebook(freq))
        return torch.cat(reassigned).float().mean()

    def syncCodebook(self):
        dist.barrier()
        for encoder in self._encoders:
            encoder.syncCodebook()

    def forward(self, x: torch.Tensor):
        quantizeds = list()
        codes = list()
        oneHots = list()
        logits = list()
        for encoder in self._encoders:
            #          ↓ residual
            quantized, x, code, oneHot, logit = encoder(x)
            # [n, c, h, w]
            quantizeds.append(quantized)
            # [n, m, h, w]
            codes.append(code)
            # [n, m, h, w, k]
            oneHots.append(oneHot)
            # [n, m, h, w, k]
            logits.append(logit)
        formerLevel = None
        for decoder, quantized in zip(self._decoders[::-1], quantizeds[::-1]):
            # ↓ restored
            formerLevel = decoder(quantized, formerLevel)

        # update freq in entropy coder
        self._entropyCoder(oneHots)

        return formerLevel, codes, logits


class McQuicPriorCoder(PriorCoder, NNTrainableModule):
    def __init__(self, channel: int = 256, m: int = 2, k: List[int] = [8192, 2048, 512], permutationRate: float = 0.0):
        super().__init__()
        NNTrainableModule.__init__(self)

        quantizer = UMGMQuantizer(channel, m, k, permutationRate, {
            "latentStageEncoder": lambda: nn.Sequential(
                Downsample2DLayer(channel, channel),
                ResidualLayer(channel, channel),
            ),
            "quantizationHead": lambda: nn.Sequential(
                ResidualLayer(channel, channel, inplace=False),
                nn.Conv2d(channel, channel, kernel_size=3, padding=1),
            ),
            "latentHead": lambda: nn.Sequential(
                ResidualLayer(channel, channel, inplace=False),
                nn.Conv2d(channel, channel, kernel_size=3, padding=1),
            ),
            "restoreHead": lambda: nn.Sequential(
                ResidualLayer(channel, channel, inplace=False),
                Upsample2DLayer(channel, channel)
            ),
            "dequantizationHead": lambda: nn.Sequential(
                nn.Conv2d(channel, channel, kernel_size=3, padding=1),
                ResidualLayer(channel, channel),
            ),
            "sideHead": lambda: nn.Sequential(
                nn.Conv2d(channel, channel, kernel_size=3, padding=1),
                ResidualLayer(channel, channel),
            ),
        })

        self._quantizer = quantizer

    def forward(self, input : torch.Tensor, prior : torch.Tensor = None, **kwargs):
        yHat, codes, logits = self._quantizer(input)

        # calculate prior_entropy
        prior_logits = 0
        for i, code in enumerate(codes):
            prior_logits += math.log(self._quantizer._k[i]) * code.numel()
        self.update_cache("metric_dict",
            prior_entropy = prior_logits / input.shape[0], # normalize by batch size
        )

        return yHat

    def reAssignCodebook(self) -> torch.Tensor:
        return self._quantizer.reAssignCodebook()

    def syncCodebook(self):
        return self._quantizer.syncCodebook()

    @property
    def Codebooks(self):
        return self._quantizer.Codebooks

    @property
    def CDFs(self):
        return self._quantizer.CDFs

    @property
    def NormalizedFreq(self):
        return self._quantizer.NormalizedFreq

    @property
    def CodeUsage(self):
        return torch.cat(list((freq > Consts.Eps).flatten() for freq in self._quantizer.NormalizedFreq)).float().mean()

    def set_vamp_posterior(self, posterior):
        raise NotImplementedError()

    def encode(self, input : torch.Tensor, *args, **kwargs) -> bytes:
        return self._quantizer.compress_bytes(input)

    def decode(self, byte_string, *args, **kwargs) -> torch.Tensor:
        return self._quantizer.decompress_bytes(byte_string)

