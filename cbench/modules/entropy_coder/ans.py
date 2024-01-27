import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions

from typing import Tuple, Optional

from .base import TorchQuantizedEntropyCoder
from cbench.ans import Rans64Encoder, Rans64Decoder, TansEncoder, TansDecoder


class ANSEntropyCoder(TorchQuantizedEntropyCoder):
    """ANS entropy coder wrapper for cbench.ans (c++ binding)

        Args:
            coder_type (str, optional): rans64 / tans. Defaults to "rans64".
            use_bypass_coding (bool, optional): whether to enable out-of-range coding. Defaults to False.
            freq_precision (int, optional): bits allocated for ANS frequency. Defaults to 16.
    """
    def __init__(self, *args, 
        coder_type="rans64",
        use_bypass_coding=False,
        freq_precision=16,
        **kwargs):
        super().__init__(*args, **kwargs)
        
        self.coder_type = coder_type
        self.use_bypass_coding = use_bypass_coding
        self.freq_precision = freq_precision
        # self.update_state()

    def _select_best_indexes(self, prior) -> torch.LongTensor:
        """ quantize prior distributions to indices

        Args:
            prior (_type_): _description_

        Raises:
            NotImplementedError: _description_

        Returns:
            torch.LongTensor: _description_
        """        
        raise NotImplementedError()
    
    def _get_ar_params(self, prior) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Args:
            prior ([type]): [description]

        Returns:
            Optional[Tuple[np.ndarray, np.ndarray]]: ar_indexes, ar_offsets
        """        
        return None
    
    def _get_ans_params(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: freqs, nfreqs, offsets
        """        
        raise NotImplementedError()
    
    def _data_preprocess_with_prior(self, data, prior, **kwargs):
        data = self._data_preprocess(data, **kwargs) 
        data = data % self.data_precision # use mod to control range
        # NOTE: by default prior is not used
        return data

    def _data_postprocess_with_prior(self, data, prior, **kwargs):
        data = self._data_postprocess(data, **kwargs)
        # NOTE: by default prior is not used
        return data

    def encode(self, data, *args, prior=None, **kwargs):
        with self.profiler.start_time_profile("time_prior_preprocess_encode"):
            if prior is not None:
                indexes = self._select_best_indexes(prior)
                # quant_prior = self._init_dist_params()[indexes.reshape(-1)]
                indexes = indexes.detach().cpu().contiguous().numpy()
            else:
                raise ValueError("prior should not be None!")
            
            ar_indexes, ar_offsets = None, None
            ar_params = self._get_ar_params(prior)
            if ar_params is not None:
                ar_indexes, ar_offsets = ar_params
                # ar_indexes = ar_indexes.detach().cpu().numpy()
                # ar_offsets = ar_offsets.detach().cpu().numpy()

        with self.profiler.start_time_profile("time_data_preprocess_encode"):
            data = self._data_preprocess_with_prior(data.contiguous(), prior)
            # data = data.reshape(-1)

        with self.profiler.start_time_profile("time_ans_encode"):
            byte_string = self.encoder.encode_with_indexes(data, indexes, ar_indexes=ar_indexes, ar_offsets=ar_offsets)
            # peek = self.encoder.peek_cache()
            # byte_string = self.encoder.flush()

        return byte_string

    def decode(self, byte_string: bytes, *args, prior=None, data_length=None, **kwargs):
        assert(prior is not None) # TODO: default prior?
        data_shape = prior.shape[:-1]
        with self.profiler.start_time_profile("time_prior_preprocess_decode"):
            if prior is not None:
                indexes = self._select_best_indexes(prior)
                indexes = indexes.detach().cpu().contiguous().numpy()
            else:
                raise ValueError("prior should not be None!")

            ar_indexes, ar_offsets = None, None
            ar_params = self._get_ar_params(prior)
            if ar_params is not None:
                ar_indexes, ar_offsets = ar_params
                # ar_indexes = ar_indexes.detach().cpu().numpy()
                # ar_offsets = ar_offsets.detach().cpu().numpy()

        with self.profiler.start_time_profile("time_ans_decode"):
            symbols = self.decoder.decode_with_indexes(byte_string, indexes, ar_indexes=ar_indexes, ar_offsets=ar_offsets)

        with self.profiler.start_time_profile("time_data_postprocess_decode"):
            data = np.array(symbols).reshape(*data_shape)
            data = self._data_postprocess_with_prior(data, prior)
        return data

    def update_state(self, *args, **kwargs) -> None:
        if self.coder_type == "rans" or self.coder_type == "rans64":
            encoder = Rans64Encoder(freq_precision=self.freq_precision, bypass_coding=self.use_bypass_coding)
            decoder = Rans64Decoder(freq_precision=self.freq_precision, bypass_coding=self.use_bypass_coding)
        elif self.coder_type == "tans":
            encoder = TansEncoder(table_log=self.freq_precision, max_symbol_value=self.data_precision-1, bypass_coding=self.use_bypass_coding)
            decoder = TansDecoder(table_log=self.freq_precision, max_symbol_value=self.data_precision-1, bypass_coding=self.use_bypass_coding)
        else:
            raise NotImplementedError(f"Unknown coder type {self.coder_type}")

        freqs, nfreqs, offsets = self._get_ans_params()
        encoder.init_params(freqs, nfreqs, offsets)
        decoder.init_params(freqs, nfreqs, offsets)

        self.encoder = encoder
        self.decoder = decoder
