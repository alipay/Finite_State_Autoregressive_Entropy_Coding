import torch
import numpy as np
import time
import unittest

class TestANS(unittest.TestCase):

    def _generate_ans_params(self, num_dists, num_symbols):
        freqs = np.random.randint(1, 1024, (num_dists, num_symbols))
        nfreqs = np.zeros(num_dists) + num_symbols
        offsets = np.zeros(num_dists)
        return freqs, nfreqs, offsets

    def test_import(self):
        from cbench.ans import Rans64Encoder, Rans64Decoder

    def test_rans64_coding(self):
        from cbench.ans import Rans64Encoder, Rans64Decoder
        from cbench.utils.ar_utils import create_ar_offsets

        num_dists, num_symbols = 8, 512
        bypass_num = 32
        
        freqs, nfreqs, offsets = self._generate_ans_params(num_dists, num_symbols)

        encoder = Rans64Encoder(bypass_coding=(bypass_num > 0))
        decoder = Rans64Decoder(bypass_coding=(bypass_num > 0))

        start_time = time.time()
        encoder.init_params(freqs, nfreqs, offsets)
        decoder.init_params(freqs, nfreqs, offsets)
        print(f"Rans init time: {time.time() - start_time}")

        data_shape = (1000, 3, 32, 32)
        data = np.random.randint(0, num_symbols+bypass_num, data_shape) # test bypass coding
        indexes = np.random.randint(0, num_dists, data_shape)

        start_time = time.time()
        byte_string = encoder.encode_with_indexes(data, indexes)
        data_hat = decoder.decode_with_indexes(byte_string, indexes)
        print(f"Rans coding time: {time.time() - start_time}")

        self.assertEqual(data.reshape(-1).tolist(), data_hat.reshape(-1).tolist())

    def test_rans64_ar_coding(self):
        from cbench.ans import Rans64Encoder, Rans64Decoder
        from cbench.utils.ar_utils import create_ar_offsets

        num_dists, num_symbols = 8, 512
        
        freqs, nfreqs, offsets = self._generate_ans_params(num_dists, num_symbols)

        ar_dim_offsets = [[0,-1,0], [0,0,-1]]
        ar_table = np.random.randint(0, num_dists, [1, num_dists] + [num_symbols+1] * len(ar_dim_offsets))

        encoder = Rans64Encoder(bypass_coding=False)
        decoder = Rans64Decoder(bypass_coding=False)

        data_shape = (1000, 3, 32, 32)
        data = np.random.randint(0, num_symbols, data_shape) # test bypass coding
        indexes = np.random.randint(0, num_dists, data_shape)

        start_time = time.time()
        encoder.init_params(freqs, nfreqs, offsets)
        decoder.init_params(freqs, nfreqs, offsets)
        encoder.init_ar_params(ar_table, [ar_dim_offsets])
        decoder.init_ar_params(ar_table, [ar_dim_offsets])
        print(f"Rans AR init time: {time.time() - start_time}")

        start_time = time.time()
        ar_offsets = create_ar_offsets(indexes.shape, ar_dim_offsets)
        ar_indexes = np.zeros_like(indexes)
        byte_string = encoder.encode_with_indexes(data, indexes, ar_indexes=ar_indexes, ar_offsets=ar_offsets)
        data_hat = decoder.decode_with_indexes(byte_string, indexes, ar_indexes=ar_indexes, ar_offsets=ar_offsets)
        print(f"Rans AR coding time: {time.time() - start_time}")

        self.assertEqual(data.reshape(-1).tolist(), data_hat.reshape(-1).tolist())


    def test_rans64_cdf_coding(self):
        from cbench.ans import Rans64Encoder, Rans64Decoder
        from cbench.ans import pmf_to_quantized_cdf

        num_dists, num_symbols = 8, 512
        bypass_num = 32
        
        freqs, nfreqs, offsets = self._generate_ans_params(num_dists, num_symbols)

        encoder = Rans64Encoder(bypass_coding=(bypass_num > 0))
        decoder = Rans64Decoder(bypass_coding=(bypass_num > 0))

        data_shape = (1000, 3, 32, 32)
        data = np.random.randint(0, num_symbols+bypass_num, data_shape) # test bypass coding
        indexes = np.random.randint(0, num_dists, data_shape)

        # freq to cdf
        freqs = freqs.astype(np.float32) / freqs.sum()
        cdfs = [pmf_to_quantized_cdf(f.tolist() + [1e-8], 16) for f in freqs]
        cdf_sizes = np.array([len(cdf) for cdf in cdfs])
        cdfs_np = np.zeros((num_dists, cdf_sizes.max()))
        for idx, cdf in enumerate(cdfs):
            cdfs_np[:cdf_sizes[idx]] = np.array(cdf)

        encoder.init_cdf_params(cdfs_np, cdf_sizes, offsets)
        decoder.init_cdf_params(cdfs_np, cdf_sizes, offsets)

        byte_string = encoder.encode_with_indexes(data, indexes)
        data_hat = decoder.decode_with_indexes(byte_string, indexes)

        self.assertEqual(data.reshape(-1).tolist(), data_hat.reshape(-1).tolist())
    
    def test_tans_coding(self):
        from cbench.ans import TansEncoder, TansDecoder

        num_dists, num_symbols = 8, 512
        bypass_num = 32

        freqs, nfreqs, offsets = self._generate_ans_params(num_dists, num_symbols)

        encoder = TansEncoder(max_symbol_value=num_symbols-1, bypass_coding=(bypass_num > 0))
        decoder = TansDecoder(max_symbol_value=num_symbols-1, bypass_coding=(bypass_num > 0))

        start_time = time.time()
        encoder.init_params(freqs, nfreqs, offsets)
        decoder.init_params(freqs, nfreqs, offsets)
        print(f"Tans init time: {time.time() - start_time}")

        data_shape = (1000, 3, 32, 32)
        data = np.random.randint(0, num_symbols+bypass_num, data_shape)
        indexes = np.random.randint(0, num_dists, data_shape)

        start_time = time.time()
        byte_string = encoder.encode_with_indexes(data, indexes)
        data_hat = decoder.decode_with_indexes(byte_string, indexes)
        print(f"Tans coding time: {time.time() - start_time}")

        self.assertEqual(data.reshape(-1).tolist(), data_hat.reshape(-1).tolist())

    def test_tans_ar_coding(self):
        from cbench.ans import TansEncoder, TansDecoder
        from cbench.utils.ar_utils import create_ar_offsets

        num_dists, num_symbols = 8, 512
        
        freqs, nfreqs, offsets = self._generate_ans_params(num_dists, num_symbols)

        ar_dim_offsets = [[0,-1,0], [0,0,-1]]
        ar_table = np.random.randint(0, num_dists, [1, num_dists] + [num_symbols+1] * len(ar_dim_offsets))

        encoder = TansEncoder(max_symbol_value=num_symbols-1)
        decoder = TansDecoder(max_symbol_value=num_symbols-1)

        data_shape = (1000, 3, 32, 32)
        data = np.random.randint(0, num_symbols, data_shape) # test bypass coding
        indexes = np.random.randint(0, num_dists, data_shape)

        start_time = time.time()
        encoder.init_params(freqs, nfreqs, offsets)
        decoder.init_params(freqs, nfreqs, offsets)
        encoder.init_ar_params(ar_table, [ar_dim_offsets])
        decoder.init_ar_params(ar_table, [ar_dim_offsets])
        print(f"Tans AR init time: {time.time() - start_time}")

        start_time = time.time()
        ar_offsets = create_ar_offsets(indexes.shape, ar_dim_offsets)
        ar_indexes = np.zeros_like(indexes)
        byte_string = encoder.encode_with_indexes(data, indexes, ar_indexes=ar_indexes, ar_offsets=ar_offsets)
        data_hat = decoder.decode_with_indexes(byte_string, indexes, ar_indexes=ar_indexes, ar_offsets=ar_offsets)
        print(f"Tans AR coding time: {time.time() - start_time}")

        self.assertEqual(data.reshape(-1).tolist(), data_hat.reshape(-1).tolist())

if __name__ == "__main__":
    unittest.main()