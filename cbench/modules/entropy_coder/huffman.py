import numpy as np
from heapq import heappush, heappop

from .base import EntropyCoder


class PyHuffmanCoder(EntropyCoder):
    def __init__(self, *args, max_symbol=255, **kwargs):
        self.max_symbol = max_symbol

    def encode(self, data: np.ndarray, *args, prior: np.ndarray = None, **kwargs):
        # assume prior is frequency
        # first normalize prior
        if prior is None:
            prior = np.ones(self.max_symbol)
        # prior /= prior.sum(0, keepdim=True)

        codebook = self._build_codebook({i: p for i, p in enumerate(prior)})

        # perform encoding
        bit_string = b''.join([codebook[symbol] for symbol in data])
        # stash extra bits to form bytes (TODO: how to remove extra bits when decoding?)
        if len(bit_string) % 8 > 0:
            bit_string += b'0' * (8 - len(bit_string) % 8)
        bit_string = np.fromstring(bit_string, dtype=np.uint8) - ord('0')

        # to bytes
        byte_string = np.packbits(bit_string.reshape(-1, 8), axis=-1).tobytes()
        return byte_string

    def decode(self, byte_string: bytes, *args, prior=None, **kwargs):
        # assume prior is frequency
        # first normalize prior
        if prior is None:
            prior = np.ones(self.max_symbol)
        # prior /= prior.sum(0, keepdim=True)

        codebook = self._build_codebook({i: p for i, p in enumerate(prior)})
        codebook_inv = {v: k for k, v in codebook.items()}

        byte_string = np.frombuffer(byte_string, dtype=np.uint8)
        bit_string = (np.unpackbits(byte_string).reshape(-1) + ord('0')).tobytes()

        # decode bit_string
        symbols = []
        bits_head = 0
        bits_head_search = 0
        while bits_head_search < len(bit_string):
            bits_head_search += 1
            search_bits = bit_string[bits_head:bits_head_search]
            if search_bits in codebook_inv:
                symbols.append(codebook_inv[search_bits])
                bits_head = bits_head_search

        return symbols

    # https://github.com/pynflate/pynflate/blob/master/src/pynflate/huffman.py
    def _build_codebook(self, frequencies):
        if len(frequencies) == 1:
            letter, = frequencies
            return {letter: b'0'}

        queue = []
        res = {letter: b'' for letter in frequencies}

        for letter, frequency in frequencies.items():
            heappush(queue, (frequency, [letter]))

        while len(queue) > 1:
            first_freq, first_letters = heappop(queue)
            second_freq, second_letters = heappop(queue)

            for letter in first_letters:
                res[letter] = b'0' + res[letter]

            for letter in second_letters:
                res[letter] = b'1' + res[letter]

            heappush(
                queue,
                (
                    first_freq + second_freq,
                    first_letters + second_letters
                )
            )

        return res

if __name__ == "__main__":
    coder = PyHuffmanCoder()
    data = [1, 0, 0, 2, 3, 3, 0, 0]
    prior = [4, 1, 1, 2]
    byte_string = coder.encode(data, prior=prior)
    data_decode = coder.decode(byte_string, prior=prior)
    print(byte_string)
    print(data, data_decode)
