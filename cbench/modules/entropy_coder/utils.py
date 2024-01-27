import math
from typing import Any, Dict, List
import struct
import io
import numpy as np

from cbench.utils.bytes_ops import merge_bytes, split_merged_bytes
from .base import EntropyCoder, EntropyCoderInterface
from ..base import BaseTrainableModule, ModuleList, TrainableModuleInterface


# TODO: endian issue?
class BinaryHeadConstructor(object):
    def __init__(self, *args : List[int], **kwargs : Dict[str, int]) -> None:
        self.data = dict()
        self.max_values = dict()
        for i, arg in enumerate(args):
            self.max_values['{}'.format(i)] = arg
        for key, arg in kwargs.items():
            self.max_values[key] = arg

        max_state = 1
        max_bytes = 0
        for max_value in self.max_values.values():
            max_state = (max_state+1) * max_value
            while max_state > 255:
                max_bytes += 1
                max_state >>= 8
        if max_state > 0:
            max_bytes += 1
        self.max_bytes = max_bytes

    def add_data(self, key, value) -> None:
        if key in self.max_values:
            assert value < self.max_values[key]
            self.data[key] = value

    def set_data(self, *args : List[int], **kwargs : Dict[str, int]) -> None:
        for i, value in enumerate(args):
            key = '{}'.format(i)
            if key in self.max_values:
                assert value < self.max_values[key]
                self.data[key] = value
        for key, value in kwargs.items():
            if key in self.max_values:
                assert value < self.max_values[key]
                self.data[key] = value
            
    def get_max_bytes(self) -> int:
        return self.max_bytes
    
    def get_bytes(self) -> bytes:
        state = 0
        output_bytes = []
        # output bytes inversely
        for key in reversed(list(self.max_values.keys())):
            assert key in self.data
            state = int(state * int(self.max_values[key])) + self.data[key]
            # output state to bytes
            while state > 255:
                state_next = state >> 8
                out_value = state - (state_next << 8)
                output_bytes.append(out_value)
                state = state_next
        
        # final state
        output_bytes.append(state)
        while len(output_bytes) < self.max_bytes:
            output_bytes.append(0)
        # reversed output
        return bytes(reversed(output_bytes))

    def get_data_from_bytes(self, byte_string) -> dict:
        assert len(byte_string) == self.max_bytes

        # initialize state
        # state = byte_string[0]
        # if len(byte_string) > 1:
        #     for byte in byte_string[1:]:
        #         state = (state << 8) + int(byte)

        # get data
        state = 0
        bytes_idx = 0
        for key, max_value in self.max_values.items():
            while state < max_value and bytes_idx < len(byte_string):
                state = (state << 8) + int(byte_string[bytes_idx])
                bytes_idx += 1
            state_next = state // max_value
            self.data[key] = state - int(state_next * int(max_value))
            state = state_next

        assert state == 0
        return self.data



class GroupedEntropyCoder(BaseTrainableModule, EntropyCoderInterface):
    def __init__(self, entropy_coders : List[EntropyCoderInterface], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.entropy_coders = entropy_coders

    def encode(self, data : List[Any], *args, prior=None, **kwargs) -> bytes:
        assert(len(data) == len(self.entropy_coders))
        byte_strings = []
        for i in range(len(data)):
            # print(data[i])
            prior_i = prior[i] if isinstance(prior, (list, tuple)) else None
            byte_string = self.entropy_coders[i].encode(data[i], *args, prior=prior_i, **kwargs)
            # print(len(byte_string))
            byte_strings.append(byte_string)

        return merge_bytes(byte_strings, num_segments=len(self.entropy_coders))

    def decode(self, byte_string: bytes, *args, prior=None, **kwargs) -> List[Any]:
        byte_strings = split_merged_bytes(byte_string, num_segments=len(self.entropy_coders))

        data = []
        for i in range(len(self.entropy_coders)):
            prior_i = prior[i] if isinstance(prior, (list, tuple)) else None
            data.append(self.entropy_coders[i].decode(byte_strings[i], *args, prior=prior_i, **kwargs))
            # print(data[i])
        return data

    def set_stream(self, byte_string: bytes, *args, **kwargs):
        byte_strings = b''
        for i in range(len(byte_string)):
            byte_string = self.entropy_coders[i].set_stream(byte_strings[i], *args, **kwargs)

    def decode_from_stream(self, *args, prior=None, **kwargs) -> List[Any]:
        data = []
        for i in range(len(self.entropy_coders)):
            prior_i = prior[i] if isinstance(prior, (list, tuple)) else None
            data.append(self.entropy_coders[i].decode_from_stream(*args, prior=prior_i, **kwargs))
        return data

    def train_full(self, dataloader, *args, **kwargs) -> None:   
        if len(dataloader) == 0: return
        dataloader_split = [[] for i in range(len(dataloader[0]))]
        for data in dataloader:
            for i in range(len(dataloader_split)):
                dataloader_split[i].append(data[i])

        for i in range(len(self.entropy_coders)):
            if isinstance(self.entropy_coders[i], TrainableModuleInterface):
                self.entropy_coders[i].train_full(dataloader_split[i], *args, **kwargs)

    def train_iter(self, data, *args, **kwargs) -> None:
        for i in range(len(self.entropy_coders)):
            if isinstance(self.entropy_coders[i], TrainableModuleInterface):
                self.entropy_coders[i].train_full(data[i], *args, **kwargs)
