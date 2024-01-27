import glob
import os
from typing import List
import torch
import numpy as np
import scipy.io.matlab as mio

from .basic import MappingDataset, IterableDataset

class TensorFileDataset(MappingDataset): 
    def __init__(self, file_name, *args,
                 tensor_key : str = None,
                 tensor_shape : List = None,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.file_name = file_name
        self.tensor_key = tensor_key
        self.tensor_shape = tensor_shape
        self.data_loaded = False # lazy load

    def _load_data(self, file_name, tensor_key=None):
        raise NotImplementedError()
    
    def call_load_data(self):
        if not self.data_loaded:
            data = self._load_data(self.file_name, tensor_key=self.tensor_key)
            self.data = data
            if self.tensor_shape is not None:
                self.data = self.data.reshape(data.shape[0], *self.tensor_shape)
            self.data_loaded = True

    def __getitem__(self, index):
        if not self.data_loaded:
            self.call_load_data() 
        return self.do_transform(self.data[index])

    def __len__(self):
        if not self.data_loaded:
            self.call_load_data() 
        return len(self.data)


class NumpyTensorDataset(TensorFileDataset): 
    def _load_data(self, file_name, tensor_key=None):
        data = np.load(file_name)
        if tensor_key is not None:
            if tensor_key in data:
                data = data[tensor_key]
            else:
                raise KeyError(f"tensor_key {tensor_key} not found in file! Available keys are {list(data.keys())}")
        assert(isinstance(data, np.ndarray))
        return data


class ResizedImageNetDataset(NumpyTensorDataset): 
    def __init__(self, file_name, *args, image_size=(32, 32), **kwargs):
        self.image_size = image_size
        super().__init__(file_name, *args, tensor_key="data", tensor_shape=None, **kwargs)

    def _load_data(self, file_name, tensor_key=None):
        data = super()._load_data(file_name, tensor_key)
        # NCHW to NHWC
        data = data.reshape(data.shape[0], 3, *self.image_size)
        data = data.transpose((0, 2, 3, 1))
        return data


class MatlabTensorDataset(TensorFileDataset): 
    def _load_data(self, file_name, tensor_key=None):
        data = mio.loadmat(file_name)
        if tensor_key is not None:
            if tensor_key in data:
                data = data[tensor_key]
            else:
                raise KeyError(f"tensor_key {tensor_key} not found in file! Available keys are {list(data.keys())}")
        assert(isinstance(data, np.ndarray))
        return data


class TorchTensorDataset(TensorFileDataset): 
    def _load_data(self, file_name, tensor_key=None):
        data = torch.load(file_name)
        if tensor_key is not None:
            if tensor_key in data:
                data = data[tensor_key]
            else:
                raise KeyError(f"tensor_key {tensor_key} not found in file! Available keys are {list(data.keys())}")
        assert(isinstance(data, torch.Tensor))
        return data
