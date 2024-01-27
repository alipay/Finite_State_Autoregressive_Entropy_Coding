import glob
import os
import torch
import numpy as np
from typing import Iterable
from torchvision.transforms import ToTensor, ToPILImage

from .basic import MappingDataset, IterableDataset


# helps loading only image data from PIL image datasets such as torchvision cifar10
class TensorImageDatasetWrapper(MappingDataset): 
    def __init__(self, dataset: Iterable, *args,
                 **kwargs):
        self.dataset = dataset

    def __getitem__(self, index):
        data = self.dataset[index]
        if isinstance(data, torch.Tensor):
            return data
        if isinstance(data, (list, tuple)):
            for data_item in data:
                if isinstance(data_item, torch.Tensor):
                    return data_item
        raise ValueError("data {} is invalid for TensorImageDatasetWrapper!".format(data))
        # return None

    def __len__(self):
        return len(self.dataset)


class NumpyImageDatasetWrapper(MappingDataset): 
    def __init__(self, dataset: Iterable, *args,
                 **kwargs):
        self.dataset = dataset

    def __getitem__(self, index):
        data = self.dataset[index]
        if isinstance(data, torch.Tensor):
            # BGR to RGB; CHW to HWC; float to uint8
            return (data[[2, 1, 0]].permute(1, 2, 0) * 255).numpy().astype(np.uint8)
        if isinstance(data, np.ndarray):
            return data
        if isinstance(data, (list, tuple)):
            for data_item in data:
                if isinstance(data_item, torch.Tensor):
                    # return ToPILImage()(data_item)
                    return (data_item[[2, 1, 0]].permute(1, 2, 0) * 255).numpy().astype(np.uint8)
                if isinstance(data_item, np.ndarray):
                    return data_item
        raise ValueError("data {} is invalid for NumpyImageDatasetWrapper!".format(data))
        # return None

    def __len__(self):
        return len(self.dataset)


class PILImageDatasetWrapper(MappingDataset): 
    def __init__(self, dataset: Iterable, *args,
                 **kwargs):
        self.dataset = dataset

    def __getitem__(self, index):
        data = self.dataset[index]
        if isinstance(data, (torch.Tensor, np.ndarray)):
            return ToPILImage()(data)
        if isinstance(data, np.ndarray):
            return data
        if isinstance(data, (list, tuple)):
            for data_item in data:
                if isinstance(data_item, (torch.Tensor, np.ndarray)):
                    return ToPILImage()(data_item)
        raise ValueError("data {} is invalid for NumpyImageDatasetWrapper!".format(data))
        # return None

    def __len__(self):
        return len(self.dataset)