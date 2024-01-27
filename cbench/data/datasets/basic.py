# TODO: this seems to be similar to torch.utils.data, maybe use pytorch dataset instead
import bisect
import numpy as np
from typing import Callable, List
from torch.utils.data.dataset import Dataset

class BasicDataset(object):
    # TODO: check dataset iterable and indexable
    def __init__(self, *args, transform : Callable = None, **kwargs):
        self.transform = transform

    def do_transform(self, data):
        if self.transform is not None:
            data = self.transform(data)
        return data

class MappingDataset(BasicDataset):
    def __getitem__(self, index):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()


class IterableDataset(BasicDataset):
    def __iter__(self):
        raise NotImplementedError()


class ConcatMappingDataset(MappingDataset):
    def __init__(self, datasets : List[MappingDataset], *args, transform : Callable = None, **kwargs):
        super().__init__(*args, transform=transform, **kwargs)
        self.datasets = datasets
        self.dataset_lengths = [len(d) for d in datasets]
        self.cumulative_sizes = np.cumsum(np.array(self.dataset_lengths))

    # from torch.utils.data.dataset.Dataset
    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        data_sample = self.datasets[dataset_idx][sample_idx]
        return self.do_transform(data_sample)

    def __len__(self):
        return self.cumulative_sizes[-1]
