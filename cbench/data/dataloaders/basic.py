# TODO: torch related classes should be implemented independently
import torch.utils.data

from ..base import DataLoaderInterface


class BasicDataLoader(DataLoaderInterface):
    # TODO: check dataset iterable and indexable
    def __init__(self, dataset, *args, max_samples=-1, **kwargs):
        self.dataset = dataset
        self.max_samples = max_samples

    def get_length(self):
        if self.max_samples > 0:
            return min(len(self.dataset), self.max_samples)
        else:
            return len(self.dataset)

    def iterate(self):
        if self.max_samples > 0:
            return iter(self.dataset[:self.get_length()])
        else:
            return iter(self.dataset)

    def get_data_at(self, index):
        if self.max_samples > 0:
            index = min(index, self.max_samples-1)
        return self.dataset[index]


class PyTorchDataLoader(torch.utils.data.DataLoader, DataLoaderInterface):
    def __init__(self, dataset, *args, **kwargs):
        # TODO: maybe should check dataset class?
        # kwargs['num_workers'] = 0
        # kwargs['batch_size'] = 1
        super().__init__(dataset, *args, **kwargs)

    def get_length(self):
        return len(self)

    def iterate(self):
        # return next(self)
        for data in self:
            yield data

    def get_data_at(self, index):
        return self.dataset[index]


