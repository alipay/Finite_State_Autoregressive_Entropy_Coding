from configs.class_builder import ClassBuilder, ParamSlot

from cbench.data.datasets.basic import MappingDataset
import torch

class RandomImageGenerator(MappingDataset):
    def __init__(self, *args, length=128, image_size=(3, 32, 32), **kwargs):
        super().__init__(*args, **kwargs)
        self.length = length
        self.image_size = image_size

    def __getitem__(self, index):
        torch.manual_seed(index)
        img = torch.rand(*self.image_size)
        return img

    def __len__(self):
        return self.length

config = ClassBuilder(RandomImageGenerator,
    length=ParamSlot(),
    image_size=ParamSlot(),
)