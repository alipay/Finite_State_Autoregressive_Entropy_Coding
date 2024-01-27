import os

import torch
import torchvision
from torchvision import transforms as T
from torchvision.datasets.folder import IMG_EXTENSIONS
from torchvision.io import read_image

from typing import Iterable

from .basic import MappingDataset, IterableDataset
from ..transforms import RandomGamma, RandomHorizontalFlip, RandomVerticalFlip, RandomPlanckianJitter, RandomAutocontrast

# helps loading only image data from PIL image datasets such as torchvision cifar10
class ImageDatasetWrapper(torchvision.datasets.VisionDataset): 
    def __init__(self, root: str, 
        num_repeats=1,
        max_num_images=0,
        max_cache_size=0,
        cache_after_transform=None,
        enable_augmentation=True,
        random_crop_size=256,
        random_gamma=True,
        random_planckian_jitter=1.0,
        random_horizontal_flip=0.5,
        random_vertical_flip=0.5,
        random_auto_contrast=0.0,
        post_transforms=[],
        **kwargs):
        transforms = [
            T.ConvertImageDtype(torch.float32),
            # T.ToTensor(),
        ]
        if enable_augmentation:
            if random_crop_size > 0:
                transforms.append(T.RandomCrop(random_crop_size, pad_if_needed=True))
            if random_gamma:
                transforms.append(RandomGamma())
            if random_auto_contrast > 0:
                transforms.append(RandomAutocontrast(p=random_auto_contrast))
            if random_planckian_jitter > 0:
                transforms.append(RandomPlanckianJitter(p=random_planckian_jitter))
            if random_horizontal_flip > 0:
                transforms.append(RandomHorizontalFlip(p=random_horizontal_flip))
            if random_vertical_flip > 0:
                transforms.append(RandomVerticalFlip(p=random_vertical_flip))
        # transforms.append(T.Normalize(0.5, 0.5))
        transforms.extend(post_transforms)
        super().__init__(root, transform=T.Compose(transforms))

        # build image file list
        self.file_list = []
        for root, _, fnames in sorted(os.walk(root, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if any([path.lower().endswith(ext) for ext in IMG_EXTENSIONS]):
                    self.file_list.append(path)
        if num_repeats > 1:
            self.file_list = self.file_list * num_repeats
        if max_num_images > 0:
            self.file_list = self.file_list[:max_num_images]

        if len(self.file_list) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            msg += "Supported extensions are: {}".format(",".join(IMG_EXTENSIONS))
            raise RuntimeError(msg)
        
        # inner cache
        self.use_cache = (max_cache_size > 0)
        self.cache_after_transform = not enable_augmentation if cache_after_transform is None else cache_after_transform
        if self.use_cache:
            self.max_cache_size = max_cache_size
            self.current_cache_size = 0
            self.cache = dict()

    def _fetch_or_add_to_cache(self, index : int):
        if index in self.cache:
            # read from cache
            sample = self.cache[index]
        else:
            # add to cache
            path = self.file_list[index]
            sample = read_image(path)
            if self.cache_after_transform:
                # No need to force RGB. Transforms will handle it.
                if self.transform is not None:
                    sample = self.transform(sample)

            estimated_tensor_bytes = 8 + sample.element_size() * sample.nelement()
            if estimated_tensor_bytes + self.current_cache_size <= self.max_cache_size:
                self.cache[index] = sample.detach()
                self.current_cache_size += estimated_tensor_bytes
        return sample.detach()

    def __getitem__(self, index: int) -> torch.Tensor:
        if self.use_cache:
            sample = self._fetch_or_add_to_cache(index)
            if self.cache_after_transform:
                return sample
        else:
            path = self.file_list[index]
            sample = read_image(path)

        # No need to force RGB. Transforms will handle it.
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def __len__(self) -> int:
        return len(self.file_list)

