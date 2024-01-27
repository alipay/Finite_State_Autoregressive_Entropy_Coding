from configs.class_builder import ClassBuilder, ParamSlot

import torchvision
from cbench.data.datasets.tensors import ResizedImageNetDataset

data_path = "data/ImageNet/Imagenet64_val_npz/val_data.npz"

# oss dataset download/upload
try:
    import os
    from configs.oss_utils import OSSUtils
    oss = OSSUtils()
    # if os.path.exists(cifar10_download_file):
    #     if not oss.exists(cifar10_download_file):
    #         oss.upload(cifar10_download_file, cifar10_download_file)
    # else:
    #     oss.download_archive_and_extract(cifar10_download_file, cifar10_download_file)
    if not os.path.exists(data_path):
        oss.sync_file(data_path, data_path)
except:
    print("OSS not applicable! Prepare the dataset locally!")


config = ClassBuilder(
    ResizedImageNetDataset,
    file_name=data_path,
    image_size=(64, 64),
    transform=ClassBuilder(
        torchvision.transforms.ToTensor,
    ),
)

if __name__ == "__main__":
    config.build_class()