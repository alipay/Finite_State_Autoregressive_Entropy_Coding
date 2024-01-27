from configs.class_builder import ClassBuilder, ParamSlot
from configs.env import DEFAULT_MAX_MEMORY_CACHE

from cbench.data.datasets.torchvision_datasets import ImageDatasetWrapper

# oss dataset download/upload
try:
    import os
    from configs.oss_utils import OSSUtils
    oss = OSSUtils()
    data_dir = "data/openimages/clean_train"
    # if os.path.exists(cifar10_download_file):
    #     if not oss.exists(cifar10_download_file):
    #         oss.upload(cifar10_download_file, cifar10_download_file)
    # else:
    #     oss.download_archive_and_extract(cifar10_download_file, cifar10_download_file)
    if not os.path.exists(data_dir):
        oss.sync_directory(data_dir, data_dir)
except:
    print("OSS not applicable! Manually download files!")

config = ClassBuilder(
    ImageDatasetWrapper,
    root="data/openimages/clean_train",
    enable_augmentation=True,
    random_crop_size=128,
    max_cache_size=DEFAULT_MAX_MEMORY_CACHE,
)