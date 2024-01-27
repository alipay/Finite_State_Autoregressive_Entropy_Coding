from configs.class_builder import ClassBuilder, ParamSlot, ClassBuilderList
from configs.import_utils import import_config_from_module
from . import openimages as base_module

from cbench.data.transforms import AlignedCrop

# oss dataset download/upload
try:
    import os
    from configs.oss_utils import OSSUtils
    oss = OSSUtils()
    data_dir = "data/openimages/clean_val"
    # if os.path.exists(cifar10_download_file):
    #     if not oss.exists(cifar10_download_file):
    #         oss.upload(cifar10_download_file, cifar10_download_file)
    # else:
    #     oss.download_archive_and_extract(cifar10_download_file, cifar10_download_file)
    if not os.path.exists(data_dir):
        oss.sync_directory(data_dir, data_dir)
except:
    print("OSS not applicable! Manually download files!")

config = import_config_from_module(base_module).update_args(
    root="data/openimages/clean_val",
    enable_augmentation=False,
    post_transforms=ClassBuilderList(
        ClassBuilder(AlignedCrop, max_size=(128, 128))
    ),
)