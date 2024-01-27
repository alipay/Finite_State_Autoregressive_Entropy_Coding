import traceback
from configs.class_builder import ClassBuilder, ClassBuilderList, ParamSlot

import torchvision
from cbench.data.datasets.basic import ConcatMappingDataset
from cbench.data.datasets.tensors import ResizedImageNetDataset

data_path = "data/ImageNet/Imagenet32_train_npz/"

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
    # if not os.path.exists(data_path):
    oss.sync_directory(data_path, data_path)
except:
    traceback.print_exc()
    print("OSS not applicable! Prepare the dataset locally!")


config = ClassBuilder(
    ConcatMappingDataset,
    ClassBuilderList(
        *[
            ClassBuilder(
                ResizedImageNetDataset,
                file_name=os.path.join(data_path, np_file),
                image_size=(32, 32),
            ) for np_file in os.listdir(data_path)
        ]
    ),
    transform=ClassBuilder(
        torchvision.transforms.ToTensor,
    ),
)

if __name__ == "__main__":
    config.build_class()