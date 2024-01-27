from configs.class_builder import ClassBuilder, ParamSlot
import torchvision
from torchvision.datasets.utils import extract_archive

# oss dataset download/upload
try:
    import os
    from configs.oss_utils import OSSUtils
    oss = OSSUtils()
    data_dir = "data/celeba"
    # if os.path.exists(cifar10_download_file):
    #     if not oss.exists(cifar10_download_file):
    #         oss.upload(cifar10_download_file, cifar10_download_file)
    # else:
    #     oss.download_archive_and_extract(cifar10_download_file, cifar10_download_file)
    if not os.path.exists(data_dir):
        oss.sync_directory(data_dir, data_dir)
    download = False
except:
    print("OSS not applicable! Using normal download!")
    download = True

config = ClassBuilder(
    torchvision.datasets.CelebA,
    root="data/celeba",
    transform=ClassBuilder(
        torchvision.transforms.ToTensor,
    ),
    download=download,
    train=ParamSlot("train", default=True),
)