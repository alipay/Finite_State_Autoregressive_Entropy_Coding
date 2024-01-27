from configs.class_builder import ClassBuilder, ParamSlot
import torchvision
from torchvision.datasets.utils import extract_archive

# oss dataset download/upload
try:
    import os
    from configs.oss_utils import OSSUtils
    oss = OSSUtils()
    cifar10_download_file = "data/cifar10/cifar-10-python.tar.gz"
    # if os.path.exists(cifar10_download_file):
    #     if not oss.exists(cifar10_download_file):
    #         oss.upload(cifar10_download_file, cifar10_download_file)
    # else:
    #     oss.download_archive_and_extract(cifar10_download_file, cifar10_download_file)
    if not os.path.exists(cifar10_download_file):
        oss.sync_file(cifar10_download_file, cifar10_download_file)
        extract_archive(cifar10_download_file)
    download = False
except:
    print("OSS not applicable! Using normal download!")
    download = True

config = ClassBuilder(
    torchvision.datasets.CIFAR10,
    root="data/cifar10",
    transform=ClassBuilder(
        torchvision.transforms.ToTensor,
    ),
    download=download,
    train=ParamSlot("train", default=True),
)