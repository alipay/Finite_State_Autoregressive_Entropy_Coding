from configs.class_builder import ClassBuilder, ParamSlot

import torchvision

config = ClassBuilder(
    torchvision.datasets.MNIST,
    root="data/mnist",
    transform=ClassBuilder(
        torchvision.transforms.ToTensor,
    ),
    download=True,
    train=ParamSlot("train", default=True),
)