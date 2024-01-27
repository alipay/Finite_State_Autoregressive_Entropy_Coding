from configs.class_builder import ClassBuilder, ParamSlot

from cbench.data.datasets.image_dataset_wrapper import NumpyImageDatasetWrapper

config = ClassBuilder(
    NumpyImageDatasetWrapper,
    dataset=ParamSlot("dataset"),
)