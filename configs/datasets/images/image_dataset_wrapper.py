from configs.class_builder import ClassBuilder, ParamSlot

from cbench.data.datasets.image_dataset_wrapper import TensorImageDatasetWrapper

config = ClassBuilder(
    TensorImageDatasetWrapper,
    dataset=ParamSlot("dataset"),
)