from configs.class_builder import ClassBuilder, ParamSlot

from cbench.data.datasets.image_dataset_wrapper import PILImageDatasetWrapper

config = ClassBuilder(
    PILImageDatasetWrapper,
    dataset=ParamSlot("dataset"),
)