import cbench.data.dataloaders
from configs.class_builder import ClassBuilder, ParamSlot

config = ClassBuilder(
    cbench.data.dataloaders.BasicDataLoader,
    ParamSlot("dataset"),
    max_samples=ParamSlot(),
)
