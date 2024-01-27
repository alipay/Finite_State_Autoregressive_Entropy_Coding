from configs.class_builder import ClassBuilder, ParamSlot, ClassBuilderList
from configs.import_utils import import_config_from_module
from . import clic as base_module

from cbench.data.transforms import AlignedCrop

config = import_config_from_module(base_module).update_args(
    root="data/CLIC/test",
    enable_augmentation=False,
    post_transforms=ClassBuilderList(
        ClassBuilder(AlignedCrop)
    )
)