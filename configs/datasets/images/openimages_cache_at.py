from configs.class_builder import ClassBuilder, ParamSlot, ClassBuilderList
from configs.import_utils import import_config_from_module
from . import openimages as base_module

config = import_config_from_module(base_module).update_args(
    cache_after_transform=True,
)