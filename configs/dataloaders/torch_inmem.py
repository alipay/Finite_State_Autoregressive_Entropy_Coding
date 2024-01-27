from configs.import_utils import import_config_from_module
from configs.class_builder import ClassBuilder, ParamSlot
from . import torch as base_module

config = import_config_from_module(base_module).update_args(
    num_workers=0,
    persistent_workers=False,
)
