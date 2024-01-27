from configs.class_builder import ClassBuilder, ParamSlot
from configs.import_utils import import_config_from_module
from . import enwik8 as base_module

config = import_config_from_module(base_module).update_args(
    segment_length=ParamSlot("segment_length", default=16*1024),
)