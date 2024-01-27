from configs.class_builder import ClassBuilder, ParamSlot
from configs.import_utils import import_class_builder_from_module
from cbench.modules.entropy_coder.dist_entropy import DistributionEntropyCoder

from . import torch_quant as base_module

config = import_class_builder_from_module(base_module).update_class(DistributionEntropyCoder)\
    .add_all_kwargs_as_param_slot()
