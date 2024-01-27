from configs.class_builder import ClassBuilder, ClassBuilderList, ParamSlot
from cbench.modules.entropy_coder.base import TorchQuantizedEntropyCoder

config = ClassBuilder(TorchQuantizedEntropyCoder).add_all_kwargs_as_param_slot()