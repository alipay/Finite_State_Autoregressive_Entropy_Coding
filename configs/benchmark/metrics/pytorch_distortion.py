from cbench.benchmark.metrics.pytorch_distortion import PytorchBatchedDistortion

from configs.import_utils import import_config_from_module
from configs.class_builder import ClassBuilder, ParamSlot

config = ClassBuilder(
    PytorchBatchedDistortion,
).add_all_kwargs_as_param_slot()

