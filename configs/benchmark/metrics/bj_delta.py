from cbench.benchmark.metrics.bj_delta import BJDeltaMetric

from configs.import_utils import import_config_from_module
from configs.class_builder import ClassBuilder, ParamSlot

config = ClassBuilder(
    BJDeltaMetric,
).add_all_kwargs_as_param_slot()

