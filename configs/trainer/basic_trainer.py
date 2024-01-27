import cbench.benchmark.trainer
from configs.import_utils import import_config_from_module
from configs.class_builder import ClassBuilder, ParamSlot

config = ClassBuilder(
    cbench.benchmark.trainer.BasicTrainer,
)

