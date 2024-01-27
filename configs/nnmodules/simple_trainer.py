from cbench.nn.trainer import SimpleNNTrainerEngine

from configs.import_utils import import_config_from_module, import_all_config_from_dir
from configs.class_builder import ClassBuilder, ParamSlot

config = ClassBuilder(SimpleNNTrainerEngine,
    train_loader=ParamSlot(),
    val_loader=ParamSlot(),
    test_loader=ParamSlot(),
    max_epochs=ParamSlot(default=100),
)