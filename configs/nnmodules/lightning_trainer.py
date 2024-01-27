from cbench.nn.trainer import LightningNNTrainerEngine

from configs.import_utils import import_config_from_module, import_all_config_from_dir
from configs.class_builder import ClassBuilder, ParamSlot

config = ClassBuilder(LightningNNTrainerEngine,
    train_loader=ParamSlot(),
    val_loader=ParamSlot(),
    test_loader=ParamSlot(),
    trainer_config=ParamSlot(
        choices=import_all_config_from_dir("trainer_configs", caller_file=__file__),
        default="empty",
    ),
    max_epochs=ParamSlot(default=100),
)