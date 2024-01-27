import cbench.nn.trainer
from configs.import_utils import import_config_from_module, import_all_config_from_dir
from configs.class_builder import ClassBuilder, ParamSlot

config = ClassBuilder(
    cbench.nn.trainer.TorchGeneralTrainer,
).add_all_kwargs_as_param_slot()\
.update_args(
    dataloader_training=ParamSlot("dataloader_training"),
    dataloader_validation=ParamSlot("dataloader_validation"),
    num_epoch=ParamSlot("num_epoch", default=100),
    check_val_every_n_epoch=ParamSlot("check_val_every_n_epoch", default=1),
    model_wrapper_config=ParamSlot("model_wrapper_config", 
        choices=import_all_config_from_dir("model_wrapper_configs", caller_file=__file__),
        default="empty",
    ),
    trainer_config=ParamSlot("trainer_config", 
        choices=import_all_config_from_dir("trainer_configs", caller_file=__file__),
        default="empty",
    ),
    param_scheduler_configs=ParamSlot("param_scheduler_configs",
        choices=import_all_config_from_dir("param_scheduler_configs", caller_file=__file__),
        default="empty",
    ),
)

