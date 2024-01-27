import cbench.benchmark
from configs.import_utils import import_all_config_from_dir, import_config_from_module
from configs.class_builder import ClassBuilder, ParamSlot

from . import lossless_compression_grouped as base_module
config = import_config_from_module(base_module).update_args(
    need_training=True,
    training_dataloader=ParamSlot("training_dataloader"),
    trainer=ParamSlot("trainer"),
    training_config=ParamSlot("training_config", 
        choices=import_all_config_from_dir("training_configs", caller_file=__file__, convert_to_named_param=False),
        default="empty",
    )
)
