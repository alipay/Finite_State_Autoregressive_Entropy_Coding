from configs.import_utils import import_raw_config_from_module

from . import pl_gpu as base_module

config = import_raw_config_from_module(base_module)
config.update(
    gradient_clip_val=1.,
)