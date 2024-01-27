from configs.import_utils import import_raw_config_from_module
from configs.env import DEFAULT_GPU_DEVICES

from . import pl_base as base_module

config = import_raw_config_from_module(base_module)
config.update(
    accelerator="gpu", 
    devices=DEFAULT_GPU_DEVICES, 
    strategy="ddp",
    # precision=16
)