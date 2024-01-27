import cbench.data.dataloaders
import torch.utils.data
from configs.class_builder import ClassBuilder, ParamSlot
from configs.env import DEFAULT_CPU_CORES

config = ClassBuilder(
    # torch.utils.data.DataLoader,
    cbench.data.dataloaders.PyTorchDataLoader,
    dataset=ParamSlot("dataset"),
    batch_size=ParamSlot("batch_size", default=1),
    shuffle=ParamSlot("shuffle", default=True),
    num_workers=ParamSlot("num_workers", default=DEFAULT_CPU_CORES),
    persistent_workers=(DEFAULT_CPU_CORES > 0), # for using cache with dataset
)
