import cbench.benchmark
from configs.import_utils import import_config_from_module
from configs.class_builder import ClassBuilder, ParamSlot
from configs.env import DEFAULT_CPU_CORES
# import importlib

config = ClassBuilder(
    cbench.benchmark.BasicLosslessCompressionBenchmark,
    codec=ParamSlot("codec"),
    dataloader=ParamSlot("dataloader"),
    # ParamSlot("codec", default=import_config_from_module(default_codec)),
    # ParamSlot("dataloader", default=import_config_from_module(default_dataloader).update_slot_params(
    #     dataset=import_config_from_module(default_dataset)
    #     ),
    # ),
    num_testing_workers=ParamSlot("num_testing_workers", default=DEFAULT_CPU_CORES),
    # skip_trainer_testing=ParamSlot(),
    # force_basic_testing=ParamSlot(),
    # force_testing_device=ParamSlot(),
    # testing_variable_rate_levels=ParamSlot(),
).add_all_kwargs_as_param_slot()
# from configs.dataloaders.basic import config as default_dataloader2
# print(default_dataloader2)

