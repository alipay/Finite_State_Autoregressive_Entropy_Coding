import cbench.benchmark
from configs.import_utils import import_config_from_module
from configs.class_builder import ClassBuilder, ClassBuilderList, ParamSlot

config = ClassBuilder(
    cbench.benchmark.GroupedLosslessCompressionBenchmark,
    codec_group=ParamSlot("codec_group"),
    dataloader=ParamSlot("dataloader"),
)

# import cbench.benchmark
# from configs.import_utils import import_config_from_module
# from configs.class_builder import ClassBuilder, ParamSlot

# from . import lossless_compression as base_module

# config = import_config_from_module(base_module).update_class(
#     cbench.benchmark.GroupedLosslessCompressionBenchmark
# )