from configs.class_builder import ClassBuilder, ParamSlot
from cbench.codecs.pycodecs import ImageFLIFCodec

config = ClassBuilder(
    ImageFLIFCodec,
).add_all_kwargs_as_param_slot()