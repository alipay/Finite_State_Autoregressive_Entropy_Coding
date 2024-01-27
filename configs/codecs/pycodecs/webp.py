from configs.class_builder import ClassBuilder, ParamSlot
from cbench.codecs.pycodecs import ImageWebPCodec

config = ClassBuilder(
    ImageWebPCodec,
).add_all_kwargs_as_param_slot()