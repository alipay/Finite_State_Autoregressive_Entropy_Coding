from configs.class_builder import ClassBuilder, ParamSlot
from cbench.codecs.pycodecs import PILPNGCodec

config = ClassBuilder(
    PILPNGCodec,
).add_all_kwargs_as_param_slot()