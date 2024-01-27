from configs.class_builder import ClassBuilder, ParamSlot
from cbench.codecs.pycodecs import PILJPEGCodec

config = ClassBuilder(
    PILJPEGCodec,
).add_all_kwargs_as_param_slot()