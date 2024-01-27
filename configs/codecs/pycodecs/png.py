from configs.class_builder import ClassBuilder, ParamSlot
from cbench.codecs.pycodecs import ImagePNGCodec

config = ClassBuilder(
    ImagePNGCodec,
).add_all_kwargs_as_param_slot()