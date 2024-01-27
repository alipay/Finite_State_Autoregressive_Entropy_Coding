from configs.class_builder import ClassBuilder, ParamSlot
from cbench.codecs.pycodecs import PILWebPLosslessCodec

config = ClassBuilder(
    PILWebPLosslessCodec,
).add_all_kwargs_as_param_slot()