from configs.class_builder import ClassBuilder, ParamSlot
from configs.import_utils import import_class_builder_from_module

from .. import lossless_autoencoder_bbv2 as base_module

config = import_class_builder_from_module(base_module)\
    .update_args(
                distortion_type="normal_quant",
                out_channels=6,
)