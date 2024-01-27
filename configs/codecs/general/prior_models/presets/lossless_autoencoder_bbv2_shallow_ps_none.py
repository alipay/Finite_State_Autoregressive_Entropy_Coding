from configs.class_builder import ClassBuilder, ParamSlot
from configs.import_utils import import_class_builder_from_module

from .. import lossless_autoencoder_bbv2 as base_module

config = import_class_builder_from_module(base_module)\
    .update_args(
                distortion_type="none",
                out_channels=6,
                hidden_channels=64,
                num_downsample_layers=1,
                upsample_method="pixelshuffle",
                num_residual_layers=1,
)