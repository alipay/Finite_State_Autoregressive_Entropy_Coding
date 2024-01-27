from configs.class_builder import ClassBuilder, ParamSlot
from configs.import_utils import import_class_builder_from_module
from cbench.modules.prior_model.autoencoder_v2 import LosslessAutoEncoderPriorModelBackboneV2

from . import base_autoencoder as base_module

config = import_class_builder_from_module(base_module)\
    .update_class(LosslessAutoEncoderPriorModelBackboneV2)\
    .add_all_kwargs_as_param_slot()\
    .update_args(
    distortion_type=ParamSlot(default="ce"),
    prior_coder=ParamSlot(),
    hidden_channels=ParamSlot(),
    num_downsample_layers=ParamSlot(),
    num_residual_layers=ParamSlot(),
)