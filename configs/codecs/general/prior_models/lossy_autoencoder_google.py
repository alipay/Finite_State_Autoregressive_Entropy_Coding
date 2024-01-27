from configs.class_builder import ClassBuilder, ParamSlot
from configs.import_utils import import_class_builder_from_module
from cbench.modules.prior_model.autoencoder_v2 import GoogleLossyAutoEncoderPriorModel

from . import base_autoencoder as base_module

config = import_class_builder_from_module(base_module)\
    .update_class(GoogleLossyAutoEncoderPriorModel)\
    .add_all_kwargs_as_param_slot()\
    .update_args(
    prior_coder=ParamSlot(),
)