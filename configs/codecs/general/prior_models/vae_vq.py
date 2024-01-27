from email.mime import base
from configs.class_builder import ClassBuilder, ParamSlot
from configs.import_utils import import_class_builder_from_module

from . import base_lossless_autoencoder as base_module
from .prior_coders import vq as prior_coder_module
config = import_class_builder_from_module(base_module).update_slot_params(
    prior_coder=import_class_builder_from_module(prior_coder_module)
)