from configs.class_builder import ClassBuilder, ParamSlot
from cbench.modules.prior_model.autoencoder_v2 import AutoEncoderPriorModel

config = ClassBuilder(AutoEncoderPriorModel,
).add_all_kwargs_as_param_slot()