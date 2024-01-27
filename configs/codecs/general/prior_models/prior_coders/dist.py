from configs.class_builder import ClassBuilder, ParamSlot
from cbench.modules.prior_model.prior_coder import DistributionPriorCoder

config = ClassBuilder(DistributionPriorCoder,
    in_channels=ParamSlot(),
    latent_channels=ParamSlot(),
    # TODO: add a base class for these params
    skip_layers_if_equal_channels=ParamSlot(),
    freeze_input_layer=ParamSlot(),
    freeze_output_layer=ParamSlot(),
).add_all_kwargs_as_param_slot()