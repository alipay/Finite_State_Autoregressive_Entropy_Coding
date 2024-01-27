from configs.class_builder import ClassBuilder, ParamSlot
from cbench.modules.prior_model.prior_coder import HierarchicalNNPriorCoder

config = ClassBuilder(HierarchicalNNPriorCoder,
    encoders=ParamSlot(),
    decoders=ParamSlot(),
    prior_coders=ParamSlot(),
).add_all_kwargs_as_param_slot()