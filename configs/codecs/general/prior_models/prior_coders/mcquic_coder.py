from configs.class_builder import ClassBuilder, ParamSlot
from cbench.modules.prior_model.prior_coder.mcquic_coder import McQuicPriorCoder

config = ClassBuilder(McQuicPriorCoder)\
    .add_all_kwargs_as_param_slot()