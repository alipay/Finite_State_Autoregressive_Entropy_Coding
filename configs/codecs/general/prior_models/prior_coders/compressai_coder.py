from configs.class_builder import ClassBuilder, ParamSlot
from cbench.modules.prior_model.prior_coder.compressai_coder import CompressAIEntropyBottleneckPriorCoder

config = ClassBuilder(CompressAIEntropyBottleneckPriorCoder)\
    .add_all_kwargs_as_param_slot()