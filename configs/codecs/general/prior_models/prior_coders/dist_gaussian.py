from configs.class_builder import ClassBuilder, ParamSlot
from configs.import_utils import import_class_builder_from_module

from . import dist as base_module

from cbench.modules.prior_model.prior_coder import GaussianDistributionPriorCoder

config = import_class_builder_from_module(base_module)\
    .update_class(GaussianDistributionPriorCoder)\
#     .update_args(
# )