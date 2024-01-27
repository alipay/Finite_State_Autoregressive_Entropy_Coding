from configs.class_builder import ClassBuilder, ParamSlot
from configs.import_utils import import_class_builder_from_module, import_all_config_from_dir

from . import dist as base_module

from cbench.modules.prior_model.prior_coder import AutoregressivePriorImplDistributionPriorCoder

# temp workaround for None
ar_offsets_choices = dict(
    none=None,
)
ar_offsets_choices.update(**import_all_config_from_dir("ar_offsets", caller_file=__file__))
config = import_class_builder_from_module(base_module)\
    .update_class(AutoregressivePriorImplDistributionPriorCoder)\
    .add_all_kwargs_as_param_slot()\
    .update_args(
        ar_offsets=ParamSlot(
            choices=ar_offsets_choices,
            default="none",
        )
    )

