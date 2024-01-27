from configs.class_builder import ClassBuilder, ParamSlot
from configs.import_utils import import_class_builder_from_module, import_all_config_from_dir
from cbench.modules.entropy_coder.dist_entropy import AutoregressiveImplDistributionEntropyCoder

from . import dist as base_module

ar_offsets_choices = dict(
    none=None,
)
ar_offsets_choices.update(**import_all_config_from_dir("ar_offsets", caller_file=__file__))

config = import_class_builder_from_module(base_module).update_class(AutoregressiveImplDistributionEntropyCoder)\
    .add_all_kwargs_as_param_slot()\
    .update_args(
        ar_offsets=ParamSlot(
            choices=ar_offsets_choices,
            default="none",
        )
    )
