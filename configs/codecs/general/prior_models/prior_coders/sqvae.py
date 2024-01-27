from configs.class_builder import ClassBuilder, ParamSlot
from configs.import_utils import import_class_builder_from_module, import_all_config_from_dir

from cbench.modules.prior_model.prior_coder import SQVAEPriorCoder

config = ClassBuilder(SQVAEPriorCoder)\
    .add_all_kwargs_as_param_slot()
