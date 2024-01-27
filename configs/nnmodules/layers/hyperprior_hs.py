from configs.class_builder import ClassBuilder, ParamSlot
from cbench.nn.models.google import HyperpriorHyperSynthesisModel

config = ClassBuilder(HyperpriorHyperSynthesisModel,
    N=ParamSlot(),
    M=ParamSlot(),
).add_all_kwargs_as_param_slot()