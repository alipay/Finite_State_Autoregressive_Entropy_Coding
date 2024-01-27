from configs.class_builder import ClassBuilder, ParamSlot
from cbench.modules.preprocessor.image_predictor import ThreeWayAutoregressivePreprocessor

config = ClassBuilder(ThreeWayAutoregressivePreprocessor,
).add_all_kwargs_as_param_slot()