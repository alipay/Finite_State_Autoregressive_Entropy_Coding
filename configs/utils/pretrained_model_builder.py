from typing import Callable, List, Union
import os
import csv
import pickle
import copy
import hashlib
from cbench.nn.base import SelfTrainableInterface

from configs.class_builder import ClassBuilder, ClassBuilderBase, ClassBuilderList, NamedParamBase, ParamSlot
from configs.env import DEFAULT_PRETRAINED_PATH

class PretrainedModelBuilder(ClassBuilder):
    def __init__(self, class_init: Callable, *args, 
        pretrained_output_dir=DEFAULT_PRETRAINED_PATH,
        **kwargs):
        assert(issubclass(class_init, SelfTrainableInterface))
        self.pretrained_output_dir = pretrained_output_dir
        if 'output_dir' in kwargs:
            print("output_dir is not a valid parameter for PretrainedModelBuilder!")
            print("Use pretrained_output_dir instead!")
            kwargs.pop('output_dir')
        # require a trainer slot
        super().__init__(class_init, *args, 
            trainer=ParamSlot("trainer"),
            **kwargs
        )

    def build_class(self, *args, **kwargs):
        model_name = self.get_name_under_limit()
        pretrained_output_dir = os.path.join(self.pretrained_output_dir, model_name)
        if not os.path.exists(pretrained_output_dir):
            os.makedirs(pretrained_output_dir, exist_ok=True)
        with open(os.path.join(pretrained_output_dir, "exp_name.txt"), 'w') as f:
            f.write(self.name)
        return super().build_class(*args, output_dir=pretrained_output_dir, **kwargs)