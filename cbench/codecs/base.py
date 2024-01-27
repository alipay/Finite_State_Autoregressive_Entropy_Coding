import abc
import pickle
from typing import Any, Dict, Iterator, List, Tuple

from cbench.modules.base import BaseModule, TrainableModuleInterface
from cbench.nn.base import NNTrainableModule, PLNNTrainableModule
from cbench.utils.logging_utils import MetricLogger

class CodecInterface(abc.ABC):
    @abc.abstractmethod
    def compress(self, data, *args, **kwargs) -> bytes:
        pass

    @abc.abstractmethod
    def decompress(self, data: bytes, *args, **kwargs):
        pass


class PickleSerilizeFunctions(abc.ABC):
    def serialize(self, data, *args, **kwargs) -> bytes:
        return pickle.dumps(data)

    def deserialize(self, data: bytes, *args, **kwargs):
        return pickle.loads(data)


class BaseCodec(BaseModule, CodecInterface, PickleSerilizeFunctions):
    pass
    # moved to BaseModule
    # def __init__(self, *args, **kwargs):
    #     self._profiler = MetricLogger()

    # @property
    # def profiler(self):
    #     return self._profiler

    # @profiler.setter
    # def profiler(self, profiler):
    #     self._profiler = profiler

# TODO: this is similar to BaseTrainableModule!
# consider merge it!
class BaseTrainableCodec(BaseCodec, NNTrainableModule): # inherit from (PL)NNTrainableModule to enable nn modules
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        NNTrainableModule.__init__(self)

    def forward(self, *args, **kwargs):
        # default forward pass to the compress process
        self.compress(*args, **kwargs)

    def get_parameters(self, *args, **kwargs) -> Dict[str, Any]:
        parameters = dict()
        for name, module in self.get_named_submodules():
            if isinstance(module, TrainableModuleInterface):
                parameters[name] = (module.get_parameters(*args, **kwargs))
        return parameters

    # def iter_trainable_parameters(self, *args, **kwargs) -> Iterator:
    #     for name, module in self.get_named_submodules():
    #         if isinstance(module, TrainableModuleInterface):
    #             for param in module.iter_trainable_parameters():
    #                 yield param

    def load_parameters(self, parameters: Dict[str, Any], *args, **kwargs) -> None:
        for name, module in self.get_named_submodules():
            if isinstance(module, TrainableModuleInterface):
                module.load_parameters(parameters[name], *args, **kwargs)

    def update_state(self, *args, **kwargs) -> None:
        for module in self.get_submodules():
            if isinstance(module, TrainableModuleInterface):
                module.update_state(*args, **kwargs)

    # def train_full(self, dataloader, *args, **kwargs) -> None:
    #     for module in self.get_submodules():
    #         if isinstance(module, TrainableModuleInterface):
    #             module.train_full(dataloader, *args, **kwargs)

    # def train_iter(self, data, *args, **kwargs) -> None:
    #     for module in self.get_submodules():
    #         if isinstance(module, TrainableModuleInterface):
    #             module.train_iter(data, *args, **kwargs)

