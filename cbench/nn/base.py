from typing import Any, Callable, Dict, Iterator, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader

import pytorch_lightning as pl

import logging
from collections import OrderedDict

from cbench.utils.engine import BaseEngine
from cbench.modules.base import TrainableModuleInterface
from cbench.utils.logger import setup_logger

class BasicNNTrainerEngine(BaseEngine):
    """
    Similar to pl.Trainer. Implements BaseEngine to support logging and checkpointing. 
    """    
    def __init__(self, model : nn.Module = None,
        train_loader : DataLoader = None,
        val_loader : DataLoader = None,
        test_loader : DataLoader = None,
        on_initialize_start_hook : Callable = None,
        on_initialize_end_hook : Callable = None,
        on_train_start_hook : Callable = None,
        on_train_end_hook : Callable = None,
        checkpoint_dir="experiments", 
        **kwargs
    ):
        # TODO: checkpoint_dir is confusing to output_dir, consider removing this eventually!
        super().__init__(
            output_dir=checkpoint_dir,
        )
        self.set_model(model)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        # by default testing data is the same as validation data
        self.val_loader = val_loader if val_loader is not None else test_loader
        self.test_loader = test_loader if test_loader is not None else val_loader
        # self.checkpoint_dir = checkpoint_dir
        self.on_initialize_start_hook = on_initialize_start_hook
        self.on_initialize_end_hook = on_initialize_end_hook
        self.on_train_start_hook = on_train_start_hook
        self.on_train_end_hook = on_train_end_hook

        self.extra_options = kwargs
        
        self.has_initialized = False

    def set_model(self, model : nn.Module):
        self.model = model

    # for compability
    @property
    def checkpoint_dir(self):
        return self.output_dir

    @property
    def _module_dict_with_state(self):
        return dict(
            model=self.model,
        )

    def state_dict(self):
        return {k:v.state_dict() for k,v in self._module_dict_with_state.items()}
    
    def load_state_dict(self, checkpoint, strict=True):
        for k,v in self._module_dict_with_state.items():
            if isinstance(v, nn.Module):
                v.load_state_dict(checkpoint[k], strict=strict)
            # else:
            #     v.load_state_dict(checkpoint[k])

    def load_model(self, checkpoint, strict=True):
        self.model.load_state_dict(checkpoint, strict=strict)
    
    def _initialize(self, *args, **kwargs):
        """
        Called before any process start. Could include training scheduling, loading checkpoints, processing model structure, etc.
        """
        raise NotImplementedError()

    def _train(self, *args, **kwargs):
        """
        Main training process.
        """
        raise NotImplementedError()

    def _validate(self, *args, **kwargs) -> Any:
        """
        Main validation process. Should return a metric for comparisons.
        """
        raise NotImplementedError()

    def _test(self, *args, **kwargs):
        """
        Main testing process.
        """
        raise NotImplementedError()

    def initialize(self, *args, **kwargs):
        if self.has_initialized:
            self.logger.warn("Reinitializing Trainer!")
        if self.on_initialize_start_hook is not None:
            self.on_initialize_start_hook(self)
        self._initialize(*args, **kwargs, **self.extra_options)
        self.has_initialized = True
        self.logger.info("Trainer Initialized!")
        if self.on_initialize_end_hook is not None:
            self.on_initialize_end_hook(self)

    def do_train(self, *args, **kwargs):
        # NOTE: should check loader in subprocess, i.e. _train
        # assert(not self.train_loader is None)
        self.initialize()
        self.logger.info("Beginning training...")
        if self.on_train_start_hook is not None:
            self.on_train_start_hook(self)
        self._train(**self.extra_options)
        if self.on_train_end_hook is not None:
            self.on_train_end_hook(self)

    def do_validate(self, *args, **kwargs):
        # assert(not self.val_loader is None)
        self.initialize()
        self.logger.info("Beginning validation...")
        return self._validate(**self.extra_options)    
        
    def do_test(self, *args, **kwargs):
        # assert(not self.test_loader is None)
        self.initialize()
        self.logger.info("Beginning testing...")
        return self._test(**self.extra_options)


class NNCacheImpl(object):
    """
    This class implements caching functions for Modules. Should implement named_children.
    Caches are python dicts.
    """    
    def __init__(self):
        # allow direct access with name
        # self.loss_dict = dict()
        # self.metric_dict = dict()
        # self.hist_dict = dict()
        for cache_name in self.cache_names:
            # setattr(self, cache_name, dict())
            self.__dict__[cache_name] = dict()

        self.optim_state = 0

    def set_optim_state(self, state : Any = 0):
        """ Set optim state for this module and all its submodules

        Args:
            state (str, optional): state name. Defaults to None.
        """        
        self.optim_state = state
        for name, module in self.named_children():
            if isinstance(module, NNTrainableModule):
                module.set_optim_state(state)

    # TODO: define value constraint on caches
    # e.g. for loss and metric, it must be a scalar tensor
    @property
    def cache_names(self) -> List[str]:
        return [
            "common", "loss_dict", "metric_dict", "moniter_dict", "hist_dict", "image_dict"
        ]

    def named_children(self) -> Iterator[Tuple[str, Any]]:
        """
        should be implemented in Module

        Raises:
            NotImplementedError: [description]

        Yields:
            Iterator[Tuple[str, Any]]: [description]
        """        
        raise NotImplementedError()

    def get_cache(self, cache_name="common", prefix=None, recursive=True) -> Dict[str, Any]:
        """
        Recursively get a cache .

        Args:
            cache_name (str, optional): Should in self.cache_names. Defaults to "common".
            prefix ([type], optional): Prefix of returned cache keys. If None, cache_name is used. Defaults to None.
            recursive (bool, optional): Should do recursive get. Defaults to True.

        Returns:
            Dict[str, Any]: Returned caches. All keys has "prefix" variable as prefix.
        """        
        result_dict = dict()
        cache_dict = getattr(self, cache_name)
        if prefix is None:
            prefix = cache_name
        assert isinstance(cache_dict, dict)
        if len(prefix) > 0:
            for k, v in cache_dict.items():
                kn = '/'.join((prefix, k))
                result_dict[kn] = v
        else:
            result_dict.update(cache_dict)
        # recursive get
        if recursive:
            for name, module in self.named_children():
                prefix_new = '/'.join((prefix, name)) if len(prefix) > 0 else name
                # TODO: better to implement named_children ModuleList/nn.ModuleList and ModuleDict/nn.ModuleDict
                if isinstance(module, nn.ModuleList):
                    for idx, sub_module in enumerate(module):
                        if isinstance(sub_module, NNCacheImpl):
                            prefix_sub = '/'.join((prefix_new, f"{idx}"))
                            result_dict.update(sub_module.get_cache(cache_name, prefix=prefix_sub, recursive=True))
                if isinstance(module, nn.ModuleDict):
                    for sub_name, sub_module in module.items():
                        if isinstance(sub_module, NNCacheImpl):
                            prefix_sub = '/'.join((prefix_new, sub_name))
                            result_dict.update(sub_module.get_cache(cache_name, prefix=prefix_sub, recursive=True))
                if isinstance(module, NNCacheImpl):
                    result_dict.update(module.get_cache(cache_name, prefix=prefix_new, recursive=True))
        return result_dict

    def get_raw_cache(self, cache_name="common") -> Dict[str, Any]:
        """Get raw cache data .

        Args:
            cache_name (str, optional): Should in self.cache_names. Defaults to "common".

        Returns:
            Dict[str, Any]: Raw cache dict.
        """        
        return getattr(self, cache_name)

    def get_all_cache(self, prefix=None, recursive=True) -> Dict[str, Dict[str, Any]]:
        """Get all cache values .

        Args:
            prefix ([type], optional): Prefix of returned cache keys. If None, cache_name is used. Defaults to None.
            recursive (bool, optional): Should do recursive get. Defaults to True.

        Returns:
            Dict[str, Dict[str, Any]]: All cache dicts with {cache_name : cache_dict} structure.
        """        
        result_dict = dict()
        for cache_name in self.cache_names:
            result_dict[cache_name] = self.get_cache(cache_name, prefix=prefix, recursive=recursive)
        return result_dict

    def update_cache(self, cache_name="common", **kwargs):
        """Update a cache with kwargs

        Args:
            cache_name (str, optional): Should in self.cache_names. Defaults to "common".
        """        
        cache_dict = getattr(self, cache_name)
        assert(isinstance(cache_dict, dict))
        # for kw, value in kwargs.items():
        #     # TODO: auto avoid memory issue
        #     if isinstance(value, torch.Tensor):
        #         if value.numel() == 1:
        #             value = value.item()
        #         else:
        #             value = value.cpu().detach_()
        #     cache_dict[kw] = value
        cache_dict.update(**kwargs)

    def reset_cache(self, cache_name="common", recursive=True) -> None:
        """Reset a cache to empty dict .

        Args:
            cache_name (str, optional): Should in self.cache_names. Defaults to "common".
            recursive (bool, optional): Should do recursive get. Defaults to True.
        """        
        setattr(self, cache_name, dict())
        # recursive reset
        if recursive:
            for name, module in self.named_children():
                if isinstance(module, nn.ModuleList):
                    for idx, sub_module in enumerate(module):
                        if isinstance(sub_module, NNCacheImpl):
                            sub_module.reset_cache(cache_name)
                if isinstance(module, nn.ModuleDict):
                    for sub_name, sub_module in module.items():
                        if isinstance(sub_module, NNCacheImpl):
                            sub_module.reset_cache(cache_name)
                if isinstance(module, NNCacheImpl):
                    module.reset_cache(cache_name)

    def reset_all_cache(self, recursive=True) -> None:
        """Reset all cache .

        Args:
            recursive (bool, optional): Should do recursive get. Defaults to True.
        """        
        for cache_name in self.cache_names:
            self.reset_cache(cache_name, recursive=recursive)

    # # below funcs are for backward compability
    # @property
    # def loss_dict(self):
    #     return self.loss_dict

    # @property
    # def metric_dict(self):
    #     return self.metric_dict

    def get_loss_dict(self, prefix : str = "losses") -> Dict[str, torch.Tensor]:
        return self.get_cache("loss_dict", prefix=prefix)

    def reset_loss_dict(self) -> None:
        self.reset_cache("loss_dict")

    def get_metric_dict(self, prefix : str = "metrics") -> Dict[str, torch.Tensor]:
        return self.get_cache("metric_dict", prefix=prefix)

    def reset_metric_dict(self) -> None:
        self.reset_cache("metric_dict")


class NNTrainableModule(nn.Module, NNCacheImpl, TrainableModuleInterface):
    """NNTrainableModule is an extended nn.Module that:
        1. Adds caching functions (NNCacheImpl) 
        2. Supports training interface.
        3. Adds a _device_indicator buffer which supports self.device similar to torch.Tensor.
    """    
    def __init__(self):
        super().__init__()
        NNCacheImpl.__init__(self)
        self.register_buffer("_device_indicator", torch.zeros(1), persistent=False)
        # cache for trainer
        # self.loss_dict = dict()
        # self.metric_dict = dict()

    @property
    def device(self):
        return self._device_indicator.device

    def set_custom_state(self, state : str = None):
        """ Set custom state for this module and all its submodules

        Args:
            state (str, optional): state name. Defaults to None.
        """        
        for name, module in self.named_children():
            if isinstance(module, NNTrainableModule):
                module.set_custom_state(state)

    def get_parameters(self, *args, **kwargs) -> dict:
        return self.state_dict()

    def load_parameters(self, parameters: dict, *args, **kwargs) -> None:
        self.load_state_dict(parameters)

    def iter_trainable_parameters(self, *args, **kwargs) -> Iterator:
        return self.parameters()

    def train_full(self, dataloader, *args, **kwargs) -> None:
        for data in dataloader:
            self.train_iter(data, *args, **kwargs)

    def train_iter(self, data, *args, **kwargs) -> None:
        self.forward(data, *args, **kwargs)
        # TODO: parameter update should be handled with an inner or extra optimizer!

    def update_state(self, *args, **kwargs) -> None:
        super().update_state(*args, **kwargs)
        # basically we only set model to eval mode here (this should be handled before testing)
        # self.eval()

    def forward(self, *args, **kwargs):
        self.reset_loss_dict()
        self.reset_metric_dict()


# a self-trainable module interface
class SelfTrainableInterface(NNCacheImpl):
    def __init__(self, trainer : BasicNNTrainerEngine = None, output_dir=None, **kwargs):
        super().__init__()
        NNCacheImpl.__init__(self)
        self.trainer = trainer
        if trainer is not None:
            self.set_trainer(trainer)
            trainer.setup_engine(output_dir=output_dir)

    def set_trainer(self, trainer : BasicNNTrainerEngine, **kwargs):
        self.trainer = trainer
        self.trainer.set_model(self)
        self.trainer_config = kwargs

    def is_trainer_output_setup(self):
        return self.trainer.output_dir is not None
    
    def setup_trainer_engine(self, output_dir=None, logger=None, **kwargs):
        self.trainer.setup_engine(output_dir=output_dir, logger=logger)

    def do_train(self):
        if self.trainer is not None:
            self.trainer.initialize(**self.trainer_config)
            self.trainer.do_train()


# a self-trainable module powered by pl.LightningModule
class PLNNTrainableModule(pl.LightningModule, SelfTrainableInterface, TrainableModuleInterface):
    def __init__(self, trainer : BasicNNTrainerEngine = None, output_dir=None, **kwargs):
        super().__init__()
        SelfTrainableInterface.__init__(self, trainer, output_dir=output_dir)

    def get_parameters(self, *args, **kwargs) -> dict:
        return self.state_dict()

    def load_parameters(self, parameters: dict, *args, **kwargs) -> None:
        self.load_state_dict(parameters)

    def iter_trainable_parameters(self, *args, **kwargs) -> Iterator:
        return self.parameters()

    def train_full(self, dataloader, *args, **kwargs) -> None:
        # for data in dataloader:
        #     self.train_iter(data, *args, **kwargs)
        # if self.trainer is not None:
        #     self.trainer.initialize(*args, **kwargs, **self.trainer_config)
        self.logger.warn("PLNNTrainableModule is self trained! No need to call train() function!")

    def train_iter(self, data, *args, **kwargs) -> None:
        # self.forward(data, *args, **kwargs)
        self.logger.warn("PLNNTrainableModule is self trained! No need to call train() function!")

    # usually training is not required for self trained modules!
    def forward(self, *args, **kwargs):
        pass
