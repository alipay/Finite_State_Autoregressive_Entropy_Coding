import itertools
import os
import pickle
import random
from typing import Any

from cbench.utils.engine import BaseEngine
from cbench.codecs.base import CodecInterface
from cbench.data.base import DataLoaderInterface
from cbench.modules.base import TrainableModuleInterface
from cbench.utils.logging_utils import MetricLogger, SmoothedValue


class BasicTrainer(BaseEngine):
    def __init__(self, dataloader: DataLoaderInterface, *args,
                 metric_logger=None,
                 # training schedules (TODO: maybe could add another Scheduler class?)
                 use_iterative_training=False,
                 max_training_samples=-1,
                 num_epoch=1,
                 **kwargs) -> None:

        super().__init__(*args, **kwargs)

        self.dataloader = dataloader
        
        # metric logger
        self.metric_logger = MetricLogger() if metric_logger is None else metric_logger
        
        self.use_iterative_training = use_iterative_training
        self.max_training_samples = max_training_samples
        self.num_epoch = num_epoch

    def reset(self, *args, **kwargs) -> None :
        # if output_dir is not None:
        #     if not os.path.exists(output_dir):
        #         os.makedirs(output_dir)
        #     self.output_dir = output_dir
        self.metric_logger.reset()

    def train_module_one_epoch(self, module: TrainableModuleInterface, *args, **kwargs) -> None :
        if not isinstance(module, TrainableModuleInterface):
            self.logger.warn("Module not trainable! Skipping training!")
            return

        if self.use_iterative_training:
            for idx, data in enumerate(self.dataloader.iterate()):
                if self.max_training_samples > 0 and self.max_training_samples <= idx:
                    break
                module.train_iter(data)
        else:
            
            if self.max_training_samples > 0 and len(self.dataloader) > self.max_training_samples:
                # random.shuffle(training_samples)
                training_samples = [data for data in itertools.islice(self.dataloader, self.max_training_samples)]
            else:
                training_samples = list(self.dataloader)
            module.train_full(training_samples)

        module.update_state()
        # TODO: save checkpoint per epoch?

    def load_checkpoint(self, module: TrainableModuleInterface, *args, 
                    checkpoint_file="params.pkl",
                    **kwargs) -> Any:
        checkpoint_path = os.path.join(self.output_dir, checkpoint_file)
        # load checkpoint (TODO: resume training)
        if os.path.exists(checkpoint_path):
            self.logger.info("Loading checkpoint from {} ...".format(checkpoint_path))
            with open(checkpoint_path, 'rb') as f:
                params = pickle.load(f)
                module.load_parameters(params)

    def train_module(self, module: TrainableModuleInterface, *args, 
                     load_checkpoint=True,
                     save_checkpoint=True,
                     checkpoint_file="params.pkl",
                     **kwargs) -> None :
        if not isinstance(module, TrainableModuleInterface):
            self.logger.warn("Module not trainable! Skipping training!")
            return

        self.metric_logger.reset()

        # training step
        checkpoint_path = os.path.join(self.output_dir, checkpoint_file)
        # load checkpoint (TODO: resume training)
        if load_checkpoint and os.path.exists(checkpoint_path):
            self.logger.info("Loading checkpoint from {} ...".format(checkpoint_path))
            with open(checkpoint_path, 'rb') as f:
                params = pickle.load(f)
                module.load_parameters(params)
        else:
            for epoch_idx in range(self.num_epoch):
                self.logger.info("Training Epoch {}".format(epoch_idx))
                self.train_module_one_epoch(module, *args, **kwargs)

        # save checkpoint
        if save_checkpoint:
            self.logger.info("Saving checkpoint to {} ...".format(checkpoint_path))
            params = module.get_parameters()
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(params, f)

    # some training modules could implement its own testing stream such as gpu-based nn trainer    
    def test_module(self, module: TrainableModuleInterface, *args, 
                    checkpoint_file="params.pkl",
                    **kwargs) -> Any:
        raise NotImplementedError()
    