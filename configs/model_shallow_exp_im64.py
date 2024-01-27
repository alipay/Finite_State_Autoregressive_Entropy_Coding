import torch
import os

from configs.import_utils import import_config_from_module, import_class_builder_from_module
from configs.class_builder import ClassBuilder, ClassBuilderList, NamedParam
from configs.oss_utils import OSSUtils
from configs.utils.group_benchmark_builder import GroupedCodecBenchmarkBuilder
from configs.env import DEFAULT_PRETRAINED_PATH

import configs.benchmark.lossless_compression_trainable as default_benchmark

import configs.trainer.nn_trainer as default_trainer

import configs.datasets.images.image_dataset_wrapper as wrapper_dataset
import configs.datasets.images.imagenet64_train as dataset_training
import configs.datasets.images.imagenet64_val as dataset_testing

import configs.dataloaders.torch as default_dataloader

import configs.presets.exp_model_list_shallow as default_models
import configs.presets.exp_model_list_shallow_noanneal as default_models_noanneal

gpu_available = torch.cuda.is_available()
num_gpus = torch.cuda.device_count()
batch_size_total = 256
batch_size_gpu = batch_size_total // num_gpus if num_gpus > 0 else batch_size_total
batch_size_cpu = 1
batch_size = batch_size_gpu if gpu_available else batch_size_cpu

num_epoch = 100 if gpu_available else 1

default_nn_training_dataset = import_config_from_module(wrapper_dataset).update_slot_params(
    dataset=import_config_from_module(dataset_training),
)
default_nn_training_dataloader = import_config_from_module(default_dataloader).update_slot_params(
    dataset=default_nn_training_dataset,
    batch_size=batch_size,
)

default_nn_validation_dataset = import_config_from_module(wrapper_dataset).update_slot_params(
    dataset=import_config_from_module(dataset_testing),
)
default_nn_validation_dataloader = import_config_from_module(default_dataloader).update_slot_params(
    dataset=default_nn_validation_dataset,
    batch_size=batch_size,
    shuffle=False,
)

config = GroupedCodecBenchmarkBuilder(
    codec_group_builder = (
        import_config_from_module(default_models) + \
        import_config_from_module(default_models_noanneal)
    ),
    benchmark_builder=
        import_config_from_module(default_benchmark)
        .set_override_name("model-shallow-noanneal-im64-exp")
        .update_slot_params(
            dataloader=default_nn_validation_dataloader,
            trainer=import_config_from_module(default_trainer).update_slot_params(
                dataloader_training=default_nn_training_dataloader,
                dataloader_validation=default_nn_validation_dataloader,
                num_epoch=num_epoch,
                model_wrapper_config="adam_slow",
                param_scheduler_configs="v2d_cat_reduce_exp10",
                trainer_config="pl_gpu_clipgrad" if gpu_available else "pl_base",
            ),
            force_basic_testing=True,
            force_testing_device="cuda",
            skip_decompress=True,
        ) \
)
