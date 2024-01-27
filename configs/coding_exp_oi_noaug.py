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
import configs.datasets.images.openimages_cache_at as dataset_training
import configs.datasets.images.openimages_val as dataset_validation
import configs.datasets.images.kodak as dataset_testing
import configs.datasets.images.clic_val as dataset_testing2

import configs.dataloaders.torch_inmem as default_dataloader

import configs.presets.exp_coding_models_shallow as default_models

gpu_available = torch.cuda.is_available()
num_gpus = torch.cuda.device_count()
batch_size_total = 32 # 64
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
    dataset=import_config_from_module(dataset_validation),
)
default_nn_validation_dataloader = import_config_from_module(default_dataloader).update_slot_params(
    dataset=default_nn_validation_dataset,
    batch_size=batch_size,
    shuffle=False,
)

default_testing_dataset = import_config_from_module(wrapper_dataset).update_slot_params(
    dataset=import_config_from_module(dataset_testing),
)
default_testing_dataloader = import_config_from_module(default_dataloader).update_slot_params(
    dataset=default_testing_dataset,
    batch_size=1,
    shuffle=False,
)

default_testing_dataset2 = import_config_from_module(wrapper_dataset).update_slot_params(
    dataset=import_config_from_module(dataset_testing2),
)
default_testing_dataloader2 = import_config_from_module(default_dataloader).update_slot_params(
    dataset=default_testing_dataset2,
    batch_size=1,
    shuffle=False,
)

config = GroupedCodecBenchmarkBuilder(
    codec_group_builder = import_config_from_module(default_models),
    benchmark_builder=
        import_config_from_module(default_benchmark)
        .set_override_name("coding-shallow-exp-oi-noaug")
        .update_slot_params(
            dataloader=default_testing_dataloader,
            extra_testing_dataloaders=ClassBuilderList(default_testing_dataloader2),
            trainer=import_config_from_module(default_trainer).update_slot_params(
                dataloader_training=default_nn_training_dataloader,
                dataloader_validation=default_nn_validation_dataloader,
                num_epoch=num_epoch,
                model_wrapper_config="adam_slow",
                param_scheduler_configs="v2d_cat_reduce_exp10",
                trainer_config="pl_gpu_clipgrad" if gpu_available else "pl_base",
            ),
            force_basic_testing=True,
            # force_testing_device="cuda",
        ) \
)
