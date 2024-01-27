import math
import torch
import os

from configs.import_utils import import_config_from_module, import_class_builder_from_module
from configs.class_builder import ClassBuilder, ClassBuilderList, NamedParam
from configs.oss_utils import OSSUtils
from configs.utils.group_benchmark_builder import GroupedCodecBenchmarkBuilder
from configs.env import DEFAULT_PRETRAINED_PATH

import configs.benchmark.lossless_compression_trainable as default_benchmark

import configs.codecs.general.base as general_codec
# import configs.codecs.general.prior_models.vqvae_sp as prior_model
# default_codec_config = import_config_from_module(general_codec).update_slot_params(
#     prior_model=import_config_from_module(prior_model).update_slot_params(
#         single_decoder=True
#     ),
# )
import configs.codecs.general.prior_models.base_lossless_autoencoder
import configs.codecs.general.prior_models.lossless_autoencoder_bbv2
import configs.codecs.general.prior_models.base_lossy_autoencoder
import configs.codecs.general.prior_models.aev2_vqvae_v2backbone



import configs.codecs.general.prior_models.prior_coders.vq
import configs.codecs.general.prior_models.prior_coders.dist_gaussian

import configs.codecs.general.entropy_models.rans
import configs.codecs.general.entropy_models.dist_gaussian
import configs.codecs.general.entropy_models.ar

import configs.codecs.general.preprocessors.twar

import configs.trainer.nn_trainer as default_trainer
import configs.nnmodules.lightning_trainer as default_nn_trainer

import configs.nnmodules.pretrained.vqvae
import configs.nnmodules.pretrained.vqvae_gssoft

import configs.datasets.images.image_dataset_wrapper as wrapper_dataset
import configs.datasets.torchvision.cifar10 as dataset_training
import configs.datasets.torchvision.cifar10_test as dataset_testing
# import configs.datasets.images.imagenet64_train as dataset_training
# import configs.datasets.images.imagenet64_val as dataset_testing
# import configs.datasets.images.clic as dataset_training
# import configs.datasets.images.clic_test as dataset_testing
# import configs.datasets.images.random_image_generator as dataset_training
# import configs.datasets.images.random_image_generator as dataset_testing

import configs.dataloaders.torch_inmem as default_dataloader

gpu_available = torch.cuda.is_available()
num_gpus = torch.cuda.device_count()
batch_size_total = 1024
batch_size_gpu = batch_size_total // num_gpus if num_gpus > 0 else batch_size_total
batch_size_cpu = 4
batch_size = batch_size_gpu if gpu_available else batch_size_cpu

num_epoch = 500 if gpu_available else 1

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
    codec_group_builder=ClassBuilderList(
        # import_class_builder_from_module(general_codec)
        # .set_override_name('FSAR-O1-MLP3')
        # .update_slot_params(
        #     entropy_coder=import_class_builder_from_module(configs.codecs.general.entropy_models.ar).update_slot_params(
        #         prior_trainable=True,
        #         use_autoregressive_prior=True,
        #         ar_fs_method="MLP3",
        #         # ar_mlp_per_channel=True,
        #     )
        # ),

        # import_class_builder_from_module(general_codec)
        # .set_override_name('FSAR-O1-MLP3-per-channel')
        # .update_slot_params(
        #     entropy_coder=import_class_builder_from_module(configs.codecs.general.entropy_models.ar).update_slot_params(
        #         prior_trainable=True,
        #         use_autoregressive_prior=True,
        #         ar_fs_method="MLP3",
        #         ar_mlp_per_channel=True,
        #     )
        # ),

        # import_class_builder_from_module(general_codec)
        # .set_override_name('FSAR-O2C-MLP3-per-channel')
        # .update_slot_params(
        #     entropy_coder=import_class_builder_from_module(configs.codecs.general.entropy_models.ar).update_slot_params(
        #         prior_trainable=True,
        #         use_autoregressive_prior=True,
        #         ar_fs_method="MLP3",
        #         ar_mlp_per_channel=True,
        #         ar_offsets='c2',
        #     )
        # ),

        import_class_builder_from_module(general_codec)
        .set_override_name('FSAR-O1S-MLP3-per-channel')
        .update_slot_params(
            entropy_coder=import_class_builder_from_module(configs.codecs.general.entropy_models.ar).update_slot_params(
                prior_trainable=True,
                use_autoregressive_prior=True,
                ar_fs_method="MLP3",
                ar_mlp_per_channel=True,
                ar_offsets='l',
            )
        ),

        import_class_builder_from_module(general_codec)
        .set_override_name('FSAR-O2S-MLP3-per-channel')
        .update_slot_params(
            entropy_coder=import_class_builder_from_module(configs.codecs.general.entropy_models.ar).update_slot_params(
                prior_trainable=True,
                use_autoregressive_prior=True,
                ar_fs_method="MLP3",
                ar_mlp_per_channel=True,
                ar_offsets='lt',
            )
        ),

        import_class_builder_from_module(general_codec)
        .set_override_name('FSAR-O3S-MLP3-per-channel')
        .update_slot_params(
            entropy_coder=import_class_builder_from_module(configs.codecs.general.entropy_models.ar).update_slot_params(
                prior_trainable=True,
                use_autoregressive_prior=True,
                ar_fs_method="MLP3",
                ar_mlp_per_channel=True,
                ar_offsets='clt',
            )
        ),

        # import_class_builder_from_module(general_codec)
        # .set_override_name('FSAR-O2SC-MLP3-per-channel')
        # .update_slot_params(
        #     entropy_coder=import_class_builder_from_module(configs.codecs.general.entropy_models.ar).update_slot_params(
        #         prior_trainable=True,
        #         use_autoregressive_prior=True,
        #         ar_fs_method="MLP3",
        #         ar_mlp_per_channel=True,
        #         ar_offsets='cl',
        #     )
        # ),

        # import_class_builder_from_module(general_codec)
        # .set_override_name('FSAR-O4C-MLP3-per-channel')
        # .update_slot_params(
        #     entropy_coder=import_class_builder_from_module(configs.codecs.general.entropy_models.ar).update_slot_params(
        #         prior_trainable=True,
        #         use_autoregressive_prior=True,
        #         ar_fs_method="MLP3",
        #         ar_mlp_per_channel=True,
        #         ar_window_size=4,
        #     )
        # ),

        import_class_builder_from_module(general_codec)
        .set_override_name('maskconv3x3')
        .update_slot_params(
            entropy_coder=import_class_builder_from_module(configs.codecs.general.entropy_models.ar).update_slot_params(
                prior_trainable=True,
                use_autoregressive_prior=True,
                ar_method="maskconv3x3",
            )
        ),

        import_class_builder_from_module(general_codec)
        .set_override_name('maskconv5x5')
        .update_slot_params(
            entropy_coder=import_class_builder_from_module(configs.codecs.general.entropy_models.ar).update_slot_params(
                prior_trainable=True,
                use_autoregressive_prior=True,
                ar_method="maskconv5x5",
            )
        ),

        # import_class_builder_from_module(general_codec)
        # .set_override_name('maskconv3d3x3x3')
        # .update_slot_params(
        #     entropy_coder=import_class_builder_from_module(configs.codecs.general.entropy_models.ar).update_slot_params(
        #         prior_trainable=True,
        #         use_autoregressive_prior=True,
        #         ar_method="maskconv3d3x3x3",
        #     )
        # ),

        # import_class_builder_from_module(general_codec)
        # .set_override_name('maskconv3d5x5x5')
        # .update_slot_params(
        #     entropy_coder=import_class_builder_from_module(configs.codecs.general.entropy_models.ar).update_slot_params(
        #         prior_trainable=True,
        #         use_autoregressive_prior=True,
        #         ar_method="maskconv3d5x5x5",
        #     )
        # ),

        # import_class_builder_from_module(general_codec)
        # .set_override_name('checkerboard3x3')
        # .update_slot_params(
        #     entropy_coder=import_class_builder_from_module(configs.codecs.general.entropy_models.ar).update_slot_params(
        #         prior_trainable=True,
        #         use_autoregressive_prior=True,
        #         ar_method="checkerboard3x3",
        #     )
        # ),

        # import_class_builder_from_module(general_codec)
        # .set_override_name('checkerboard5x5')
        # .update_slot_params(
        #     entropy_coder=import_class_builder_from_module(configs.codecs.general.entropy_models.ar).update_slot_params(
        #         prior_trainable=True,
        #         use_autoregressive_prior=True,
        #         ar_method="checkerboard5x5",
        #     )
        # ),
    ),
    benchmark_builder=
        import_config_from_module(default_benchmark)
        .set_override_name("dsar-exp")
        # .set_override_name("v2d_cat_reduce_exp200")
        .update_slot_params(
            # codec_group=default_codec_config,
            # .batch_update_slot_params(**{
            #     # "entropy_coder.0.0.num_predcnts": ClassBuilder.SLOT_ALL_CHOICES
            #     "entropy_coder.num_predcnts": [1, 2, 4, 8, 16, 32, 64, 128]
            # }),
            dataloader=default_nn_validation_dataloader,
            # training_dataloader=import_config_from_module(default_dataloader).update_slot_params(
            #     dataset=import_config_from_module(wrapper_dataset).update_slot_params(
            #         dataset=import_config_from_module(dataset_training),
            #     ),
            #     # batch_size=256,
            # ),
            trainer=import_config_from_module(default_trainer).update_slot_params(
                dataloader_training=default_nn_training_dataloader,
                dataloader_validation=default_nn_validation_dataloader,
                num_epoch=num_epoch,
                model_wrapper_config="adam",
                check_val_every_n_epoch=10,
                # param_scheduler_configs="v2d_cat_reduce_ep2000_step20",
                # param_scheduler_configs="v2d_cat_reduce_exp200",
                trainer_config="pl_gpu_clipgrad" if gpu_available else "pl_base",
            ),
            force_basic_testing=True,
            force_testing_device="cuda",
        ) \
        # .batch_update_slot_params(**{
        #     "codec.entropy_coder.0.0.num_predcnts": ClassBuilder.SLOT_ALL_CHOICES
        # })
)
# print(list(config.iter_parameters()))