from configs.import_utils import import_config_from_module, import_class_builder_from_module
from configs.class_builder import ClassBuilder, ClassBuilderList, NamedParam
from configs.oss_utils import OSSUtils
import copy

import configs.benchmark.lossless_compression_trainable as default_benchmark

import configs.codecs.general.base as general_codec

import configs.codecs.general.prior_models.base_lossless_autoencoder
import configs.codecs.general.prior_models.lossless_autoencoder_bbv2
import configs.codecs.general.prior_models.base_lossy_autoencoder
import configs.codecs.general.prior_models.aev2_vqvae_v2backbone



import configs.codecs.general.prior_models.prior_coders.vq
import configs.codecs.general.prior_models.prior_coders.dist_gaussian
import configs.codecs.general.prior_models.prior_coders.compressai_coder
import configs.codecs.general.prior_models.prior_coders.mcquic_coder
import configs.codecs.general.prior_models.prior_coders.dist_ar_cat_svq

import configs.codecs.general.prior_models.presets.lossless_autoencoder_bbv2_shallow_none
import configs.codecs.general.prior_models.presets.lossless_autoencoder_bbv2_shallow_c32_none
import configs.codecs.general.prior_models.presets.lossless_autoencoder_bbv2_shallow_c32_r4_none

import configs.codecs.general.entropy_models.rans
import configs.codecs.general.entropy_models.dist_gaussian
# import configs.codecs.general.entropy_models.bbans.bbans_bbv2

import configs.codecs.general.preprocessors.twar

backbone_module = (configs.codecs.general.prior_models.presets.lossless_autoencoder_bbv2_shallow_none)
coder_config = (

    import_class_builder_from_module(general_codec)
        .set_override_name("Coder-ar-vq")
        .update_slot_params(
            prior_model=import_config_from_module(backbone_module).update_slot_params(
                prior_coder=import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.vq).update_slot_params(
                    latent_dim=2,
                    embedding_dim=32,
                    num_embeddings=256,
                ),
            ),
            entropy_coder=import_class_builder_from_module(configs.codecs.general.entropy_models.dist_gaussian).update_slot_params(
                use_autoregressive_prior=True,
                ar_method="linear",
                ar_offsets="twar",
                ar_offsets_per_channel=True,
                ar_output_as_mean_offset=True,
                ar_default_sample=0.0,
                sigmoid_mean=True,
                mean_step=1./1023,
                distortion_method="cdf_delta",
            ),
        )
)


backbone_c32_module = (configs.codecs.general.prior_models.presets.lossless_autoencoder_bbv2_shallow_c32_none)
coder_c32_config = (
    import_class_builder_from_module(general_codec)
        .set_override_name("Coder-ar-c32-vq")
        .update_slot_params(
            prior_model=import_config_from_module(backbone_c32_module).update_slot_params(
                prior_coder=import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.vq).update_slot_params(
                    latent_dim=1,
                    embedding_dim=32,
                    num_embeddings=256,
                ),
            ),
            entropy_coder=import_class_builder_from_module(configs.codecs.general.entropy_models.dist_gaussian).update_slot_params(
                use_autoregressive_prior=True,
                ar_method="linear",
                ar_offsets="twar",
                ar_offsets_per_channel=True,
                ar_output_as_mean_offset=True,
                ar_default_sample=0.0,
                sigmoid_mean=True,
                mean_step=1./1023,
                distortion_method="cdf_delta",
            ),
        )
)

backbone_c32_r4_module = (configs.codecs.general.prior_models.presets.lossless_autoencoder_bbv2_shallow_c32_r4_none)
coder_c32_r4_config = (
    import_class_builder_from_module(general_codec)
        .set_override_name("Coder-ar-c32-r4-vq")
        .update_slot_params(
            prior_model=import_config_from_module(backbone_c32_r4_module).update_slot_params(
                prior_coder=import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.vq).update_slot_params(
                    latent_dim=1,
                    embedding_dim=32,
                    num_embeddings=256,
                ),
            ),
            entropy_coder=import_class_builder_from_module(configs.codecs.general.entropy_models.dist_gaussian).update_slot_params(
                use_autoregressive_prior=True,
                ar_method="linear",
                ar_offsets="twar",
                ar_offsets_per_channel=True,
                ar_output_as_mean_offset=True,
                ar_default_sample=0.0,
                sigmoid_mean=True,
                mean_step=1./1023,
                distortion_method="cdf_delta",
            ),
        )
)


config = ClassBuilderList(

    # import_class_builder_from_module(general_codec)
    #     .set_override_name("bits_back")
    #     .update_slot_params(
    #         entropy_coder=import_class_builder_from_module(configs.codecs.general.entropy_models.bbans.bbans_bbv2).update_slot_params(
    #             hidden_channels=32,
    #             num_downsample_layers=1,
    #             num_residual_layers=1,
    #             fixed_batch_size=1
    #         )
    #     ),


    copy.deepcopy(coder_c32_config),

    copy.deepcopy(coder_c32_config)
        .set_override_name("Coder-ar-c32-STHQ")
        .update_slot_params(
            prior_model=import_config_from_module(backbone_c32_module).update_slot_params(
                prior_coder=import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.dist_ar_cat_svq).update_slot_params(
                    skip_layers_if_equal_channels=True,
                    latent_channels=1,
                    in_channels=32,
                    embedding_dim=32,
                    categorical_dim=256,
                    use_st_hardmax=True,
                    use_sample_kl=True,
                    force_st=True,
                    st_weight=1.0,
                    entropy_temp=0.0,
                    embedding_variance_trainable=False,
                ),
            ),
        ),

    copy.deepcopy(coder_c32_config)
        .set_override_name("Coder-ar-c32-STHQ-FSAR-O2S-arid")
        .update_slot_params(
            prior_model=import_config_from_module(backbone_c32_module).update_slot_params(
                prior_coder=import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.dist_ar_cat_svq).update_slot_params(
                    skip_layers_if_equal_channels=True,
                    latent_channels=1,
                    in_channels=32,
                    embedding_dim=32,
                    categorical_dim=256,
                    use_st_hardmax=True,
                    use_sample_kl=True,
                    force_st=True,
                    st_weight=1.0,
                    entropy_temp=0.0,
                    embedding_variance_trainable=False,
                    use_autoregressive_prior=True,
                    ar_fs_method="MLP3",
                    ar_mlp_per_channel=True,
                    ar_offsets="lt",
                    ar_input_detach=True,
                    kl_prior_detach_posterior=True,
                ),
            ),
        ),

    # copy.deepcopy(coder_c32_r4_config),

    # copy.deepcopy(coder_c32_r4_config)
    #     .set_override_name("Coder-ar-c32-r4-STHQ")
    #     .update_slot_params(
    #         prior_model=import_config_from_module(backbone_c32_r4_module).update_slot_params(
    #             prior_coder=import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.dist_ar_cat_svq).update_slot_params(
    #                 skip_layers_if_equal_channels=True,
    #                 latent_channels=1,
    #                 in_channels=32,
    #                 embedding_dim=32,
    #                 categorical_dim=256,
    #                 use_st_hardmax=True,
    #                 use_sample_kl=True,
    #                 force_st=True,
    #                 st_weight=1.0,
    #                 entropy_temp=0.0,
    #                 embedding_variance_trainable=False,
    #             ),
    #         ),
    #     ),

    # copy.deepcopy(coder_c32_r4_config)
    #     .set_override_name("Coder-ar-c32-r4-STHQ-FSAR-O2S-arid")
    #     .update_slot_params(
    #         prior_model=import_config_from_module(backbone_c32_r4_module).update_slot_params(
    #             prior_coder=import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.dist_ar_cat_svq).update_slot_params(
    #                 skip_layers_if_equal_channels=True,
    #                 latent_channels=1,
    #                 in_channels=32,
    #                 embedding_dim=32,
    #                 categorical_dim=256,
    #                 use_st_hardmax=True,
    #                 use_sample_kl=True,
    #                 force_st=True,
    #                 st_weight=1.0,
    #                 entropy_temp=0.0,
    #                 embedding_variance_trainable=False,
    #                 use_autoregressive_prior=True,
    #                 ar_fs_method="MLP3",
    #                 ar_mlp_per_channel=True,
    #                 ar_offsets="lt",
    #                 ar_input_detach=True,
    #                 kl_prior_detach_posterior=True,
    #             ),
    #         ),
    #     ),

    # import_class_builder_from_module(general_codec)
    #     .set_override_name("Coder-c32-l2-ar-lin-twar-sigmoid")
    #     .update_slot_params(
    #         prior_model=import_config_from_module(backbone_c32_module).update_slot_params(
    #             prior_coder=import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.vq).update_slot_params(
    #                 latent_dim=2,
    #                 embedding_dim=16,
    #                 num_embeddings=256,
    #             ),
    #         ),
    #         entropy_coder=import_class_builder_from_module(configs.codecs.general.entropy_models.dist_gaussian).update_slot_params(
    #             use_autoregressive_prior=True,
    #             ar_method="linear",
    #             ar_offsets="twar",
    #             ar_offsets_per_channel=True,
    #             ar_output_as_mean_offset=True,
    #             ar_default_sample=0.0,
    #             sigmoid_mean=True,
    #             distortion_method="cdf_delta",
    #         ),
    #     ),

    # import_class_builder_from_module(general_codec)
    #     .set_override_name("Coder-c32-l2-ar-lin-twar-sigmoid-STHQ")
    #     .update_slot_params(
    #         prior_model=import_config_from_module(backbone_c32_module).update_slot_params(
    #             prior_coder=import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.dist_ar_cat_svq).update_slot_params(
    #                 skip_layers_if_equal_channels=True,
    #                 latent_channels=2,
    #                 in_channels=32,
    #                 embedding_dim=16,
    #                 categorical_dim=256,
    #                 use_st_hardmax=True,
    #                 use_sample_kl=True,
    #                 force_st=True,
    #                 st_weight=1.0,
    #                 entropy_temp=0.0,
    #                 embedding_variance_trainable=False,
    #             ),
    #         ),
    #         entropy_coder=import_class_builder_from_module(configs.codecs.general.entropy_models.dist_gaussian).update_slot_params(
    #             use_autoregressive_prior=True,
    #             ar_method="linear",
    #             ar_offsets="twar",
    #             ar_offsets_per_channel=True,
    #             ar_output_as_mean_offset=True,
    #             ar_default_sample=0.0,
    #             sigmoid_mean=True,
    #             distortion_method="cdf_delta",
    #         ),
    #     ),

    # import_class_builder_from_module(general_codec)
    #     .set_override_name("Coder-c32-l2-ar-lin-twar-sigmoid-STHQ-FSAR-O2S-arid")
    #     .update_slot_params(
    #         prior_model=import_config_from_module(backbone_c32_module).update_slot_params(
    #             prior_coder=import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.dist_ar_cat_svq).update_slot_params(
    #                 skip_layers_if_equal_channels=True,
    #                 latent_channels=2,
    #                 in_channels=32,
    #                 embedding_dim=16,
    #                 categorical_dim=256,
    #                 use_st_hardmax=True,
    #                 use_sample_kl=True,
    #                 force_st=True,
    #                 st_weight=1.0,
    #                 entropy_temp=0.0,
    #                 embedding_variance_trainable=False,
    #                 use_autoregressive_prior=True,
    #                 ar_fs_method="MLP3",
    #                 ar_mlp_per_channel=True,
    #                 ar_offsets="lt",
    #                 ar_input_detach=True,
    #                 kl_prior_detach_posterior=True,
    #             ),
    #         ),
    #         entropy_coder=import_class_builder_from_module(configs.codecs.general.entropy_models.dist_gaussian).update_slot_params(
    #             use_autoregressive_prior=True,
    #             ar_method="linear",
    #             ar_offsets="twar",
    #             ar_offsets_per_channel=True,
    #             ar_output_as_mean_offset=True,
    #             ar_default_sample=0.0,
    #             sigmoid_mean=True,
    #             distortion_method="cdf_delta",
    #         ),
    #     ),

    # import_class_builder_from_module(general_codec)
    #     .set_override_name("Coder-ar-lin-twar-sigmoid")
    #     .update_slot_params(
    #         prior_model=import_config_from_module(backbone_module).update_slot_params(
    #             prior_coder=import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.vq).update_slot_params(
    #                 latent_dim=2,
    #                 embedding_dim=32,
    #                 num_embeddings=256,
    #             ),
    #         ),
    #         entropy_coder=import_class_builder_from_module(configs.codecs.general.entropy_models.dist_gaussian).update_slot_params(
    #             use_autoregressive_prior=True,
    #             ar_method="linear",
    #             ar_offsets="twar",
    #             ar_offsets_per_channel=True,
    #             ar_output_as_mean_offset=True,
    #             ar_default_sample=0.0,
    #             sigmoid_mean=True,
    #             distortion_method="cdf_delta",
    #         ),
    #     ),

    # import_class_builder_from_module(general_codec)
    #     .set_override_name("Coder-ar-lin-twar-sigmoid-STHQ")
    #     .update_slot_params(
    #         prior_model=import_config_from_module(backbone_module).update_slot_params(
    #             prior_coder=import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.dist_ar_cat_svq).update_slot_params(
    #                 skip_layers_if_equal_channels=True,
    #                 latent_channels=2,
    #                 in_channels=64,
    #                 embedding_dim=32,
    #                 categorical_dim=256,
    #                 use_st_hardmax=True,
    #                 use_sample_kl=True,
    #                 force_st=True,
    #                 st_weight=1.0,
    #                 entropy_temp=0.0,
    #                 embedding_variance_trainable=False,
    #             ),
    #         ),
    #         entropy_coder=import_class_builder_from_module(configs.codecs.general.entropy_models.dist_gaussian).update_slot_params(
    #             use_autoregressive_prior=True,
    #             ar_method="linear",
    #             ar_offsets="twar",
    #             ar_offsets_per_channel=True,
    #             ar_output_as_mean_offset=True,
    #             ar_default_sample=0.0,
    #             sigmoid_mean=True,
    #             distortion_method="cdf_delta",
    #         ),
    #     ),

    # import_class_builder_from_module(general_codec)
    #     .set_override_name("Coder-ar-lin-twar-sigmoid-STHQ-FSAR-O2S-arid")
    #     .update_slot_params(
    #         prior_model=import_config_from_module(backbone_module).update_slot_params(
    #             prior_coder=import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.dist_ar_cat_svq).update_slot_params(
    #                 skip_layers_if_equal_channels=True,
    #                 latent_channels=2,
    #                 in_channels=64,
    #                 embedding_dim=32,
    #                 categorical_dim=256,
    #                 use_st_hardmax=True,
    #                 use_sample_kl=True,
    #                 force_st=True,
    #                 st_weight=1.0,
    #                 entropy_temp=0.0,
    #                 embedding_variance_trainable=False,
    #                 use_autoregressive_prior=True,
    #                 ar_fs_method="MLP3",
    #                 ar_mlp_per_channel=True,
    #                 ar_offsets="lt",
    #                 ar_input_detach=True,
    #                 kl_prior_detach_posterior=True,
    #             ),
    #         ),
    #         entropy_coder=import_class_builder_from_module(configs.codecs.general.entropy_models.dist_gaussian).update_slot_params(
    #             use_autoregressive_prior=True,
    #             ar_method="linear",
    #             ar_offsets="twar",
    #             ar_offsets_per_channel=True,
    #             ar_output_as_mean_offset=True,
    #             ar_default_sample=0.0,
    #             sigmoid_mean=True,
    #             distortion_method="cdf_delta",
    #         ),
    #     ),
)
