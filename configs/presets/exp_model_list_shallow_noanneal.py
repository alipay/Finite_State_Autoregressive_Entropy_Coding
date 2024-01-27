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


import configs.codecs.general.prior_models.presets.lossless_autoencoder_bbv2_nods_normal_quant
import configs.codecs.general.prior_models.presets.lossless_autoencoder_bbv2_shallow_normal_quant
import configs.codecs.general.prior_models.presets.lossless_autoencoder_bbv2_shallow_none


import configs.codecs.general.prior_models.prior_coders.vq
import configs.codecs.general.prior_models.prior_coders.dist_gaussian
import configs.codecs.general.prior_models.prior_coders.compressai_coder
import configs.codecs.general.prior_models.prior_coders.mcquic_coder
import configs.codecs.general.prior_models.prior_coders.dist_ar_cat_svq

import configs.codecs.general.entropy_models.rans
import configs.codecs.general.entropy_models.dist_gaussian
# import configs.codecs.general.entropy_models.bbans.bbans_bbv2

backbone_module = (configs.codecs.general.prior_models.presets.lossless_autoencoder_bbv2_shallow_none)
entropy_coder_config = import_class_builder_from_module(configs.codecs.general.entropy_models.dist_gaussian).update_slot_params(
    distortion_method="cdf_delta",
)

common_config = dict(
    skip_layers_if_equal_channels=True,
    latent_channels=2,
    in_channels=64,
    embedding_dim=32,
    categorical_dim=256,
    use_st_hardmax=True,
    use_sample_kl=True,
    force_st=True,
    st_weight=1.0,
    entropy_temp=0.0,
    embedding_variance_trainable=False,
)
config = ClassBuilderList(

    # sqvae
    # import_class_builder_from_module(general_codec)
    #     .set_override_name("sqvae-c2")
    #     .update_slot_params(
    #         prior_model=import_class_builder_from_module(backbone_module).update_slot_params(
    #             prior_coder=import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.dist_ar_cat_svq).update_slot_params(
    #                 skip_layers_if_equal_channels=True,
    #                 latent_channels=2,
    #                 in_channels=64,
    #                 embedding_dim=32,
    #                 categorical_dim=256,
    #                 gs_temp=1.0,
    #                 gs_temp_anneal=True,
    #             ),
    #         ),
    #     ),

    import_class_builder_from_module(general_codec)
        .set_override_name("V2DVQ-c2-maskconv3x3")
        .update_slot_params(
            prior_model=import_class_builder_from_module(backbone_module).update_slot_params(
                prior_coder=import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.dist_ar_cat_svq).update_slot_params(
                    use_autoregressive_prior=True,
                    ar_method="maskconv3x3",
                    kl_prior_detach_posterior=True,
                    **common_config
                ),
            ),
            entropy_coder=copy.deepcopy(entropy_coder_config),
        ),


    import_class_builder_from_module(general_codec)
        .set_override_name("V2DVQ-c2-checkerboard3x3")
        .update_slot_params(
            prior_model=import_class_builder_from_module(backbone_module).update_slot_params(
                prior_coder=import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.dist_ar_cat_svq).update_slot_params(
                    use_autoregressive_prior=True,
                    ar_method="checkerboard3x3",
                    kl_prior_detach_posterior=True,
                    **common_config
                ),
            ),
            entropy_coder=copy.deepcopy(entropy_coder_config),
        ),


    # base
    import_class_builder_from_module(general_codec)
        .set_override_name("V2DVQ-c2")
        .update_slot_params(
            prior_model=import_class_builder_from_module(backbone_module).update_slot_params(
                prior_coder=import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.dist_ar_cat_svq).update_slot_params(
                    **common_config
                ),
            ),
            entropy_coder=copy.deepcopy(entropy_coder_config),
        ),

    # finite state ar
    import_class_builder_from_module(general_codec)
        .set_override_name("V2DVQ-c2-FSAR-O1S")
        .update_slot_params(
            prior_model=import_class_builder_from_module(backbone_module).update_slot_params(
                prior_coder=import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.dist_ar_cat_svq).update_slot_params(
                    use_autoregressive_prior=True,
                    ar_fs_method="MLP3",
                    ar_mlp_per_channel=True,
                    ar_offsets="l",
                    kl_prior_detach_posterior=True,
                    **common_config
                ),
            ),
            entropy_coder=copy.deepcopy(entropy_coder_config),
        ),

    import_class_builder_from_module(general_codec)
        .set_override_name("V2DVQ-c2-FSAR-O2S")
        .update_slot_params(
            prior_model=import_class_builder_from_module(backbone_module).update_slot_params(
                prior_coder=import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.dist_ar_cat_svq).update_slot_params(
                    use_autoregressive_prior=True,
                    ar_fs_method="MLP3",
                    ar_mlp_per_channel=True,
                    ar_offsets="lt",
                    kl_prior_detach_posterior=True,
                    **common_config
                ),
            ),
            entropy_coder=copy.deepcopy(entropy_coder_config),
        ),

    # import_class_builder_from_module(general_codec)
    #     .set_override_name("V2DVQ-c2-FSAR-O2S-catreduce1.4")
    #     .update_slot_params(
    #         prior_model=import_class_builder_from_module(backbone_module).update_slot_params(
    #             prior_coder=import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.dist_ar_cat_svq).update_slot_params(
    #                 use_autoregressive_prior=True,
    #                 ar_fs_method="MLP3",
    #                 ar_mlp_per_channel=True,
    #                 ar_offsets="lt",
    #                 kl_prior_detach_posterior=True,
    #                 cat_reduce=True,
    #                 cat_reduce_method="entmax",
    #                 cat_reduce_channel_same=True,
    #                 cat_reduce_entmax_alpha=1.4,
    #                 **common_config
    #             ),
    #         ),
    #         entropy_coder=copy.deepcopy(entropy_coder_config),
    #     ),

    import_class_builder_from_module(general_codec)
        .set_override_name("V2DVQ-c2-FSAR-O2S-catreduce1.5")
        .update_slot_params(
            prior_model=import_class_builder_from_module(backbone_module).update_slot_params(
                prior_coder=import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.dist_ar_cat_svq).update_slot_params(
                    use_autoregressive_prior=True,
                    ar_fs_method="MLP3",
                    ar_mlp_per_channel=True,
                    ar_offsets="lt",
                    kl_prior_detach_posterior=True,
                    cat_reduce=True,
                    cat_reduce_method="entmax",
                    cat_reduce_channel_same=True,
                    cat_reduce_entmax_alpha=1.4,
                    **common_config
                ),
            ),
            entropy_coder=copy.deepcopy(entropy_coder_config),
        ),

    import_class_builder_from_module(general_codec)
        .set_override_name("V2DVQ-c2-FSAR-O3S")
        .update_slot_params(
            prior_model=import_class_builder_from_module(backbone_module).update_slot_params(
                prior_coder=import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.dist_ar_cat_svq).update_slot_params(
                    use_autoregressive_prior=True,
                    ar_fs_method="MLP3",
                    ar_mlp_per_channel=True,
                    ar_offsets="ctx3",
                    kl_prior_detach_posterior=True,
                    **common_config
                ),
            ),
            entropy_coder=copy.deepcopy(entropy_coder_config),
        ),

)
