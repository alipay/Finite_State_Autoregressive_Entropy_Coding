from configs.import_utils import import_config_from_module, import_class_builder_from_module
from configs.class_builder import ClassBuilder, ClassBuilderList, NamedParam
from configs.oss_utils import OSSUtils

import configs.benchmark.lossless_compression_trainable as default_benchmark

import configs.codecs.general.base as general_codec

import configs.codecs.general.prior_models.base_lossless_autoencoder
import configs.codecs.general.prior_models.lossless_autoencoder_bbv2
import configs.codecs.general.prior_models.base_lossy_autoencoder
import configs.codecs.general.prior_models.aev2_vqvae_v2backbone


import configs.codecs.general.prior_models.presets.lossless_autoencoder_bbv2_nods_normal_quant
import configs.codecs.general.prior_models.presets.lossless_autoencoder_bbv2_shallow_normal_quant


import configs.codecs.general.prior_models.prior_coders.vq
import configs.codecs.general.prior_models.prior_coders.dist_gaussian
import configs.codecs.general.prior_models.prior_coders.compressai_coder
import configs.codecs.general.prior_models.prior_coders.mcquic_coder
import configs.codecs.general.prior_models.prior_coders.dist_ar_cat_svq

import configs.codecs.general.entropy_models.rans
# import configs.codecs.general.entropy_models.bbans.bbans_bbv2

backbone_module = (configs.codecs.general.prior_models.presets.lossless_autoencoder_bbv2_shallow_normal_quant)

config = ClassBuilderList(

    # vqvae
    import_class_builder_from_module(general_codec)
        .set_override_name("vqvae-c2")
        .update_slot_params(
            prior_model=import_class_builder_from_module(backbone_module).update_slot_params(
                prior_coder=import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.vq).update_slot_params(
                    latent_dim=2,
                    embedding_dim=32,
                    num_embeddings=256,
                )
            ),
        ),

    # soft-vqvae
    import_class_builder_from_module(general_codec)
        .set_override_name("softvqvae-c2")
        .update_slot_params(
            prior_model=import_class_builder_from_module(backbone_module).update_slot_params(
                prior_coder=import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.vq).update_slot_params(
                    latent_dim=2,
                    embedding_dim=32,
                    num_embeddings=256,
                    dist_type="RelaxedOneHotCategorical",
                    commitment_cost=0.0,
                )
            ),
        ),

    # sqvae
    import_class_builder_from_module(general_codec)
        .set_override_name("sqvae-c2")
        .update_slot_params(
            prior_model=import_class_builder_from_module(backbone_module).update_slot_params(
                prior_coder=import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.dist_ar_cat_svq).update_slot_params(
                    skip_layers_if_equal_channels=True,
                    latent_channels=2,
                    in_channels=64,
                    embedding_dim=32,
                    categorical_dim=256,
                    gs_temp=1.0,
                    gs_temp_anneal=True,
                ),
            ),
        ),

    # base
    # import_class_builder_from_module(general_codec)
    #     .set_override_name("V2DVQ-c2")
    #     .update_slot_params(
    #         prior_model=import_class_builder_from_module(backbone_module).update_slot_params(
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
    #                 entropy_temp_anneal=True,
    #                 embedding_variance_trainable=False,
    #             ),
    #         ),
    #     ),

    # c2 models
    import_class_builder_from_module(general_codec)
        .set_override_name("V2DVQ-c2-nosth")
        .update_slot_params(
            prior_model=import_class_builder_from_module(backbone_module).update_slot_params(
                prior_coder=import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.dist_ar_cat_svq).update_slot_params(
                    skip_layers_if_equal_channels=True,
                    latent_channels=2,
                    in_channels=64,
                    embedding_dim=32,
                    categorical_dim=256,
                    force_st=True,
                    st_weight=1.0,
                    entropy_temp=0.0,
                    embedding_variance_trainable=False,
                ),
            ),
        ),

    import_class_builder_from_module(general_codec)
        .set_override_name("V2DVQ-c2-noreg")
        .update_slot_params(
            prior_model=import_class_builder_from_module(backbone_module).update_slot_params(
                prior_coder=import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.dist_ar_cat_svq).update_slot_params(
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
                    cont_loss_weight=0.0,
                ),
            ),
        ),

    import_class_builder_from_module(general_codec)
        .set_override_name("V2DVQ-c2-noanneal")
        .update_slot_params(
            prior_model=import_class_builder_from_module(backbone_module).update_slot_params(
                prior_coder=import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.dist_ar_cat_svq).update_slot_params(
                    skip_layers_if_equal_channels=True,
                    latent_channels=2,
                    in_channels=64,
                    embedding_dim=32,
                    categorical_dim=256,
                    use_st_hardmax=True,
                    use_sample_kl=True,
                    force_st=True,
                    st_weight=1.0,
                    entropy_temp=1.0,
                    embedding_variance_trainable=False,
                ),
            ),
        ),

    # import_class_builder_from_module(general_codec)
    #     .set_override_name("V2DVQ-c2-noent")
    #     .update_slot_params(
    #         prior_model=import_class_builder_from_module(backbone_module).update_slot_params(
    #             prior_coder=import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.dist_ar_cat_svq).update_slot_params(
    #                 skip_layers_if_equal_channels=True,
    #                 latent_channels=2,
    #                 in_channels=64,
    #                 embedding_dim=32,
    #                 categorical_dim=256,
    #                 use_st_hardmax=True,
    #                 force_st=True,
    #                 st_weight=1.0,
    #                 entropy_temp=0.0,
    #                 embedding_variance_trainable=False,
    #             ),
    #         ),
    #     ),

    # import_class_builder_from_module(general_codec)
    #     .set_override_name("V2DVQ-c2-nost")
    #     .update_slot_params(
    #         prior_model=import_class_builder_from_module(backbone_module).update_slot_params(
    #             prior_coder=import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.dist_ar_cat_svq).update_slot_params(
    #                 skip_layers_if_equal_channels=True,
    #                 latent_channels=2,
    #                 in_channels=64,
    #                 embedding_dim=32,
    #                 categorical_dim=256,
    #                 use_st_hardmax=True,
    #                 entropy_temp_anneal=True,
    #                 embedding_variance_trainable=False,
    #             ),
    #         ),
    #     ),

)
