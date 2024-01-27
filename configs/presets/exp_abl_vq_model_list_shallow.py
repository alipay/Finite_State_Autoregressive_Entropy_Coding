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

    # soft-to-hard-vq
    import_class_builder_from_module(general_codec)
        .set_override_name("softtohardvq-c2")
        .update_slot_params(
            prior_model=import_class_builder_from_module(backbone_module).update_slot_params(
                prior_coder=import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.vq).update_slot_params(
                    latent_dim=2,
                    embedding_dim=32,
                    num_embeddings=256,
                    use_soft_vq=True,
                    vq_cost=0.0,
                    commitment_cost=0.0,
                    relax_temp=1.0,
                    relax_temp_anneal=True,
                )
            ),
        ),

    # vqvae-ema
    import_class_builder_from_module(general_codec)
        .set_override_name("vqvae-ema-c2")
        .update_slot_params(
            prior_model=import_class_builder_from_module(backbone_module).update_slot_params(
                prior_coder=import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.vq).update_slot_params(
                    latent_dim=2,
                    embedding_dim=32,
                    num_embeddings=256,
                    use_ema_update=True,
                )
            ),
        ),


    import_class_builder_from_module(general_codec)
        .set_override_name("V2DVQ-EM-c2")
        .update_slot_params(
            prior_model=import_class_builder_from_module(backbone_module).update_slot_params(
                prior_coder=import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.dist_ar_cat_svq).update_slot_params(
                    train_em_update=True,
                    skip_layers_if_equal_channels=True,
                    latent_channels=2,
                    in_channels=64,
                    embedding_dim=32,
                    categorical_dim=256,
                    force_hardmax=True,
                    use_sample_kl=True,
                    force_st=True,
                    st_weight=1.0,
                    entropy_temp=0.0,
                    embedding_variance_trainable=False,
                ),
                train_em_update=True,
            ),
        ),

    # # 
    # import_class_builder_from_module(general_codec)
    #     .set_override_name("V2DVQ-anneal-c2")
    #     .update_slot_params(
    #         prior_model=import_class_builder_from_module(backbone_module).update_slot_params(
    #             prior_coder=import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.dist_ar_cat_svq).update_slot_params(
    #                 skip_layers_if_equal_channels=True,
    #                 latent_channels=2,
    #                 in_channels=64,
    #                 embedding_dim=32,
    #                 categorical_dim=256,
    #                 use_sample_kl=True,
    #                 force_st=True,
    #                 st_weight=1.0,
    #                 entropy_temp=0.0,
    #                 embedding_variance_trainable=False,
    #                 relax_temp=1.0,
    #                 relax_temp_anneal=True,
    #             ),
    #         ),
    #     ),

    # import_class_builder_from_module(general_codec)
    #     .set_override_name("softtohardvq-gapcontrol-c2")
    #     .update_slot_params(
    #         prior_model=import_class_builder_from_module(backbone_module).update_slot_params(
    #             prior_coder=import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.vq).update_slot_params(
    #                 latent_dim=2,
    #                 embedding_dim=32,
    #                 num_embeddings=256,
    #                 use_soft_vq=True,
    #                 vq_cost=0.0,
    #                 commitment_cost=0.0,
    #                 # relax_temp=1.0,
    #                 # relax_temp_anneal=True,
    #             ),
    #             gap_control_loss_anneal=True,
    #         ),
    #     ),

    import_class_builder_from_module(general_codec)
        .set_override_name("softvq-gapcontrol-c2")
        .update_slot_params(
            prior_model=import_class_builder_from_module(backbone_module).update_slot_params(
                prior_coder=import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.vq).update_slot_params(
                    latent_dim=2,
                    embedding_dim=32,
                    num_embeddings=256,
                    dist_type="RelaxedOneHotCategorical",
                    commitment_cost=0.0,
                ),
                gap_control_loss_anneal=True,
            ),
        ),

    # import_class_builder_from_module(general_codec)
    #     .set_override_name("V2DVQ-gapcontrol-c2")
    #     .update_slot_params(
    #         prior_model=import_class_builder_from_module(backbone_module).update_slot_params(
    #             prior_coder=import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.dist_ar_cat_svq).update_slot_params(
    #                 skip_layers_if_equal_channels=True,
    #                 latent_channels=2,
    #                 in_channels=64,
    #                 embedding_dim=32,
    #                 categorical_dim=256,
    #                 use_sample_kl=True,
    #                 force_st=True,
    #                 st_weight=1.0,
    #                 entropy_temp=0.0,
    #                 embedding_variance_trainable=False,
    #                 # relax_temp=1.0,
    #                 # relax_temp_anneal=True,
    #             ),
    #             gap_control_loss_anneal=True,
    #         ),
    #     ),
)
