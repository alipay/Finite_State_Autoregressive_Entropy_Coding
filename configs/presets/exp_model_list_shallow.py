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
import configs.codecs.general.entropy_models.bbans.bbans_bbv2

backbone_module = (configs.codecs.general.prior_models.presets.lossless_autoencoder_bbv2_shallow_none)
entropy_coder_config = import_class_builder_from_module(configs.codecs.general.entropy_models.dist_gaussian).update_slot_params(
    distortion_method="cdf_delta",
)

config = ClassBuilderList(
    ### Variational ###

    # bits-back
    import_class_builder_from_module(general_codec)
        .set_override_name("bits_back")
        .update_slot_params(
            entropy_coder=import_class_builder_from_module(configs.codecs.general.entropy_models.bbans.bbans_bbv2).update_slot_params(
                out_channels=6,
                obs_codec_type="gaussian",
                hidden_channels=64,
                num_downsample_layers=1,
                num_residual_layers=1,
                fixed_batch_size=1,
            )
        ),

    ### Deterministic ###

    # entropy bottleneck
    import_class_builder_from_module(general_codec)
        .set_override_name("entropy_bottleneck")
        .update_slot_params(
            prior_model=import_class_builder_from_module(backbone_module).update_slot_params(
                prior_coder=import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.compressai_coder).update_slot_params(
                    entropy_bottleneck_channels=64,
                    use_inner_aux_opt=True,
                    use_bit_rate_loss=False,
                )
            ),
            entropy_coder=copy.deepcopy(entropy_coder_config),
        ),

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
            entropy_coder=copy.deepcopy(entropy_coder_config),
        ),

    # soft-vqvae
    # import_class_builder_from_module(general_codec)
    #     .set_override_name("softvqvae-c2")
    #     .update_slot_params(
    #         prior_model=import_class_builder_from_module(backbone_module).update_slot_params(
    #             prior_coder=import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.vq).update_slot_params(
    #                 latent_dim=2,
    #                 embedding_dim=32,
    #                 num_embeddings=256,
    #                 dist_type="RelaxedOneHotCategorical",
    #                 commitment_cost=0.0,
    #             )
    #         ),
    #     ),

    # mcquic
    import_class_builder_from_module(general_codec)
        .set_override_name("mcquic")
        .update_slot_params(
            prior_model=import_class_builder_from_module(backbone_module).update_slot_params(
                prior_coder=import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.mcquic_coder).update_slot_params(
                    channel=64,
                ),
            ),
            entropy_coder=copy.deepcopy(entropy_coder_config),
        ),

)
