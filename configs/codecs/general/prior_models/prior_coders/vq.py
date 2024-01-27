from configs.class_builder import ClassBuilder, ParamSlot
from configs.import_utils import import_class_builder_from_module, import_all_config_from_dir

from cbench.modules.prior_model.prior_coder import MultiChannelVQPriorCoder

# temp workaround for None
ar_offsets_choices = dict(
    none=None,
)
ar_offsets_choices.update(**import_all_config_from_dir("ar_offsets", caller_file=__file__))

config = ClassBuilder(MultiChannelVQPriorCoder)\
    .add_all_kwargs_as_param_slot()\
    .update_args(
    latent_dim=ParamSlot("latent_dim"),
    num_embeddings=ParamSlot("num_embeddings"),
    embedding_dim=ParamSlot("embedding_dim"),
    # use_vamp_prior=ParamSlot(),
    # use_ema_update=ParamSlot(),
    # ema_decay=ParamSlot(),
    # ema_epsilon=ParamSlot(),
    # embedding_lr_modifier=ParamSlot(),
    # ema_reduce_ddp=ParamSlot(),
    # ema_adjust_sample=ParamSlot(),
    # use_code_freq=ParamSlot(),
    # code_freq_manual_update=ParamSlot(),
    # update_code_freq_ema_decay=ParamSlot(),
    # kl_cost=ParamSlot(),
    # use_st_gumbel=ParamSlot(),
    # commitment_cost=ParamSlot(),
    # commitment_over_exp=ParamSlot(),
    # vq_cost=ParamSlot(),
    # test_sampling=ParamSlot(),
    # initialization_mean=ParamSlot(), 
    # initialization_scale=ParamSlot(),
    dist_type=ParamSlot(
        choices=[
            None, 
            "RelaxedOneHotCategorical", 
            "AsymptoticRelaxedOneHotCategorical", 
            "DoubleRelaxedOneHotCategorical"
        ]
    ),
    # relax_temp=ParamSlot(),
    # relax_temp_anneal=ParamSlot(),
    # gs_temp=ParamSlot(),
    # gs_temp_anneal=ParamSlot(),
    # entropy_temp=ParamSlot(),
    # entropy_temp_min=ParamSlot(),
    # entropy_temp_anneal=ParamSlot(),
    ar_offsets=ParamSlot(
        choices=ar_offsets_choices,
        default="none",
    )

)