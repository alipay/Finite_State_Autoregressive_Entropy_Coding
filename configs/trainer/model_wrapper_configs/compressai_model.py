config = dict(
    optimizer_type="Adam",
    optimizer_config=dict(
        base_lr=1e-4,
    ),
    # scheduler_type="ReduceLROnPlateau",
    # scheduler_config=dict(
    #     mode='min',
    #     factor=0.1,
    #     patience=10,
    #     # threshold=0.0001,
    #     # threshold_mode='rel',
    #     # cooldown=0,
    #     # min_lr=0
    #     verbose=True,
    # ),
    # scheduler_extra_config=dict(
    #     reduce_on_plateau=True,
    #     monitor='val_metric',
    #     strict=True,
    # ),
    multiopt_configs=[
        dict(
            optimizer_config = dict(
                optimizer_type="Adam",
                base_lr=0.001,
            ),
        ),
    ],
    metric_weight_table=dict(
        compression_ratio=1.0
    ),
    loss_aux_table={
        'losses/prior_model/prior_coder/loss_aux' : 0
    }
)