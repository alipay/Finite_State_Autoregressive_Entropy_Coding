config = dict(
    optimizer_type="Adam",
    optimizer_config=dict(
        base_lr=5e-4,
    ),
    metric_weight_table={'metric_dict/prior_model/estimated_epd' : 1.0}
)