config = dict(
    optimizer_type="Adabelief",
    optimizer_config=dict(
        base_lr=2e-4,
        betas=(0.5, 0.999),
        rectify=False,
    ),
    multiopt_configs=[
        dict(
            optimizer_config=dict(
                optimizer_type="Adabelief",
                base_lr=2e-4,
                betas=(0.5, 0.999),
                rectify=False,
            ),
        )
    ]
)