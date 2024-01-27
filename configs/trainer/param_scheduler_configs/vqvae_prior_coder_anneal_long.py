config = [
    ("model.prior_model.prior_coder.gs_temp", dict(
        scheduler_type="StepLR",
        scheduler_config=dict(step_size=250, gamma=0.5),
    )),
    ("model.prior_model.prior_coder.relax_temp", dict(
        scheduler_type="StepLR",
        scheduler_config=dict(step_size=250, gamma=0.5),
    )),
    ("model.prior_model.prior_coder.entropy_temp", dict(
        scheduler_type="StepLR",
        scheduler_config=dict(step_size=250, gamma=0.5),
    )),
    ("model.prior_model.prior_coder.var_scale", dict(
        scheduler_type="StepLR",
        scheduler_config=dict(step_size=250, gamma=0.5),
    )),
]
