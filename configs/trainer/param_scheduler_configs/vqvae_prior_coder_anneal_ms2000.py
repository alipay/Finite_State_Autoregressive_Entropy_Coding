config = [
    ("model.prior_model.prior_coder.gs_temp", dict(
        scheduler_type="MultiStepLR",
        scheduler_config=dict(milestones=[500, 1000, 1200, 1400, 1600, 1800], gamma=0.5),
    )),
    ("model.prior_model.prior_coder.relax_temp", dict(
        scheduler_type="MultiStepLR",
        scheduler_config=dict(milestones=[500, 1000, 1200, 1400, 1600, 1800], gamma=0.5),
    )),
    ("model.prior_model.prior_coder.entropy_temp", dict(
        scheduler_type="MultiStepLR",
        scheduler_config=dict(milestones=[500, 1000, 1200, 1400, 1600, 1800], gamma=0.5),
    )),
    ("model.prior_model.prior_coder.cat_reduce_temp", dict(
        scheduler_type="MultiStepLR",
        scheduler_config=dict(milestones=[500, 1000, 1200, 1400, 1600, 1800], gamma=0.5),
    )),
    ("model.prior_model.prior_coder.var_scale", dict(
        scheduler_type="MultiStepLR",
        scheduler_config=dict(milestones=[500, 1000, 1200, 1400, 1600, 1800], gamma=0.5),
    )),
]
