import math

config = [
    ("model.prior_model.prior_coder.gs_temp", dict(
        scheduler_type="MultiStepLR",
        scheduler_config=dict(
            milestones=[500, 1000, ] + list(range(1100, 1500, 100)) + list(range(1500, 1800, 50)) + list(range(1800, 2000, 25)), 
            gamma=0.5),
    )),
    ("model.prior_model.prior_coder.relax_temp", dict(
        scheduler_type="MultiStepLR",
        scheduler_config=dict(
            milestones=[500, 1000, ] + list(range(1100, 1500, 100)) + list(range(1500, 1800, 50)) + list(range(1800, 2000, 25)), 
            gamma=0.5),
    )),
    ("model.prior_model.prior_coder.entropy_temp", dict(
        scheduler_type="MultiStepLR",
        scheduler_config=dict(
            milestones=[500, 1000, ] + list(range(1100, 1500, 100)) + list(range(1500, 1800, 50)) + list(range(1800, 2000, 25)), 
            gamma=0.5),
    )),
    ("model.prior_model.prior_coder.cat_reduce_temp", dict(
        scheduler_type="ExponentialLR",
        scheduler_config=dict(gamma=math.exp(math.log(0.5) / 200)),
    )),
]
