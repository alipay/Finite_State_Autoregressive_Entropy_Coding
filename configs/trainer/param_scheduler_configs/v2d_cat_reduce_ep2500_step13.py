import math

config = [
    ("model.prior_model.prior_coder.relax_temp", dict(
        scheduler_type="MultiStepLR",
        scheduler_config=dict(
            milestones=[500, 1000, ] + list(range(1200, 2000, 200)) + list(range(2000, 2300, 100)) + list(range(2300, 2500, 50)), 
            gamma=0.5),
    )),
    ("model.prior_model.prior_coder.cat_reduce_temp", dict(
        scheduler_type="ExponentialLR",
        scheduler_config=dict(gamma=math.exp(math.log(0.5) / 200)),
    )),
]
