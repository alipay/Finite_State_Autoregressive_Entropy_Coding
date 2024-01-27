import math

config = [
    ("model.prior_model.prior_coder.gs_temp", dict(
        scheduler_type="ExponentialLR",
        scheduler_config=dict(gamma=math.exp(math.log(0.5) / 10)),
    )),
    ("model.prior_model.prior_coder.relax_temp", dict(
        scheduler_type="ExponentialLR",
        scheduler_config=dict(gamma=math.exp(math.log(0.5) / 10)),
    )),
    ("model.prior_model.prior_coder.entropy_temp", dict(
        scheduler_type="ExponentialLR",
        scheduler_config=dict(gamma=math.exp(math.log(0.5) / 10)),
    )),
    ("model.prior_model.prior_coder.cat_reduce_temp", dict(
        scheduler_type="ExponentialLR",
        scheduler_config=dict(gamma=math.exp(math.log(0.5) / 10)),
    )),
    ("model.prior_model.prior_coder.var_scale", dict(
        scheduler_type="ExponentialLR",
        scheduler_config=dict(gamma=math.exp(math.log(0.5) / 10)),
    )),

]
