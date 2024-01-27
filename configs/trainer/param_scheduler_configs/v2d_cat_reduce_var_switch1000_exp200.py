import math

def _lr_var(epoch):
    return 1.0 if epoch > 1000 else 0.0

config = [
    ("model.prior_model.prior_coder.gs_temp", dict(
        scheduler_type="ExponentialLR",
        scheduler_config=dict(gamma=math.exp(math.log(0.5) / 200)),
    )),
    ("model.prior_model.prior_coder.relax_temp", dict(
        scheduler_type="ExponentialLR",
        scheduler_config=dict(gamma=math.exp(math.log(0.5) / 200)),
    )),
    ("model.prior_model.prior_coder.entropy_temp", dict(
        scheduler_type="ExponentialLR",
        scheduler_config=dict(gamma=math.exp(math.log(0.5) / 200)),
    )),
    ("model.prior_model.prior_coder.cat_reduce_temp", dict(
        scheduler_type="ExponentialLR",
        scheduler_config=dict(gamma=math.exp(math.log(0.5) / 200)),
    )),
    ("model.prior_model.prior_coder.var_scale", dict(
        scheduler_type="LambdaLR",
        scheduler_config=dict(lr_lambda=_lr_var),
    )),

]
