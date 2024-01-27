import math

def _lr_linear1000(epoch):
    return 1.0 - (epoch if epoch < 1000 else 1000) / 1001

config = [
    ("model.prior_model.prior_coder.gs_temp", dict(
        scheduler_type="LambdaLR",
        scheduler_config=dict(lr_lambda=_lr_linear1000),
    )),
    ("model.prior_model.prior_coder.relax_temp", dict(
        scheduler_type="LambdaLR",
        scheduler_config=dict(lr_lambda=_lr_linear1000),
    )),
    ("model.prior_model.prior_coder.entropy_temp", dict(
        scheduler_type="LambdaLR",
        scheduler_config=dict(lr_lambda=_lr_linear1000),
    )),
    ("model.prior_model.prior_coder.cat_reduce_temp", dict(
        scheduler_type="LambdaLR",
        scheduler_config=dict(lr_lambda=_lr_linear1000),
    )),
    ("model.prior_model.prior_coder.var_scale", dict(
        scheduler_type="LambdaLR",
        scheduler_config=dict(lr_lambda=_lr_linear1000),
    )),

]
