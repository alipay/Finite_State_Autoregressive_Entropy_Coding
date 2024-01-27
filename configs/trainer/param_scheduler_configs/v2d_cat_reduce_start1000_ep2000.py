import math

def _lr_stage2(epoch):
    return math.pow(math.exp(math.log(0.5) / 200), epoch-1000) \
           if epoch > 1000 else 1.0

config = [
    ("model.prior_model.prior_coder.gs_temp", dict(
        scheduler_type="LambdaLR",
        scheduler_config=dict(
            lr_lambda=_lr_stage2
            ),
    )),
    ("model.prior_model.prior_coder.relax_temp", dict(
        scheduler_type="LambdaLR",
        scheduler_config=dict(
            lr_lambda=_lr_stage2
            ),
    )),
    ("model.prior_model.prior_coder.entropy_temp", dict(
        scheduler_type="LambdaLR",
        scheduler_config=dict(
            lr_lambda=_lr_stage2
            ),
    )),
    ("model.prior_model.prior_coder.cat_reduce_temp", dict(
        scheduler_type="LambdaLR",
        scheduler_config=dict(
            lr_lambda=_lr_stage2
            ),
    )),
]
