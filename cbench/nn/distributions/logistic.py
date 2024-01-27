# imported from newer versions of pytorch
# add safe_range parameter to stablize training

import torch
from torch._six import nan
from torch.distributions import constraints
from torch.distributions.uniform import Uniform
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import AffineTransform, SigmoidTransform
from torch.distributions.utils import broadcast_all
from torch.distributions import Beta

class Logistic(TransformedDistribution):
    arg_constraints = {'loc': constraints.real, 'scale': constraints.positive}
    support = constraints.real
    has_rsample = True
    _mean_carrier_measure = 0

    @property
    def mean(self):
        return self.loc

    # @property
    # def mode(self):
    #     return self.loc

    # @property
    # def stddev(self):
    #     return self.scale

    # @property
    # def variance(self):
    #     return self.stddev.pow(2)

    def __init__(self, loc, scale, validate_args=None):
        self.loc, self.scale = broadcast_all(loc, scale)
        super(Logistic, self).__init__(
            Uniform(torch.zeros_like(self.loc), torch.ones_like(self.loc)),
            transforms=[
                SigmoidTransform().inv,
                AffineTransform(self.loc, self.scale)
            ]
        )

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Logistic, _instance)
        new.loc = self.loc.expand(batch_shape)
        new.scale = self.scale.expand(batch_shape)
        return super(Logistic, self).expand(batch_shape, _instance=new)
