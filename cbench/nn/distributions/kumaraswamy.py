# imported from newer versions of pytorch
# add safe_range parameter to stablize training

import torch
from torch._six import nan
from torch.distributions import constraints
from torch.distributions.uniform import Uniform
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import AffineTransform, PowerTransform
from torch.distributions.utils import broadcast_all
from torch.distributions import Beta

EULER_CONST = 0.5772156649

def _moments(a, b, n):
    """
    Computes nth moment of Kumaraswamy using using torch.lgamma
    """
    arg1 = 1 + n / a
    log_value = torch.lgamma(arg1) + torch.lgamma(b) - torch.lgamma(arg1 + b)
    return b * torch.exp(log_value)

def _log_beta_fn(a, b):
    log_value = torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b)
    return log_value

class Kumaraswamy(TransformedDistribution):
    r"""
    Samples from a Kumaraswamy distribution.

    Example::

        >>> m = Kumaraswamy(torch.tensor([1.0]), torch.tensor([1.0]))
        >>> m.sample()  # sample from a Kumaraswamy distribution with concentration alpha=1 and beta=1
        tensor([ 0.1729])

    Args:
        concentration1 (float or Tensor): 1st concentration parameter of the distribution
            (often referred to as alpha)
        concentration0 (float or Tensor): 2nd concentration parameter of the distribution
            (often referred to as beta)
    """
    arg_constraints = {'concentration1': constraints.positive, 'concentration0': constraints.positive}
    support = constraints.unit_interval
    has_rsample = True

    def __init__(self, concentration1, concentration0, eps=1e-6, safe_range=0.01, validate_args=None):
        self.concentration1, self.concentration0 = broadcast_all(concentration1, concentration0)
        finfo = torch.finfo(self.concentration0.dtype)
        # added safe_range to avoid nan
        base_dist = Uniform(torch.full_like(self.concentration0, safe_range),
                            torch.full_like(self.concentration0, 1 - safe_range),
                            validate_args=validate_args)
        transforms = [PowerTransform(exponent=self.concentration0.reciprocal()),
                      AffineTransform(loc=1., scale=-1.),
                      PowerTransform(exponent=self.concentration1.reciprocal())]
        self.eps = eps
        super(Kumaraswamy, self).__init__(base_dist, transforms, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Kumaraswamy, _instance)
        new.concentration1 = self.concentration1.expand(batch_shape)
        new.concentration0 = self.concentration0.expand(batch_shape)
        return super(Kumaraswamy, self).expand(batch_shape, _instance=new)


    @property
    def mean(self):
        return _moments(self.concentration1, self.concentration0, 1)

    @property
    def mode(self):
        # Evaluate in log-space for numerical stability.
        log_mode = self.concentration0.reciprocal() * \
            (-self.concentration0).log1p() - (-self.concentration0 * self.concentration1).log1p()
        log_mode[(self.concentration0 < 1) | (self.concentration1 < 1)] = nan
        return log_mode.exp()

    @property
    def variance(self):
        return _moments(self.concentration1, self.concentration0, 2) - torch.pow(self.mean, 2)

    def entropy(self):
        t1 = (1 - self.concentration1.reciprocal())
        t0 = (1 - self.concentration0.reciprocal())
        H0 = torch.digamma(self.concentration0 + 1) + EULER_CONST
        return t0 + t1 * H0 - torch.log(self.concentration1) - torch.log(self.concentration0)

    # def rsample(self, sample_shape=torch.Size()):
    #     # return super().rsample(sample_shape)
    #     u = self.base_dist.rsample(sample_shape)
    #     a, b = self.concentration1, self.concentration0
    #     samples = (1. - u.log().div(b+self.eps).exp()).log().div(a+self.eps).exp()
    #     return samples

    def kl_beta(self, beta_dist : Beta, num_terms=10, eps=1e-8):
        a, b = self.concentration1, self.concentration0
        prior_alpha, prior_beta = beta_dist.concentration1, beta_dist.concentration0
        first_term = ((a - prior_alpha)/(a+eps)) * (-EULER_CONST - torch.digamma(b) - 1./(b+eps))
        second_term = (a+eps).log() + (b+eps).log() + _log_beta_fn(prior_alpha, prior_beta)
        third_term = -(b - 1)/(b+eps)

        sum_term = torch.zeros_like(a)
        for i in range(1, num_terms+1):
            sum_term += _log_beta_fn(float(i)/(a + eps), b).exp() / (i + a * b)

        sum_term *= (prior_beta - 1) * b

        return first_term + second_term + third_term + sum_term
