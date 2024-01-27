import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
from torch.distributions.utils import clamp_probs, broadcast_all
from torch.distributions.relaxed_categorical import ExpRelaxedCategorical, RelaxedOneHotCategorical
from torch.distributions.continuous_bernoulli import ContinuousBernoulli
from torch.distributions.utils import broadcast_all, probs_to_logits, logits_to_probs, lazy_property, clamp_probs
import math

class CategoricalRSample(distributions.Categorical):
    has_rsample = True
    def rsample(self, sample_shape=...):
        return self.probs

class ExpAsymptoticRelaxedCategorical(ExpRelaxedCategorical):
    def __init__(self, temperature, temperature_gumbel=0.5, probs=None, logits=None, validate_args=None):
        self.temperature_gumbel = temperature_gumbel
        super().__init__(temperature, probs=probs, logits=logits, validate_args=validate_args)
    
    def expand(self, batch_shape, _instance=None):
        new = super().__init__(batch_shape, _instance=_instance)
        new.temperature_gumbel = self.temperature_gumbel
        return new

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        uniforms = clamp_probs(torch.rand(shape, dtype=self.logits.dtype, device=self.logits.device))
        gumbels = -((-(uniforms.log())).log())
        scores = (self.logits * (1 - self.temperature_gumbel) + gumbels * self.temperature_gumbel) / self.temperature
        return scores - scores.logsumexp(dim=-1, keepdim=True)

class AsymptoticRelaxedOneHotCategorical(distributions.TransformedDistribution):
    arg_constraints = {'probs': distributions.constraints.simplex,
                       'logits': distributions.constraints.real}
    support = distributions.constraints.simplex
    has_rsample = True

    def __init__(self, temperature, temperature_gumbel=0.5, probs=None, logits=None, validate_args=None):
        base_dist = ExpAsymptoticRelaxedCategorical(temperature, temperature_gumbel, probs, logits)
        super().__init__(base_dist,
                         distributions.ExpTransform(),
                         validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(AsymptoticRelaxedOneHotCategorical, _instance)
        return super(AsymptoticRelaxedOneHotCategorical, self).expand(batch_shape, _instance=new)

    @property
    def temperature(self):
        return self.base_dist.temperature

    @property
    def temperature_gumbel(self):
        return self.base_dist.temperature_gumbel

    @property
    def logits(self):
        logits_norm = self.base_dist.logits
        # temperature = self.temperature.clamp(min=1e-6, max=1)
        logits_scaled = logits_norm / self.temperature_gumbel * (1 - self.temperature_gumbel)
        return logits_scaled - logits_scaled.logsumexp(dim=-1, keepdim=True)

    @property
    def probs(self):
        return torch.softmax(
            self.base_dist.logits / self.temperature_gumbel * (1 - self.temperature_gumbel),
            dim=-1
        )



class ExpDoubleRelaxedCategorical(ExpRelaxedCategorical):
    def __init__(self, temperature, temperature_gumbel=1.0, probs=None, logits=None, validate_args=None):
        self.temperature_gumbel = temperature_gumbel
        super().__init__(temperature, probs=probs, logits=logits, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = super().expand(batch_shape, _instance=_instance)
        new.temperature_gumbel = self.temperature_gumbel
        return new

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        uniforms = clamp_probs(torch.rand(shape, dtype=self.logits.dtype, device=self.logits.device))
        gumbels = -((-(uniforms.log())).log())
        scores = (self.logits + gumbels * self.temperature_gumbel) / self.temperature
        return scores - scores.logsumexp(dim=-1, keepdim=True)

    # TODO:
    def log_prob(self, value):
        K = self._categorical._num_events
        if self._validate_args:
            self._validate_sample(value)
        logits, value = broadcast_all(self.logits, value)
        log_scale = (torch.full_like(self.temperature, float(K)).lgamma() -
                     self.temperature.log().mul(-(K - 1)))
        score = logits - value.mul(self.temperature)
        score = (score - score.logsumexp(dim=-1, keepdim=True)).sum(-1)
        return score + log_scale


class DoubleRelaxedOneHotCategorical(distributions.TransformedDistribution):
    arg_constraints = {'probs': distributions.constraints.simplex,
                       'logits': distributions.constraints.real}
    support = distributions.constraints.simplex
    has_rsample = True

    def __init__(self, temperature, temperature_gumbel=1.0, probs=None, logits=None, validate_args=None):
        base_dist = ExpDoubleRelaxedCategorical(temperature, temperature_gumbel, probs, logits)
        super().__init__(base_dist,
                         distributions.ExpTransform(),
                         validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(DoubleRelaxedOneHotCategorical, _instance)
        return super(DoubleRelaxedOneHotCategorical, self).expand(batch_shape, _instance=new)

    @property
    def temperature(self):
        return self.base_dist.temperature

    @property
    def temperature_gumbel(self):
        return self.base_dist.temperature_gumbel

    @property
    def logits(self):
        return self.base_dist.logits

    @property
    def probs(self):
        return self.base_dist.probs

    # @property
    # def logits(self):
    #     logits_norm = self.base_dist.logits
    #     logits_scaled = logits_norm / self.base_dist.temperature_gumbel
    #     return logits_scaled - logits_scaled.logsumexp(dim=-1, keepdim=True)

    # @property
    # def probs(self):
    #     return torch.softmax(
    #         self.base_dist.logits / self.base_dist.temperature_gumbel,
    #         dim=-1
    #     )


class GroupedNormal(distributions.Normal):
    def __init__(self, loc, scale, validate_args=None):
        super().__init__(loc, scale, validate_args)
        batch_shape = self.loc.shape[:-1]
        event_shape = self.loc.shape[-1:]
        distributions.Distribution.__init__(self, batch_shape, event_shape, validate_args)

    def log_prob(self, value):
        return super().log_prob(value).sum(-1)

class InvertableGaussianRelaxedOneHotCategorical(distributions.TransformedDistribution):
    arg_constraints = {'loc': distributions.constraints.real, 'scale': distributions.constraints.positive}
    support = distributions.constraints.simplex
    has_rsample = True

    def __init__(self, transform, loc, scale, validate_args=None):
        base_dist = GroupedNormal(loc, scale)
        super().__init__(base_dist,
                         transform,
                         validate_args=validate_args)

    def to_categorical(self, mc_sample_size=None):
        if mc_sample_size is None:
            mc_sample_size = self.base_dist.event_shape[0]+1
        samples = self.rsample([mc_sample_size])
        probs_mean = samples.sum(0) / mc_sample_size
        return distributions.Categorical(probs=probs_mean)


class SoftmaxppTransform(distributions.Transform):
    def __init__(self, temperature, delta=1., eps=1e-8, cache_size=0):
        self.temperature = temperature
        self.delta = delta
        self.eps = eps
        super().__init__(cache_size)

    # https://github.com/cunningham-lab/igr/blob/2932a024f04e517b204a4dd68c44e16a76b9f835/Utils/Distributions.py#L240
    def _call(self, x):
        # TODO: consider transform when temperature is 0
        # if self.temperature == 0:
        #     return F.one_hot(x.argmax(-1), x.shape[-1]).type_as(x)
        lam = x / self.temperature
        lam_max = lam.max(-1, keepdim=True)[0]
        exp_lam = (lam - lam_max).exp()
        samples = exp_lam / (exp_lam.sum(-1, keepdim=True) + self.delta * (-lam_max).exp())
        probs = torch.cat([samples, 1-samples.sum(-1, keepdim=True)], dim=-1)
        return probs

    def _inverse(self, y):
        samples, last = y.split((y.shape[-1]-1, 1), dim=-1)
        samples_sum = samples.sum(-1, keepdim=True).clamp_max(1-self.eps) # stablize
        delta = self.delta # (-lam_max).exp() if self.delta is None else self.delta
        score_sum = samples_sum * delta / (1 - samples_sum) 
        scores = samples * (score_sum + delta)
        lam = scores.log()
        return lam * self.temperature

    def log_abs_det_jacobian(self, x, y):
        """
        Computes the log det jacobian `log |dy/dx|` given input and output.
        """
        samples, last = y.split((y.shape[-1]-1, 1), dim=-1)
        log_temp = torch.as_tensor(self.temperature).log() * (-samples.shape[-1])
        samples_sum = samples.sum(-1, keepdim=True).clamp_max(1-self.eps) # stablize
        log_score = (1 - samples_sum).log() + samples.log().sum(-1, keepdim=True) # keep x shape
        return log_temp + log_score


class InvertableGaussianSoftmaxppRelaxedOneHotCategorical(InvertableGaussianRelaxedOneHotCategorical):
    def __init__(self, loc, scale, temperature, validate_args=None):
        self.temperature = temperature
        transform = SoftmaxppTransform(temperature)
        super().__init__(transform, loc, scale, validate_args)

    @property
    def mean(self):
        return self.base_dist.mean

    @property
    def stddev(self):
        return self.base_dist.stddev

    @property
    def variance(self):
        return self.base_dist.variance
