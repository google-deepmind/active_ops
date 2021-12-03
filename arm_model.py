# Copyright 2020 DeepMind Technologies Limited.


# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Class that models a single arm of a bandit.

An arm object keeps track of stats obtained when an arm was pulled. This file
contains a simple arm (SingleArmModel) and Bayesian arm (SingleBayesArm) that
models independent arms.
"""

import abc
from typing import Sequence, Tuple

import numpy as np
import scipy as sp


def inv_gamma_prior_ml(var_samples: Sequence[float],
                       tol: float = 1e-6,
                       max_iters: int = 100) -> Tuple[float, float]:
  """Estimate inverse gamma prior parameters of variances with MLE.

  ML2 algorithm in https://arxiv.org/abs/1605.01019.

  Args:
    var_samples: sequence of variance samples.
    tol: tolerance of the alpha estimate.
    max_iters: max number of iterations to run.

  Returns:
    Pair of shape parameter alpha and scale parameter beta.
  """
  vs = np.asarray(var_samples)
  mu = vs.mean()
  v = vs.var()
  inv_mean = (1. / vs).mean()
  n = len(vs)
  a = mu ** 2 / v + 2
  c = -np.log(inv_mean * n) - np.log(vs).mean()

  for _ in range(max_iters):
    curr_a = a
    num = c - sp.special.digamma(a) + np.log(n * a)
    den = a ** 2 * (1 / a - sp.special.polygamma(1, a))
    inv_a = 1. / a + num / den
    a = 1. / inv_a
    if np.abs(a - curr_a) < tol:
      break
  else:
    print(
        'MLE of inverse gamma prior parameters is terminated without '+
        'convergence after %d iterations.', max_iters)
  b = a / inv_mean
  return a, b


class SingleArmModel(abc.ABC):
  """Class modelling a single arm of multi-armed bandit.

  Posterior distribution of the reward mean.
  Assume an uninformative prior.
  """

  def __init__(self):
    self._t = 0
    self._mean_x = 0.
    self._mean_x2 = 0.

  def update(self, reward: float):
    self._t += 1
    self._mean_x += 1 / self._t * (reward - self._mean_x)
    self._mean_x2 += 1 / self._t * (reward ** 2 - self._mean_x2)

  @property
  def step(self) -> int:
    return self._t

  @property
  def mean(self) -> float:
    """Return the estimate of the mean reward."""
    return self._mean_x

  @property
  def mean2(self) -> float:
    """Return the estimate of the mean squared reward."""
    return self._mean_x2

  @property
  def mean_without_prior(self) -> float:
    """Return the estimate of the mean reward ignoring the prior if exists."""
    return self._mean_x

  @property
  def sum2(self) -> float:
    return self._mean_x2 * self._t

  @property
  def stddev(self) -> float:
    """Return the estimation stddev of the mean reward."""
    if self._t <= 0:
      raise ValueError('Cannot compute the stddev if the number of pulls '
                       f'({self._t}) <= 0')
    var_x = self._mean_x2 - self._mean_x**2
    return np.sqrt(var_x / self._t)

  def sample(self) -> float:
    """Return a sample of the mean reward from the posterior."""
    return self.mean + self.stddev * np.random.randn()


class SingleBayesArm(SingleArmModel):
  """Estimate the posterior of the reward mean and variance parameters.

  The hierarchical Bayesian model for the reward is as follows:
    r ~ Norm(mu, sigma^2)
    mu ~ Norm(m, s^2)
    sigma^2 ~ IG(alpha, beta)
  where mu and sigma^2 are the mean and variance of the Gaussian distribution
  for reward r. m and s^2 are the mean and variance of the prior Gaussian
  distribution of mu. alpha and beta are the alpha and beta parameters of the
  prior inverse Gamma distribution of sigma^2.

  When new observations are added, we update the estimate of reward mean mu and
  variance sigma^2. We either sample sigma^2 with Gibbs sampling for a few steps
  or compute the MAP of the joint posterior with coordinate ascend. Given
  samples or point estimate of sigma^2, we estimate the conditoinal mean and
  variance of mu with
    mu ~ ensemble of P(mu|sigma_sample) or mu ~ P(mu|sigma^2 MAP)
  """

  def __init__(self, prior_mean: float, prior_std: float, alpha: float,
               beta: float, sample: bool, steps: int = 10, burnin: int = 0):
    """Initialize the Bayesian model for a single arm.

    Args:
      prior_mean: mean parameter for the prior of variable mu.
      prior_std: prior std parameter for the prior of variable mu.
      alpha: alpha parameter for the prior of variable mu.
      beta: beta parameter for the prior of variable mu.
      sample: sample sigma^2 or estimate a point estimate from the joint MAP.
      steps: if `sample` is True, it is the number of samples to keep from Gibbs
        sampling after burnin periord. Otherwise, it is the number of coordinate
        ascend steps after burn in.
      burnin: burn-in period in sampling or optimization of sigma^2.
    """
    super().__init__()
    self._m = prior_mean
    self._s = prior_std
    self._a = alpha
    self._b = beta

    self._sample = sample  # Sample sigma2 from posterior or compute the MAP.
    self._steps = steps
    self._burnin = burnin

    # Initialize mu and sigma**2 from the prior mode for sampling or
    # optimization.
    self._mu = self._m
    self._sigma2 = self._b / (self._a + 1)

    # Maintaining the posterior mean and atd of mu at step t and a list of
    # samples of sigma2.
    self._m_t = self._m
    self._s_t = self._s
    if self._sample:
      self._sigma2_samples = 1. / np.random.gamma(
          shape=self._a, scale=1 / self._b, size=self._steps)
    else:
      # Single sample at the prior mode.
      self._sigma2_samples = np.array([self._b / (self._a + 1)])

    self._to_update_pos = False  # Requires updating posterior at next call.

  def update(self, reward: float):
    super().update(reward)
    self._to_update_pos = True

  def _mu_cond_on_sigma2(self, sigma2):
    """Return conditional mean and variance of mu given sigma2."""
    m = self._m
    s = self._s
    t = self._t
    mx = self._mean_x

    ratio = sigma2 / s ** 2
    m_t = (t * mx + ratio * m) / (t + ratio)
    s2_t = sigma2 / (t + ratio)
    return m_t, s2_t

  def _sigma2_cond_on_mu(self, mu):
    """Return conditional mean and variance of sigma2 given mu."""
    a = self._a
    b = self._b
    t = self._t
    mx = self._mean_x
    mx2 = self._mean_x2
    varx = mx2 - mx ** 2  # Sample variance.

    a_t = a + 0.5 * t
    b_t = b + 0.5 * ((mx - mu) ** 2 + varx)
    return a_t, b_t

  def _sample_sigma2(self):
    """Obtain samples or MAP of sigma2."""

    mu = self._mu
    sigma2 = self._sigma2
    for i in range(self._steps + self._burnin):
      # Sample or optimize sigma2 given mu.
      a_t, b_t = self._sigma2_cond_on_mu(mu)
      if self._sample:
        sigma2 = 1. / np.random.gamma(shape=a_t, scale=1 / b_t)
      else:
        sigma2 = b_t / (a_t + 1)

      # Sample or optimize mu given sigma2.
      m_t, s2_t = self._mu_cond_on_sigma2(sigma2)
      if self._sample:
        mu = m_t + np.sqrt(s2_t) * np.random.randn()
      else:
        mu = m_t

      if self._sample and i >= self._burnin:
        self._sigma2_samples[i - self._burnin] = sigma2
    if not self._sample:
      self._sigma2_samples[0] = sigma2

    self._mu = mu
    self._sigma2 = sigma2
    return self._sigma2_samples

  def _update_posterior(self):
    """Sample or optimize sigma2 and update the posterior of mu."""
    if not self._to_update_pos:
      return

    sigma2_samples = self._sample_sigma2()

    ms = np.zeros(len(sigma2_samples))
    s2s = np.zeros(len(sigma2_samples))
    for i, sigma2 in enumerate(sigma2_samples):
      ms[i], s2s[i] = self._mu_cond_on_sigma2(sigma2)

    pos_mean = ms.mean()
    pos_mean_sq = (ms**2 + s2s).mean()
    pos_std = np.sqrt(pos_mean_sq - pos_mean**2)

    self._m_t = pos_mean
    self._s_t = pos_std
    self._to_update_pos = False

  @property
  def mean(self) -> float:
    """Return the estimate of the mean reward."""
    self._update_posterior()
    return self._m_t

  @property
  def stddev(self) -> float:
    """Return the estimation stddev of the mean reward."""
    self._update_posterior()
    return self._s_t

  def sample(self) -> float:
    """Return a sample of the mean reward from the posterior."""
    # Sample a sigma2.
    sigma2 = self._sigma2_samples[np.random.randint(len(self._sigma2_samples))]

    # Sample mu conditioned on sigma2.
    m_t, s2_t = self._mu_cond_on_sigma2(sigma2)

    return m_t + np.sqrt(s2_t) * np.random.randn()
