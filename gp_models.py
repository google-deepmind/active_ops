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

"""Gaussian process model at discrete indices."""

from typing import Sequence, Union

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfk = tfp.math.psd_kernels


class DistributionWrapper(object):
  """Helper class for MVNormal model with mean and stddev methods."""

  def __init__(self, mean, stddev):
    self._mean = mean
    self._stddev = stddev

  def mean(self):
    return self._mean

  def stddev(self):
    return self._stddev


class GaussianProcess(object):
  """Gaussian process model at discrete indices."""

  def __init__(self,
               num_indices: int,
               kernel: tfk.PositiveSemidefiniteKernel,
               offset: Union[float, tf.Tensor, tf.Variable],
               variance: Union[float, tf.Tensor, tf.Variable]):
    """Creates a model for a stochastic process.

    Args:
      num_indices: integer, the number of discrete indices.
      kernel: An instance of
        `tfp.positive_semidefinite_kernels.PositiveSemidefiniteKernels`. The
        type of the kernel will be used to cast the inputs and outputs of the
        model.
      offset: Scalar, offset the observations by this amount.
      variance: variance of the Gaussian observation noise.
    """
    self._n_xs = num_indices
    self._kernel = kernel
    self._offset = offset
    self._dtype = kernel.dtype
    self._variance = variance

    # self._xs is not supposed to change and is treated as constants.
    self._xs = tf.range(self.n_xs, dtype=self._dtype)[:, None]

    # These values will be updated and are treated as variables.
    self._ys_num = tf.Variable(tf.zeros(self.n_xs, dtype=self._dtype),
                               trainable=False)
    self._ys_mean = tf.Variable(tf.zeros(self.n_xs, dtype=self._dtype),
                                trainable=False)
    self._ys_sq_mean = tf.Variable(tf.zeros(self.n_xs, dtype=self._dtype),
                                   trainable=False)

  def add(self, xs, ys):
    """Adds a batch of observations to the model.

    Args:
      xs: An array (or equivalent) of shape `[B, input_dim]`, where `B` is an
        arbitrary batch dimension, and `input_dim` must be compatible with
        the trailing dimension of the already fed in observations (if any).
      ys: An array (or equivalent) of shape `[B]` or `[B, 1]`,
        where `B` is an arbitrary batch dimension.
    """
    xs = np.asarray(xs, self._dtype)
    ys = np.asarray(ys, self._dtype)
    if ys.ndim > 2 or (ys.ndim == 2 and ys.shape[1] > 1):
      raise ValueError('ys must have a shape of [B] or [B, 1]')
    ys = ys.ravel()

    ys_num = self._ys_num.numpy()
    ys_mean = self._ys_mean.numpy()
    ys_sq_mean = self._ys_sq_mean.numpy()
    for x, y in zip(xs, ys):
      i = int(x[0])
      ys_num[i] += 1.
      ys_mean[i] += (y - ys_mean[i]) / ys_num[i]
      ys_sq_mean[i] += (y ** 2 - ys_sq_mean[i]) / ys_num[i]
    self._ys_num.assign(ys_num)
    self._ys_mean.assign(ys_mean)
    self._ys_sq_mean.assign(ys_sq_mean)

  def index(self, index_points, latent_function: bool = False):
    """Compute the marginal posterior distribution at the given `index_points`.

    Args:
      index_points: A Tensor (or equivalent) of shape `[B, input_dim]`, where
        `B` is an arbitrary batch dimension, and `input_dim` must be compatible
        with the trailing dimension of the already fed in observations (if any).
      latent_function: If True, return the distribution of the latent
        function value at index points without observation noise. Otherwise,
        return the distribution of noisy observations.

    Returns:
      An object with mean and stddev methods.
    """
    _, post_mean, post_var = self._marginal_and_posterior()
    index_points = tf.squeeze(tf.cast(index_points, tf.int32), axis=1)
    post_mean = tf.gather(post_mean, index_points)
    post_var = tf.gather(post_var, index_points)
    if not latent_function:
      post_var += self._variance
    return DistributionWrapper(post_mean, tf.sqrt(post_var))

  def loss(self):
    """The negative log probability of the observations under the GP."""
    log_marg, _, _ = self._marginal_and_posterior(margin_only=True)
    return -log_marg

  @property
  def n_xs(self):
    """Returns the number of unique indices."""
    return self._n_xs

  @property
  def n_observations(self):
    """Returns the number of observations used by the model."""
    return tf.reduce_sum(self._ys_num)

  def _merge_observations(self):
    """Merge observations at the same index into a single observation."""
    # Observations.
    ys_mean = self._ys_mean - self._offset
    ys_var = self._variance  # Scalar.
    ys_s = self._ys_sq_mean - tf.square(self._ys_mean)  # Empirical variance.

    # Filter indices without observations.
    index_mask = tf.greater(self._ys_num, 0)
    xs = tf.boolean_mask(self._xs, index_mask)
    n_xs = tf.cast(tf.shape(xs)[0], self._dtype)
    ys_mean = tf.boolean_mask(ys_mean, index_mask)
    ys_s = tf.boolean_mask(ys_s, index_mask)
    ys_num = tf.boolean_mask(self._ys_num, index_mask)

    o_mean = ys_mean
    o_var = ys_var / ys_num

    # Additional likelihood term inside exp(-1/2(.)).
    extra_term = -0.5 * tf.reduce_sum(ys_num / ys_var * ys_s)
    # Additional likelihood term of 1/\sqrt(2\pi * var)
    extra_term += -0.5 * (
        tf.math.log(2.0 * np.pi) * (self.n_observations - n_xs)
        + tf.math.log(ys_var) * self.n_observations
        - tf.reduce_sum(tf.math.log(o_var)))

    return index_mask, xs, o_mean, o_var, extra_term

  @tf.function
  def _marginal_and_posterior(self, margin_only=False):
    """Compute marginal log-likelihood and posterior mean and variance."""
    index_mask, xs, o_mean, o_var, extra_term = self._merge_observations()
    n_xs = tf.cast(tf.shape(xs)[0], self._dtype)

    log_marg = extra_term - 0.5 * tf.math.log(2.0 * np.pi) * n_xs

    # K + sigma2*I  or K + Sigma (with Sigma diagonal) matrix
    # where X are training or inducing inputs
    k_x_all = self._kernel.matrix(xs, self._xs)
    k_xx = tf.boolean_mask(k_x_all, index_mask, axis=1)
    k = k_xx + tf.linalg.diag(o_var)

    chol = tf.linalg.cholesky(k)

    # L^{-1} \mu
    a = tf.linalg.triangular_solve(chol, tf.expand_dims(o_mean, 1), lower=True)
    log_marg += (
        -tf.reduce_sum(tf.math.log(tf.linalg.diag_part(chol)))
        - 0.5 * tf.reduce_sum(tf.square(a)))
    log_marg = tf.reshape(log_marg, [-1])
    if margin_only:
      return (log_marg,
              tf.zeros((), dtype=self._dtype),
              tf.zeros((), dtype=self._dtype))

    # predict at the training inputs X
    a2 = tf.linalg.triangular_solve(chol, k_x_all, lower=True)

    # posterior variance
    k_all_diag = self._kernel.apply(self._xs, self._xs)
    post_var = k_all_diag - tf.reduce_sum(tf.square(a2), 0)

    # posterior mean
    post_mean = tf.squeeze(tf.matmul(a2, a, transpose_a=True), axis=1)
    post_mean = post_mean + self._offset

    return log_marg, post_mean, post_var

  def sample(self):
    """Compute marginal log-likelihood and posterior mean and variance."""
    index_mask, _, o_mean, o_var, _ = self._merge_observations()

    # K + sigma2*I  or K + Sigma (with Sigma diagonal) matrix
    # where X are training or inducing inputs
    k_all_all = self._kernel.matrix(self._xs, self._xs)
    k_x_all = tf.boolean_mask(k_all_all, index_mask)
    k_xx = tf.boolean_mask(k_x_all, index_mask, axis=1)
    k = k_xx + tf.linalg.diag(o_var)

    chol = tf.linalg.cholesky(k)

    # L^{-1} \mu
    a = tf.linalg.triangular_solve(chol, tf.expand_dims(o_mean, 1), lower=True)

    # predict at the training inputs X
    a2 = tf.linalg.triangular_solve(chol, k_x_all, lower=True)

    # posterior mean
    post_mean = tf.squeeze(tf.matmul(a2, a, transpose_a=True), axis=1)
    post_mean = post_mean + self._offset

    # full posterior covariance matrix.
    post_var = k_all_all - tf.matmul(a2, a2, transpose_a=True)

    mvn = tfd.MultivariateNormalTriL(
        loc=post_mean, scale_tril=tf.linalg.cholesky(post_var))
    return mvn.sample()


class GaussianProcessWithSideObs(GaussianProcess):
  """Gaussian process model at discrete indices and side observations."""

  def __init__(self,
               num_indices: int,
               kernel: tfk.PositiveSemidefiniteKernel,
               offset: Union[float, tf.Tensor, tf.Variable],
               variance: Union[float, tf.Tensor, tf.Variable],
               side_observations: Sequence[Sequence[float]],
               side_observations_variance: Union[float, Sequence[float],
                                                 Sequence[Sequence[float]],
                                                 tf.Tensor, tf.Variable]):
    """Creates a model for a stochastic process.

    Args:
      num_indices: integer, the number of discrete indices.
      kernel: An instance of
        `tfp.positive_semidefinite_kernels.PositiveSemidefiniteKernels`. The
        type of the kernel will be used to cast the inputs and outputs of the
        model.
      offset: Scalar, offset the observations by this amount.
      variance: variance of the Gaussian observation noise.
      side_observations: [num_side_observation_per_index, num_indices] array of
        side observations.
      side_observations_variance: side observation variances of the same shape
        as side_observations or can be broadcast to the same shape.
    """
    super().__init__(num_indices=num_indices,
                     kernel=kernel,
                     offset=offset,
                     variance=variance)
    self._zs_var = side_observations_variance

    # self._zs is not supposed to change and is treated as constants.
    self._zs = tf.constant(side_observations, dtype=self._dtype)
    if self._zs.ndim != 2:
      raise ValueError('Side observation dimension must be 2.')
    if self._zs.shape[1] != num_indices:
      raise ValueError('Side observation dimension does not match num_indices.')

  def _merge_observations(self):
    """Merge observations and side observations at the same index."""
    # Observations.
    ys_mean = self._ys_mean - self._offset
    ys_var = self._variance  # Scalar.
    ys_s = self._ys_sq_mean - tf.square(self._ys_mean)  # Empirical variance.

    # Side observations.
    zs = self._zs - self._offset
    # Broadcast zs_var to have the same shape as zs.
    zs_var = self._zs_var + tf.zeros_like(zs)

    o_var = 1. / (tf.reduce_sum(1. / zs_var, axis=0) + self._ys_num / ys_var)
    o_mean = (tf.reduce_sum(zs / zs_var, axis=0)
              + self._ys_num / ys_var * ys_mean) * o_var

    # Additional likelihood term inside exp(-1/2(.)).
    extra_term = -0.5 * tf.reduce_sum(
        tf.reduce_sum(tf.square(zs) / zs_var, axis=0)
        + self._ys_num / ys_var * tf.square(ys_mean)
        - tf.square(o_mean) / o_var
        + self._ys_num / ys_var * ys_s)
    # Additional likelihood term of 1/\sqrt(2\pi * var)
    extra_term += -0.5 * (
        tf.math.log(2.0 * np.pi) * (
            self.n_observations + (zs.shape[0] - 1) * zs.shape[1])
        + tf.reduce_sum(tf.math.log(zs_var))
        + tf.math.log(ys_var) * self.n_observations
        - tf.reduce_sum(tf.math.log(o_var)))

    # All the indices are returned due to the side observation.
    index_mask = tf.ones(self._xs.shape[0], dtype=tf.bool)
    xs = self._xs
    return index_mask, xs, o_mean, o_var, extra_term
