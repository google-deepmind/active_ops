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

"""Kernel modules required by multivariate-normal multi-armed bandit models."""


from typing import Any, Optional, Dict, Text

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfk = tfp.math.psd_kernels


class ActionDistanceKernel(tfk.PositiveSemidefiniteKernel):
  """RBF kernel based on pre-recorded distance matrix.

  Covariance matrix based on distance between actions of different policies
  on the same set of states (with a learnable variance and lengthscale
  parameters).
  """

  def __init__(self,
               distances: np.ndarray,
               variance: float = 2.,
               lengthscale: float = 1.,
               bias_variance: Optional[float] = None,
               trainable: bool = True,
               variance_prior: Any = None,
               lengthscale_prior: Any = None,
               bias_variance_prior: Any = None,
               dtype: np.dtype = np.float32,
               name: str = 'ActionDistanceKernel'):
    """Kernel from distances in action vectors on subset of states.

    Args:
      distances: numpy array of euclidean distances between pairs of policies
      variance: variance (possibly trainable) parameter of the kernel
      lengthscale: lengthscale (possibly trainable) parameter
      bias_variance: optional value to initialise variance of the bias in kernel
      trainable: indicates if variance and lengthscale are trainable parameters
      variance_prior: config for variance prior
      lengthscale_prior: config for length scale prior
      bias_variance_prior: config for bias variance prior
      dtype: types of variance and lengthscale parameters
      name: name of the kernel
    """

    super(ActionDistanceKernel, self).__init__(
        feature_ndims=1, dtype=dtype, name=name)
    self._distances = distances
    self._log_var = tf.Variable(
        np.log(variance),
        trainable=trainable,
        dtype=self.dtype,
        name='kernel_log_var')
    self._var = tfp.util.DeferredTensor(self._log_var, tf.math.exp)
    self._variance_prior = variance_prior

    self._log_lengthscale = tf.Variable(
        np.log(lengthscale),
        trainable=trainable,
        dtype=self.dtype,
        name='kernel_log_lengthscale')
    self._lengthscale = tfp.util.DeferredTensor(self._log_lengthscale,
                                                tf.math.exp)
    self._lengthscale_prior = lengthscale_prior

    # if bias_variance parameter is passed, make a constant offset in the kernel
    if bias_variance is None:
      self._log_bias_variance = None
      self._bias_variance = tf.Variable(
          0., trainable=False, dtype=self.dtype, name='kernel_bias_variance')
    else:
      self._log_bias_variance = tf.Variable(
          np.log(bias_variance),
          trainable=True,
          dtype=self.dtype,
          name='kernel_bias_variance')
      self._bias_variance = tfp.util.DeferredTensor(self._log_bias_variance,
                                                    tf.math.exp)
    self._bias_variance_prior = bias_variance_prior

  def regularization_loss(self):
    """Regularization loss for trainable variables."""

    # Loss for variance: inverse Gamma distribution.
    prior = self._variance_prior
    if self._log_var.trainable and prior is not None and prior['use_prior']:
      inv_gamma = tfp.distributions.InverseGamma(prior['alpha'], prior['beta'])
      loss_var = -inv_gamma.log_prob(self._var)
    else:
      loss_var = 0.

    # Loss for lengthscale: inverse Gamma distribution.
    prior = self._lengthscale_prior
    if (self._log_lengthscale.trainable and
        prior is not None and prior['use_prior']):
      inv_gamma = tfp.distributions.InverseGamma(prior['alpha'], prior['beta'])
      loss_lengthscale = -inv_gamma.log_prob(self._lengthscale)
    else:
      loss_lengthscale = 0.

    # Loss for bias_variance: inverse Gamma distribution.
    prior = self._bias_variance_prior
    if (self._log_bias_variance is not None and
        prior is not None and prior['use_prior']):
      inv_gamma = tfp.distributions.InverseGamma(prior['alpha'], prior['beta'])
      loss_bias_var = -inv_gamma.log_prob(self._bias_variance)
    else:
      loss_bias_var = 0.

    return loss_var + loss_lengthscale + loss_bias_var

  def _compute_distances(self):
    # the parent kernel will just return what is recorded in the distance matrix
    return tf.convert_to_tensor(self._distances, dtype=tf.float32)

  def _apply(self, x1, x2, example_ndims=1):
    # transformation for a particular type of kernel
    distances = self._compute_distances()
    # add a constant offset kernel with trainable variance
    distances += self._bias_variance * tf.ones_like(distances, dtype=tf.float32)
    # get the relevant part of the matrix
    x1 = tf.cast(x1, tf.int32)
    x2 = tf.cast(x2, tf.int32)
    n_policies = tf.shape(distances)[0]
    distances = tf.reshape(distances, [-1])
    return tf.squeeze(tf.gather(distances, x1*n_policies+x2), -1)

  def get_lengthscale(self):
    return self._lengthscale

  def get_var(self):
    return self._var

  def get_bias_variance(self):
    return self._bias_variance

  def _batch_shape(self):
    """Parameter batch shape is ignored."""
    return tf.TensorShape([])

  def _batch_shape_tensor(self):
    """Parameter batch shape is ignored."""
    return tf.convert_to_tensor(tf.TensorShape([]))


class ActionDistanceMatern12(ActionDistanceKernel):
  """Matern kernel with v=1/2."""

  def __init__(self,
               distances: np.ndarray,
               variance: float = 2.,
               lengthscale: float = 1.,
               bias_variance: Optional[float] = None,
               trainable: bool = True,
               variance_prior: Any = None,
               lengthscale_prior: Any = None,
               bias_variance_prior: Any = None,
               dtype: np.dtype = np.float32,
               name: str = 'ActionDistanceMatern12'):

    super(ActionDistanceMatern12, self).__init__(
        distances=distances,
        variance=variance,
        lengthscale=lengthscale,
        bias_variance=bias_variance,
        trainable=trainable,
        variance_prior=variance_prior,
        lengthscale_prior=lengthscale_prior,
        bias_variance_prior=bias_variance_prior,
        dtype=dtype,
        name=name)

  def _compute_distances(self):
    # transformation for Matern kernel 3/2
    r = tf.divide(
        tf.convert_to_tensor(self._distances, dtype=tf.float32),
        self._lengthscale)
    return tf.exp(-r) * self._var


# Helper functions for kernels
def select_experiment_distances(selected_policies: Dict[Text, float],
                                policy_keys_in_distances: Dict[Text, int],
                                distances: np.ndarray) -> np.ndarray:
  """Get a submatrix of distances for the selected policies."""
  action_to_policy_keys = sorted(selected_policies)

  indexes_in_distance = []
  for action_to_policy_key in action_to_policy_keys:
    indexes_in_distance.append(policy_keys_in_distances[action_to_policy_key])

  return distances[indexes_in_distance, :][:, indexes_in_distance]
