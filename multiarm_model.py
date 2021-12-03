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

"""Class that represents a multiarm model.

Classes inherit from MultiArmModel which maintains lists of stats of single
arms. IndependentMultiArmModel treats all arms independently.
"""


import abc
from typing import Any, Dict, Optional, Sequence, Text, Type, Union

import numpy as np
import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp

import arm_model
import gp_models

tfk = tfp.math.psd_kernels


class MultiArmModel(abc.ABC):
  """Posterior distribution of the reward mean of multiple arms."""

  def __init__(self, num_arms: int):
    self._num_arms = num_arms

  @property
  def num_arms(self) -> int:
    return self._num_arms

  @property
  @abc.abstractmethod
  def steps(self):
    """Return an array of the steps of each arm."""

  @property
  @abc.abstractmethod
  def mean(self):
    """Return an array of the estimate of the mean rewards."""

  @property
  @abc.abstractmethod
  def stddev(self):
    """Return an array of the estimation stddev of the mean rewards."""

  @abc.abstractmethod
  def update(self, arm: int, reward: float):
    """Update the model given a pulled arm and the reward observation."""

  def sample(self):
    """Return an array of samples of the mean rewards from the posterior."""
    return self.mean + self.stddev * np.random.randn(self._num_arms)


class IndependentMultiArmModel(MultiArmModel):
  """Posterior distribution of the reward mean of multiple arms."""

  def __init__(self,
               num_arms: int,
               arm_class: Type[arm_model.SingleArmModel],
               arm_args: Optional[Sequence[Any]] = None,
               arm_kwargs: Optional[Sequence[Dict[Text, Any]]] = None):
    super().__init__(num_arms)
    self._arms = []
    if arm_args is None:
      arm_args = [[] for _ in range(num_arms)]
    if arm_kwargs is None:
      arm_kwargs = [{} for _ in range(num_arms)]
    for i in range(num_arms):
      self._arms.append(arm_class(*arm_args[i], **arm_kwargs[i]))

  @property
  def steps(self) -> np.ndarray:
    return np.array([arm.step for arm in self._arms])

  def update(self, arm: int, reward: float):
    """Update the model given a pulled arm and the reward observation."""
    self._arms[arm].update(reward)

  @property
  def mean(self) -> np.ndarray:
    """Return an array of the estimate of the mean rewards."""
    return np.array([arm.mean for arm in self._arms])

  @property
  def mean_without_prior(self) -> np.ndarray:
    """Return an array of the estimate of the mean rewards."""
    return np.array([arm.mean_without_prior for arm in self._arms])

  @property
  def mean2(self) -> np.ndarray:
    """Return an array of the estimate of the mean squared rewards."""
    return np.array([arm.mean2 for arm in self._arms])

  @property
  def sum2(self):
    """Return an array of the estimate of the sum of squared rewards."""
    return np.array([arm.sum2 for arm in self._arms])

  @property
  def stddev(self) -> np.ndarray:
    """Return an array of the estimation stddev of the mean rewards."""
    return np.array([arm.stddev for arm in self._arms])

  def sample(self) -> np.ndarray:
    """Return an array of samples of the mean rewards from the posterior."""
    return np.array([arm.sample() for arm in self._arms])


class MVNormalBase(MultiArmModel):
  """Base multivariate normal model for a fixed set of arms.

  Implemented by a Gaussian process indexed at 1-D integer locations.

  Please note that the GP model and its hyper-parameters are updated lazily,
  which means every time a new observation is added with `update`, it
  delays the update of the posterior mean, stddev and its hyper-parameters
  optimization until next time a public method is called. This is useful to
  avoid repeated computation of the posterior when adding multiple observations.
  But it also means that the hyper-parameters optimization is run only once in
  this case.
  """

  def __init__(self,
               num_arms: int,
               kernel: tfk.PositiveSemidefiniteKernel,
               gp: gp_models.GaussianProcess,
               optimizer_config: Optional[Dict[Text, Any]] = None,
               dtype: np.dtype = np.float32):
    super().__init__(num_arms)
    self._kernel = kernel
    self._gp = gp
    self._dtype = dtype

    self._arm_xs = tf.range(num_arms, dtype=dtype)[:, None]
    self._make_optimizer(optimizer_config)
    self._steps = np.zeros(num_arms)
    # Initialize self._mean and self._stddev.
    self._get_mean_stddev()
    self._to_update = False

  def _make_optimizer(self, optimizer_config):
    """Make the optimizer for GP hyper-parameters."""
    self._optimizer_config = optimizer_config
    if optimizer_config is not None:
      required_configs = [
          'optimizer_name', 'learning_rate', 'steps_per_update']
      for key in required_configs:
        if key not in optimizer_config:
          raise ValueError(f'{key} is required in optimizer_config')
      self._optimizer = getattr(
          snt.optimizers, optimizer_config['optimizer_name'])(
              learning_rate=optimizer_config['learning_rate'],
              **optimizer_config.get('kwargs', {}))
    else:
      self._optimizer = None

  @property
  @abc.abstractmethod
  def trainable_variables(self):
    """Trainable variables."""

  @property
  def steps(self) -> np.ndarray:
    return self._steps

  @property
  def mean(self):
    """Return an array of the estimate of the mean rewards."""
    self._update_gp()
    return self._mean  # pytype: disable=attribute-error  # bind-properties

  @property
  def stddev(self):
    """Return an array of the estimation stddev of the mean rewards."""
    self._update_gp()
    return self._stddev  # pytype: disable=attribute-error  # bind-properties

  def update(self, arm: int, reward: float):
    """Update the model given a pulled arm and the reward observation."""
    self._steps[arm] += 1
    x = tf.constant(np.array([[arm]], dtype=self._dtype))
    y = tf.constant(np.array([reward], dtype=self._dtype))
    self._gp.add(x, y)
    self._to_update = True  # Mark the model to be updated later.

  def sample(self, joint=False):
    """Sample a function from the posterior of GP."""
    self._update_gp()

    if joint:
      # Sample from the joint distributions over all arms.
      one_sample = self._gp.sample().numpy()
    else:
      # Sample from the marginal distributions, independent among arms.
      distr = self._gp.index(self._arm_xs, latent_function=True)
      one_sample = (distr.mean().numpy() +
                    distr.stddev().numpy() * np.random.randn(self._num_arms))
    return one_sample

  def _update_gp(self):
    loss_dict = {}
    if self._to_update:
      if self._optimizer is not None:
        loss_dict = self._optimize(self._optimizer_config['steps_per_update'])
      self._get_mean_stddev()
      self._to_update = False
    return loss_dict

  @tf.function
  def _optimize(self, steps):
    """Optimizer hyper-parameters."""
    model_loss = tf.constant([0.])
    regularization_loss = tf.constant([0.])
    for _ in tf.range(steps):
      with tf.GradientTape() as tape:
        model_loss = self._gp.loss()
        loss = model_loss
        regularization_loss = tf.reshape(self._regularization_loss(), [-1])
        loss += regularization_loss
      gradients = tape.gradient(loss, self.trainable_variables)
      self._optimizer.apply(gradients, self.trainable_variables)
    return {
        'model_loss': model_loss,
        'regularization_loss': regularization_loss
    }

  @abc.abstractmethod
  def _regularization_loss(self):
    """Regularization loss for trainable variables."""

  def _get_mean_stddev(self):
    """Compute and update posterior mean and stddev to self._mean/_stddev."""
    distr = self._gp.index(self._arm_xs, latent_function=True)
    self._mean = distr.mean().numpy()
    self._stddev = distr.stddev().numpy()


class MVNormal(MVNormalBase):
  """Multivariate normal model for a fixed set of arms.

  Offset can be a vector of values indexed by the arm.
  """

  def __init__(self,
               num_arms: int,
               kernel: tfk.PositiveSemidefiniteKernel,
               observation_noise_variance: float = 1.0,
               observation_noise_variance_prior: Any = None,
               optimizer_config: Optional[Dict[Text, Any]] = None,
               offset: Optional[float] = None,
               dtype: np.dtype = np.float32):
    """Initialize the multi-variate normal model.

    Args:
      num_arms: number of arms / discrete indices.
      kernel: GP kernel.
      observation_noise_variance: initial guess of the variance of the
          observation noise.
      observation_noise_variance_prior: configuration dict to specify the
          inverse Gamma prior for the variance if provided. It should include
          three keys: 'use_prior' (boolean, whether to use prior), 'alpha' and
          'beta' (prior parameters in floats).
      optimizer_config: configuration dict to specify the sonnet optimizer for
          GP hyper-parameter optimization. It must includes the following keys:
          'optimizer_name', 'learning_rate', 'steps_per_update', and optionally
          other parameters to initialize the optimizer.
      offset: initial guess of the trainable scalar mean hyper-parameter if
          provided. Otherwise, it is assumed to be fixed at zero.
      dtype: float type of the model.
    """
    # Define the GP.
    if offset is None:
      self._offset = tf.Variable(0., dtype=dtype, trainable=False,
                                 name='offset')
    else:
      self._offset = tf.Variable(offset, dtype=dtype, trainable=True,
                                 name='offset')

    self._log_obs_var = tf.Variable(
        np.log(observation_noise_variance), dtype=dtype, trainable=True,
        name='log_obs_var')
    # Add constant to avoid obs_var becoming too small and thus kernel not
    # invertible once the observations start repeating
    # self._obs_var = tfp.util.DeferredTensor(
    #     self._log_obs_var, lambda x: tf.math.exp(x) + 1.0)
    # Or use this line if do not need to restrict obs_var
    self._obs_var = tfp.util.DeferredTensor(self._log_obs_var, tf.math.exp)
    self._observation_noise_variance_prior = observation_noise_variance_prior

    gp = gp_models.GaussianProcess(
        num_indices=num_arms,
        kernel=kernel,
        offset=self._offset,
        variance=self._obs_var)

    super().__init__(num_arms=num_arms,
                     kernel=kernel,
                     gp=gp,
                     optimizer_config=optimizer_config,
                     dtype=dtype)

  @property
  def trainable_variables(self):
    possible_trainable_variables = (
        self._offset, self._log_obs_var) + self._kernel.trainable_variables
    return tuple(x for x in possible_trainable_variables
                 if x is not None and x.trainable)

  def _regularization_loss(self):
    # Loss for offset.
    loss_offset = 0.

    # Loss for observation noise variance: inverse Gamma distribution.
    prior = self._observation_noise_variance_prior
    if self._log_obs_var.trainable and prior is not None and prior['use_prior']:
      inv_gamma = tfp.distributions.InverseGamma(prior['alpha'], prior['beta'])
      loss_obs_var = -inv_gamma.log_prob(self._obs_var)
    else:
      loss_obs_var = 0.

    # Loss for kernel variables.
    if hasattr(self._kernel, 'regularization_loss'):
      loss_kernel = self._kernel.regularization_loss()
    else:
      loss_kernel = 0.

    return loss_offset + loss_obs_var + loss_kernel

  def get_observation_noise_variance(self):
    return self._obs_var.numpy()

  def get_offset(self):
    return self._offset.numpy()


class MVNormalWithSideObs(MVNormalBase):
  """Multivariate normal model for a fixed set of arms and side observations."""

  def __init__(
      self,
      num_arms: int,
      kernel: tfk.PositiveSemidefiniteKernel,
      side_observations: Union[Sequence[float], Sequence[Sequence[float]]],
      observation_noise_variance: float = 1.0,
      observation_noise_variance_prior: Any = None,
      tie_side_observations_variance_with_main_observations: bool = False,
      tie_side_observations_variance_along_sources: bool = False,
      side_observations_variance: Union[float, Sequence[float]] = 1.0,
      side_observations_variance_prior: Any = None,
      side_observations_variance_trainable: bool = False,
      optimizer_config: Optional[Dict[Text, Any]] = None,
      offset: Optional[float] = None,
      dtype: np.dtype = np.float32):
    """Initialize the multi-variate normal model with side observations.

    The side observation may come from one or multiple sources, e.g. different
    OPE estimates.

    Args:
      num_arms: number of arms / discrete indices.
      kernel: GP kernel.
      side_observations: array of `num_arms` side observations, or a 2D array
          of side observations from multiple sources in shape
          (`num_sources`, `num_arms`).
      observation_noise_variance: initial guess of the variance of the
          observation noise.
      observation_noise_variance_prior: configuration dict to specify the
          inverse Gamma prior for the variance if provided. It should include
          three keys: 'use_prior' (boolean, whether to use prior), 'alpha' and
      tie_side_observations_variance_with_main_observations: whether to share
          the variance parameter of all side observations with the variance of
          the main observations. If true, the following options related to side
          observation variance will be ignored.
      tie_side_observations_variance_along_sources: whether to share the
          variance parameter of the side observation among all the sources.
      side_observations_variance: value (or initial guess if trainable) of the
          side observation variance, or one value per source if provided as an
          array.
      side_observations_variance_prior: configuration dict to specify the
          inverse Gamma prior for the variance of side observations if provided.
      side_observations_variance_trainable: whether to train the side
          observation variance parameters.
      optimizer_config: configuration dict to specify the sonnet optimizer for
          GP hyper-parameter optimization. It must includes the following keys:
          'optimizer_name', 'learning_rate', 'steps_per_update', and optionally
          other parameters to initialize the optimizer.
      offset: initial guess of the trainable scalar mean hyper-parameter if
          provided. Otherwise, it is assumed to be fixed at zero.
      dtype: float type of the model.
    """
    # Define the GP.
    if offset is None:
      self._offset = tf.Variable(0., dtype=dtype, trainable=False,
                                 name='offset')
    else:
      self._offset = tf.Variable(offset, dtype=dtype, trainable=True,
                                 name='offset')
    self._log_obs_var = tf.Variable(
        np.log(observation_noise_variance), dtype=dtype, trainable=True,
        name='log_obs_var')
    self._obs_var = tfp.util.DeferredTensor(self._log_obs_var, tf.math.exp)
    self._observation_noise_variance_prior = observation_noise_variance_prior

    # Make sure side_obs has 2 dimensions.
    side_obs = np.asarray(side_observations)
    if side_obs.ndim == 1:
      side_obs = side_obs[None, :]

    if tie_side_observations_variance_with_main_observations:
      # Tie the side observation variance with the main observation variance.
      # Other side observation varaince related arguments are ignored.
      self._log_side_obs_var = None
      self._side_obs_var = self._obs_var
    else:
      side_obs_var = np.asarray(side_observations_variance)
      if (side_obs_var.ndim > 1 or
          (side_obs_var.ndim == 1 and side_obs_var.size != side_obs.shape[0])):
        raise ValueError('side_observations_variance should be either a scalar '
                         'or a sequence with the same length as the number of '
                         'sources.')
      if side_observations_variance_trainable:
        # If side_observations_variance is a scalar and
        # tie_side_observations_variance_along_sources if False, make a vector
        # of trainable variance, one per side observation source.
        if (side_obs_var.size == 1 and
            not tie_side_observations_variance_along_sources):
          side_obs_var = side_obs_var.ravel()[0] * np.ones(
              (side_obs.shape[0], 1))
        self._log_side_obs_var = tf.Variable(
            np.log(side_obs_var), dtype=dtype, trainable=True,
            name='log_side_obs_var')
        self._side_obs_var = tfp.util.DeferredTensor(self._log_side_obs_var,
                                                     tf.math.exp)
      else:
        self._log_side_obs_var = None
        self._side_obs_var = side_obs_var
    self._side_observations_variance_prior = side_observations_variance_prior

    gp = gp_models.GaussianProcessWithSideObs(
        num_indices=num_arms,
        kernel=kernel,
        offset=self._offset,
        variance=self._obs_var,
        side_observations=side_obs,
        side_observations_variance=self._side_obs_var)

    super().__init__(num_arms=num_arms,
                     kernel=kernel,
                     gp=gp,
                     optimizer_config=optimizer_config,
                     dtype=dtype)
    # Mark it as to be updated because we add side observations.
    self._to_update = True

  @property
  def trainable_variables(self):
    possible_trainable_variables = (
        self._offset,
        self._log_obs_var,
        self._log_side_obs_var) + self._kernel.trainable_variables
    return tuple(x for x in possible_trainable_variables
                 if x is not None and x.trainable)

  def _regularization_loss(self):
    # Loss for offset.
    loss_offset = 0.

    # Loss for observation noise variance: inverse Gamma distribution.
    prior = self._observation_noise_variance_prior
    if self._log_obs_var.trainable and prior is not None and prior['use_prior']:
      inv_gamma = tfp.distributions.InverseGamma(prior['alpha'], prior['beta'])
      loss_obs_var = -inv_gamma.log_prob(self._obs_var)
    else:
      loss_obs_var = 0.

    # Loss for side observation noise variance: inverse Gamma distribution.
    prior = self._side_observations_variance_prior
    if (self._log_side_obs_var is not None and
        prior is not None and prior['use_prior']):
      inv_gamma = tfp.distributions.InverseGamma(prior['alpha'], prior['beta'])
      loss_side_obs_var = -tf.reduce_sum(inv_gamma.log_prob(self._side_obs_var))
    else:
      loss_side_obs_var = 0.

    # Loss for kernel variables.
    if hasattr(self._kernel, 'regularization_loss'):
      loss_kernel = self._kernel.regularization_loss()
    else:
      loss_kernel = 0.

    return loss_offset + loss_obs_var + loss_side_obs_var + loss_kernel

  def get_observation_noise_variance(self):
    return self._obs_var.numpy()

  def get_side_observation_noise_variance(self):
    return self._side_obs_var.numpy()

  def get_offset(self):
    return self._offset.numpy()
