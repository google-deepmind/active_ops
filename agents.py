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

"""Agents classes that implement various sampling strategies.

Implements uniform sampling of all arms (UniformAgent), and UCB algorithms.
"""

import abc

import numpy as np
import multiarm_model


class Agent(abc.ABC):
  """Agent that executes a policy that pulls the arms."""

  def __init__(self, model: multiarm_model.MultiArmModel):
    self._model = model

  def update(self, arm: int, reward: float) -> None:
    """Update the algorithm given a pulled arm and the reward observation."""
    self._model.update(arm, reward)

  @abc.abstractmethod
  def select_action(self):
    """Choose an arm index to pull."""

  @property
  def best_arm(self) -> int:
    return np.argmax(self._model.mean)


class UniformAgent(Agent):
  """Agent that samples all arms uniformly."""

  def select_action(self) -> int:
    steps = self._model.steps
    return np.random.randint(len(steps))


class UCBAgent(Agent):
  """UCB algorithm.

  Selects a sample based on maximizing UCB criterion: mean + exploration_coef *
  st_dev. Works with any arm model that provides mean and variance estimate.
  """

  def __init__(self,
               model: multiarm_model.MultiArmModel,
               minimum_pulls: int = 0,
               initial_rand_samples: int = 0,
               exploration_coef: float = 0.0):
    super().__init__(model)
    self._minimum_pulls = minimum_pulls
    self._initial_rand_samples = initial_rand_samples
    self._exploration_coef = exploration_coef

  def select_action(self) -> int:
    steps = self._model.steps

    # Pull a random arm if number of initial random samples is not yet reached.
    if steps.sum() < self._initial_rand_samples:
      return np.random.randint(len(steps))
    # Pull a random arm that has fewer than the minimum number of pulls yet.
    init_steps = np.nonzero(steps < self._minimum_pulls)[0]
    if init_steps.size > 0:
      return init_steps[np.random.randint(init_steps.size)]
    # Compute UCB criterion.
    scores = self._model.mean + self._exploration_coef*self._model.stddev
    return np.random.choice(np.flatnonzero(scores == scores.max()))

