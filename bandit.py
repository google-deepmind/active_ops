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

"""Class of bandit prolems for active policy selection."""

from typing import Dict, Sequence, Text

import numpy as np


class MAB(object):
  """Class of multi-armed bandit."""

  def __init__(self,
               selected_policies_fqe: Dict[Text, float]):
    self._policy_name_list = sorted(selected_policies_fqe)
    self.opes = np.asarray(
        [selected_policies_fqe[policy] for policy in self._policy_name_list])
    self.num_arms = len(selected_policies_fqe)

    self._policies = None
    self._rewards = None

  @property
  def rewards(self) -> Sequence[Sequence[float]]:
    return self._rewards

  def load_reward_samples(
      self, reward_samples_dict: Dict[Text, np.ndarray]
  ) -> Dict[Text, np.ndarray]:
    """Load pre-sampled arm rewards from a dict for relevant policies.

    Args:
      reward_samples_dict: a dictionary that maps all policy names to rewards

    Returns:
      A dictionary that maps policy names to rewards for subsample of policies.
    """
    for _ in self._policy_name_list:
      self._rewards = [
          reward_samples_dict[pi] for pi in self._policy_name_list
      ]
    return reward_samples_dict

  def pull(self, arm_index: int) -> float:
    """Pull an arm and return the reward.

    Draw a sample from pre-sampled rewards.

    Args:
      arm_index: index of the arm to pull.

    Returns:
      Sampled reward of the selected arm.
    """
    if arm_index < 0 or arm_index >= self.num_arms:
      raise ValueError(f'arm_index ({arm_index}) is out of the range of '
                       f'[0, {self.num_arms-1}]')

    return self._sample_arm_reward_from_samples(arm_index)

  def _sample_arm_reward_from_samples(self, arm_index: int) -> float:
    """Draw a random sample from the pre-sampled rewards."""
    rewards = self._rewards[arm_index]
    return np.random.choice(rewards)
