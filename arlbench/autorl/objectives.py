"""This module contains the objectives for the AutoRL environment."""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from arlbench.core.algorithms import TrainFunc

# These are the ranks we are using for sorting:
# Runtime = 0
# Emissions = 1
# Reward = 2
# The reason is, that we want to measure the runtime right before and after starting
# the training to make it as accurate as possible. For the emissions, we want to have
# only the training emissions and not the calculation of other objectives

def discount_rewards(rewards, gamma=0.99):
    """Compute discounted rewards."""
    discounted = np.zeros_like(rewards, dtype=np.float32)
    running_add = 0
    for t in reversed(range(len(rewards))):
        running_add = running_add * gamma + rewards[t]
        discounted[t] = running_add
    return discounted

class Objective(ABC):
    """An abstract optimization objective for the AutoRL environment.

    It can be wrapped around the training function to calculate the objective.
    We do this be overriding the __new__() function. It allows us to imitate
    the behaviour of a basic function while keeping the advantages of a static class.
    """
    KEY: str  # Unique identifier
    RANK: int  # Sorting rank

    def __new__(cls, *args, **kwargs) -> TrainFunc:
        """Creates a new instance of this objective and directly wraps the train function.

        This is done by first creating an object and subsequently calling self.__call__().

        Returns:
            TrainFunc: Wrapped training function.
        """
        instance = super().__new__(cls)
        return instance.__call__(*args, **kwargs)

    @staticmethod
    @abstractmethod
    def __call__(
        train_func: TrainFunc, objectives: dict, optimize_objectives: str
    ) -> TrainFunc:
        """Wraps the training function with the objective calculation.

        Args:
            train_func (TrainFunc): Training function to wrap.
            objectives (dict): Dictionary to store objective.
            optimize_objectives (str): Whether to minimize/maximize the objectve.

        Returns:
            TrainFunc: Training function.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_spec() -> dict:
        """Returns a dictionary containing the specification of the objective.

        Returns:
            dict: Specification.
        """
        raise NotImplementedError

    def __lt__(self, other: Objective) -> bool:
        """Implements "less-than" comparison between two objectives. Used for sorting based on objective rank.

        Args:
            other (Objective): Other Objective to compare to.

        Returns:
            bool: Whether this Objective is less than the other Objective.
        """
        return self.RANK < other.RANK


class Runtime(Objective):
    """Runtime objective for the AutoRL environment. It measures the total training runtime."""
    KEY = "runtime"
    RANK = 0

    @staticmethod
    def __call__(
        train_func: TrainFunc, objectives: dict, optimize_objectives: str
    ) -> TrainFunc:
        """Wraps the training function with the runtime calculation."""
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = train_func(*args, **kwargs)
            runtime = time.time() - start_time

            # Naturally runtime is minimized. However, if we don't want
            # to minimize the objectives we have to flip the sign
            #if optimize_objectives != Runtime.get_spec()["optimize"]:
            #    runtime *= -1

            objectives["runtime"] = runtime
            return result

        return wrapper

    @staticmethod
    def get_spec() -> dict:
        """Returns a dictionary containing the specification of the objective."""
        return {"name": "runtime", "upper": None, "lower": 0.0, "optimize": "lower"}

class Reward(Objective):
    """Reward objective for the AutoRL environment. And applies an aggregation function."""

    KEY = "reward"
    RANK = 2

    @staticmethod
    def __call__(
        train_func: TrainFunc, objectives: dict, optimize_objectives: str, default_arg: str = "mean"
    ) -> TrainFunc:
        """Wraps the training function with the reward mean calculation."""
        aggregate = default_arg
        def wrapper(*args, **kwargs):
            result = train_func(*args, **kwargs)
            _, train_result = result
            aggregation_func = getattr(np, aggregate)
            reward = aggregation_func(train_result.eval_rewards[-1])
            reverse_optimize = default_arg in ["std", "var"]
            optimize_flag = not Reward.get_spec()["optimize"] if reverse_optimize else Reward.get_spec()["optimize"]

            # Naturally the mean of the reward is maximized. However, if we don't want
            # to maximize the objectives we have to flip the sign
            if optimize_objectives != optimize_flag:
                reward *= -1

            objectives[f"{Reward.KEY}_{aggregate}"] = reward.item()
            return result

        return wrapper

    @staticmethod
    def get_spec() -> dict:
        """Returns a dictionary containing the specification of the objective."""
        return {
            "name": Reward.KEY,
            "upper": None,
            "lower": None,
            "optimize": "upper",
        }

class DiscountedReward(Objective):
    """Discounted reward objective for the AutoRL environment. It measures the last discounted evaluation rewards."""

    KEY = "discounted_reward"
    RANK = 2

    @staticmethod
    def __call__(
        train_func: TrainFunc, objectives: dict, optimize_objectives: str, gamma: float = 0.99, default_arg: str = "mean"
    ) -> TrainFunc:
        """Wraps the training function with the reward mean calculation."""
        def wrapper(*args, **kwargs):
            result = train_func(*args, **kwargs)
            _, train_result = result
            # Content: [[step1_rewards], [step1_rewards+step2_rewards], ..., [sum(step1...stepN_rewards)]]
            cumulative_eval_reward = train_result.eval_rewards
            rewards = []
            for i in range(1, len(cumulative_eval_reward)):
                rewards.append(cumulative_eval_reward[i] - cumulative_eval_reward[i-1])
            rewards = np.array(rewards)
            rewards = discount_rewards(rewards, gamma=gamma)

            aggregation_func = getattr(np, default_arg)
            reward = aggregation_func(np.sum(rewards, axis=1))
            reverse_optimize = default_arg in ["std", "var"]
            optimize_flag = not DiscountedReward.get_spec()["optimize"] if reverse_optimize else DiscountedReward.get_spec()["optimize"]

            # Naturally the mean of the reward is maximized. However, if we don't want
            # to maximize the objectives we have to flip the sign
            if optimize_objectives != optimize_flag:
                reward *= -1

            objectives[f"{DiscountedReward.KEY}_{default_arg}"] = reward.item()
            return result

        return wrapper

    @staticmethod
    def get_spec() -> dict:
        """Returns a dictionary containing the specification of the objective."""
        return {
            "name": DiscountedReward.KEY,
            "upper": None,
            "lower": None,
            "optimize": "upper",
        }

class TrainReward(Objective):
    """Reward objective for the AutoRL environment. It measures the mean of the last training rewards."""

    KEY = "train_reward"
    RANK = 2

    @staticmethod
    def __call__(
        train_func: TrainFunc, objectives: dict, optimize_objectives: str, default_arg: str = "mean"
    ) -> TrainFunc:
        """Wraps the training function with the reward mean calculation."""
        aggregate = default_arg
        def wrapper(*args, **kwargs):
            result = train_func(*args, **kwargs)
            _, train_result = result
            n_envs = train_result.trajectories.reward.shape[-1]
            train_rewards = np.reshape(train_result.trajectories.reward, (-1,n_envs))
            train_dones = np.reshape(train_result.trajectories.done, (-1,n_envs))
            rewards = []
            for i in range(n_envs):
                episode_end_indices = np.where(train_dones[:, i])[0]
                last_indices = episode_end_indices[-3:]
                previous_indices = episode_end_indices[-4:-1]
                for start, end in zip(previous_indices, last_indices, strict=False):
                    episode_reward = train_rewards[start+1:end, i]
                    rewards.append(sum(episode_reward))

            aggregation_func = getattr(np, aggregate)
            reward = aggregation_func(rewards)
            reverse_optimize = default_arg in ["std", "var"]
            optimize_flag = not TrainReward.get_spec()["optimize"] if reverse_optimize else TrainReward.get_spec()["optimize"]
            
            # Naturally the mean of the reward is maximized. However, if we don't want
            # to maximize the objectives we have to flip the sign
            if optimize_objectives != optimize_flag:
                reward *= -1

            objectives[f"{TrainReward.KEY}_{aggregate}"] = reward.item()
            return result

        return wrapper

    @staticmethod
    def get_spec() -> dict:
        """Returns a dictionary containing the specification of the objective."""
        return {
            "name": TrainReward.KEY,
            "upper": None,
            "lower": None,
            "optimize": "upper",
        }

class DiscountedTrainReward(Objective):
    """Discounted reward objective for the AutoRL environment. It measures the mean of the last discounted training rewards."""

    KEY = "discounted_train_reward"
    RANK = 2

    @staticmethod
    def __call__(
        train_func: TrainFunc, objectives: dict, optimize_objectives: str, gamma: float = 0.99, default_arg: str = "mean"
    ) -> TrainFunc:
        """Wraps the training function with the reward mean calculation."""
        aggregate = default_arg
        def wrapper(*args, **kwargs):
            result = train_func(*args, **kwargs)
            _, train_result = result
            n_envs = train_result.trajectories.reward.shape[-1]
            train_rewards = np.reshape(train_result.trajectories.reward, (-1,n_envs))
            train_dones = np.reshape(train_result.trajectories.done, (-1,n_envs))
            rewards = []
            for i in range(n_envs):
                episode_end_indices = np.where(train_dones[:, i])[0]
                last_indices = episode_end_indices[-3:]
                previous_indices = episode_end_indices[-4:-1]
                for start, end in zip(previous_indices, last_indices, strict=False):
                    episode_reward = train_rewards[start+1:end, i]
                    episode_reward = discount_rewards(episode_reward, gamma)
                    rewards.append(sum(episode_reward))

            aggregation_func = getattr(np, aggregate)
            reward_mean = aggregation_func(rewards)
            reverse_optimize = default_arg in ["std", "var"]
            optimize_flag = not DiscountedTrainReward.get_spec()["optimize"] if reverse_optimize else DiscountedTrainReward.get_spec()["optimize"]

            # Naturally the mean of the reward is maximized. However, if we don't want
            # to maximize the objectives we have to flip the sign
            if optimize_objectives != optimize_flag:
                reward_mean *= -1

            objectives[f"{DiscountedTrainReward.KEY}_{aggregate}"] = reward_mean.item()
            return result

        return wrapper

    @staticmethod
    def get_spec() -> dict:
        """Returns a dictionary containing the specification of the objective."""
        return {
            "name": DiscountedTrainReward.KEY,
            "upper": None,
            "lower": None,
            "optimize": "upper",
        }

class Emissions(Objective):
    """Emissions objective for the AutoRL environment. It measures the emissions during the training using code carbon."""

    KEY = "emissions"
    RANK = 1

    @staticmethod
    def __call__(
        train_func: TrainFunc, objectives: dict, optimize_objectives: str
    ) -> TrainFunc:
        """Wraps the training function with the emissions calculation."""
        def wrapper(*args, **kwargs):
            from codecarbon import EmissionsTracker

            tracker = EmissionsTracker(
                save_to_file=False, output_dir="/tmp", logging_logger=None
            )
            tracker.start()
            emissions = tracker.stop()

            # Naturally emissions are minimized. However, if we don't want
            # to minimize objectives we have to flip the sign
            if emissions is not None and optimize_objectives != Emissions.get_spec()["optimize"]:
                emissions *= -1

            objectives[Emissions.KEY] = emissions

            try:
                result = train_func(*args, **kwargs)
            finally:
                objectives["emissions"] = emissions
            return result

        return wrapper

    @staticmethod
    def get_spec() -> dict:
        """Returns a dictionary containing the specification of the objective."""
        return {"name": "emissions", "upper": None, "lower": 0.0, "optimize": "lower"}


OBJECTIVES = {o.KEY: (o, o.RANK) for o in [Runtime, Reward, DiscountedReward, Emissions, TrainReward, DiscountedTrainReward]}
