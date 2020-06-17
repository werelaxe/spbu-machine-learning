from typing import Tuple

import numpy as np
from attr import dataclass


def get_state(observation):
    return np.concatenate([observation[part] for part in ["joints", "to_target", "body_velocities"]])


class BasicEnvWrapper:
    def __init__(self, env):
        self.env = env
        self.action_space_dim = self._action_space_dim()
        self.state_space_dim = self._state_space_dim()
        self.action_space = self._action_space()

    def _action_space_dim(self):
        raise NotImplementedError()

    def _state_space_dim(self):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()

    def step(self, action):
        raise NotImplementedError()

    def render(self):
        self.env.render()

    def _action_space(self):
        raise NotImplementedError()


@dataclass
class ActionSpace:
    low: float
    high: float
    shape: Tuple[int]


class MujocoEnvWrapper(BasicEnvWrapper):
    def _action_space(self):
        spec = self.env.action_spec()
        return ActionSpace(spec.minimum[0], spec.maximum[0], spec.shape)

    def __init__(self, env):
        super().__init__(env)

    def reset(self):
        return get_state(self.env.reset().observation)

    def step(self, action):
        time_step = self.env.step(action)
        return get_state(time_step.observation), time_step.reward, time_step.last(), {}

    def render(self):
        self.env.render()

    def _action_space_dim(self):
        return self.env.action_spec().shape[0]

    def _state_space_dim(self):
        return sum(states_part.shape[0] for states_part in self.env.observation_spec().values())


class GymEnvWrapper(BasicEnvWrapper):
    def _action_space(self):
        return self.env.action_space

    def _action_space_dim(self):
        return self.env.action_space.shape[0]

    def _state_space_dim(self):
        return self.env.observation_space.shape[0]

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)
