"""Unity as a stable-baselines3 VecEnv.

Unity already auto resets, so its response is exactly what a VecEnv must return:
obs belongs to the next episode and the finished one's last observation travels
beside it. SB3 reads that last observation from infos["terminal_observation"] and
bootstraps the value function only when "TimeLimit.truncated" is also set, which
is what separating terminated from truncated is for.
"""

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv

from bridge.sim_robot import Simulation
from shared.features import DIM, Features

# Actions live in [-1, 1] so the gaussian policy is symmetric; unity takes percent.
DUTY = 100.0


class SimVecEnv(VecEnv):
    def __init__(self, port=5005, host="127.0.0.1", seed=0, randomize_scenario=True):
        self.sim = Simulation(port=port, host=host, session_seed=seed)
        self.randomize_scenario = randomize_scenario
        self.features = None
        self.actions = None

        super().__init__(
            self.sim.arenas,
            spaces.Box(-np.inf, np.inf, (DIM,), np.float32),
            spaces.Box(-1.0, 1.0, (self.sim.action_dim,), np.float32),
        )

    def reset(self):
        state = self.sim.reset(randomize_scenario=self.randomize_scenario)
        self.features = [Features() for _ in range(self.num_envs)]
        return np.array([f.first(o) for f, o in zip(self.features, state.obs)], np.float32)

    def step_async(self, actions):
        self.actions = actions * DUTY

    def step_wait(self):
        state = self.sim.step(self.actions)

        obs = np.empty((self.num_envs, DIM), np.float32)
        dones = np.empty(self.num_envs, bool)
        infos = [{} for _ in range(self.num_envs)]

        for i in range(self.num_envs):
            dones[i] = state.terminated[i] or state.truncated[i]
            if not dones[i]:
                obs[i] = self.features[i].update(state.obs[i])
                continue

            # The terminal observation belongs to the episode that just ended, so it
            # goes through the extractor that still holds that episode's history.
            infos[i]["terminal_observation"] = np.array(
                self.features[i].update(state.terminal_obs[i]), np.float32)
            if state.truncated[i]:
                infos[i]["TimeLimit.truncated"] = True

            self.features[i] = Features()
            obs[i] = self.features[i].first(state.obs[i])

        return obs, np.array(state.reward, np.float32), dones, infos

    def close(self):
        self.sim.close()

    def seed(self, seed=None):
        """Unity owns episode seeds: it derives them from the handshake's session_seed
        because it auto resets without python in the loop."""
        return [None] * self.num_envs

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs

    def get_attr(self, attr_name, indices=None):
        # SB3 asks for render_mode while constructing the VecEnv. Unity draws its own
        # window; there is nothing else about an arena python can reach.
        if attr_name == "render_mode":
            return [None] * self.num_envs
        raise NotImplementedError("arenas live in unity")

    def set_attr(self, attr_name, value, indices=None):
        raise NotImplementedError("arenas live in unity")

    def env_method(self, method_name, *args, indices=None, **kwargs):
        raise NotImplementedError("arenas live in unity")
