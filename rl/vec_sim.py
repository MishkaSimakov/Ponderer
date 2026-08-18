"""Unity as a stable-baselines3 VecEnv.

Unity already auto resets, so its response is exactly what a VecEnv must return:
obs belongs to the next episode and the finished one's last observation travels
beside it. SB3 reads that last observation from infos["terminal_observation"] and
bootstraps the value function only when "TimeLimit.truncated" is also set, which
is what separating terminated from truncated is for.

One VecEnv spans every unity process: the arenas of instance 0 come first, then
those of instance 1, and so on.
"""

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv

from bridge.launcher import Unity
from bridge.sim_robot import Simulation, Step
from shared.features import DIM, Features

# Actions live in [-1, 1] so the gaussian policy is symmetric; unity takes percent.
DUTY = 100.0


def make(env=None, num_envs=1, arenas=1, base_port=5005, seed=0, graphics=False,
         timeout=60.0, log_prefix=None, extra=(), randomize_scenario=True,
         randomize_physics=True):
    """Launch num_envs players from env, or attach to that many running ones."""
    ports = [base_port + i for i in range(num_envs)]
    processes = [] if env is None else [
        Unity(env, port, arenas, graphics=graphics, extra=extra,
              log=None if log_prefix is None else "%s-%d.log" % (log_prefix, port))
        for port in ports]

    try:
        # Instances must not replay the same episodes, so each gets its own seed root.
        sims = [Simulation(port=port, session_seed=seed + i, timeout=timeout)
                for i, port in enumerate(ports)]
    except Exception:
        for process in processes:
            process.kill()
        raise

    return SimVecEnv(sims, processes, randomize_scenario=randomize_scenario,
                     randomize_physics=randomize_physics)


class SimVecEnv(VecEnv):
    def __init__(self, sims, processes=(), randomize_scenario=True, randomize_physics=True):
        self.sims = list(sims)
        self.processes = list(processes)
        self.randomize_scenario = randomize_scenario
        self.randomize_physics = randomize_physics
        self.features = None
        # Every instance runs the same build, so the term names are the same too.
        self.reward_terms = self.sims[0].reward_terms

        super().__init__(
            sum(sim.arenas for sim in self.sims),
            spaces.Box(-np.inf, np.inf, (DIM,), np.float32),
            spaces.Box(-1.0, 1.0, (self.sims[0].action_dim,), np.float32),
        )

    def reset(self):
        # Unity keeps these flags for the auto resets it does without python.
        obs = [o for sim in self.sims
               for o in sim.reset(randomize_scenario=self.randomize_scenario,
                                  randomize_physics=self.randomize_physics).obs]
        self.features = [Features() for _ in range(self.num_envs)]
        return np.array([f.first(o) for f, o in zip(self.features, obs)], np.float32)

    def step_async(self, actions):
        actions = actions * DUTY
        start = 0
        for sim in self.sims:
            sim.step_async(actions[start:start + sim.arenas])
            start += sim.arenas

    def step_wait(self):
        states = [sim.step_wait() for sim in self.sims]
        state = Step(*[[v for s in states for v in getattr(s, field)]
                       for field in Step._fields])

        obs = np.empty((self.num_envs, DIM), np.float32)
        dones = np.empty(self.num_envs, bool)
        infos = [{} for _ in range(self.num_envs)]

        for i in range(self.num_envs):
            # What each reward component contributed to rewards[i], for logging only.
            infos[i]["reward_terms"] = state.terms[i]
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
        # quit first, kill second: a player that obeys the command exits on its own.
        for sim in self.sims:
            sim.close()
        for process in self.processes:
            process.stop()

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
