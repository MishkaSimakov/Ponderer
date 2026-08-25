"""train.py against a fake unity, so a whole run can be ended in a subprocess.

    shutdown_driver.py <mode> <tensorboard> <policy> <name>

A mode is a way a run ends: steps reaches --total-steps, keyboard and oserror raise
from inside the env, sigterm prints READY and then waits to be killed.
"""

import os
import sys
import time

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv

import rl.export
import train
from rl.vec_sim import PHASES
from shared.features import DIM

ARENAS = 2
N_STEPS = 16
# Past the first rollout, so the run ends with an update and a Stats log behind it.
RAISE_AT = 20
TOTAL_STEPS = 4 * ARENAS * N_STEPS
TERMS = ("progress", "offset")


class FakeSim(VecEnv):
    """As much of SimVecEnv as PPO and rl.callbacks.Stats read. Episodes never end."""

    def __init__(self, mode):
        self.mode = mode
        self.reward_terms = TERMS
        self.timers = dict.fromkeys(PHASES, 0.0)
        self.steps = 0
        super().__init__(ARENAS,
                         spaces.Box(-np.inf, np.inf, (DIM,), np.float32),
                         spaces.Box(-1.0, 1.0, (2,), np.float32))

    def reset(self):
        return np.zeros((ARENAS, DIM), np.float32)

    def step_async(self, actions):
        pass

    def step_wait(self):
        self.steps += 1
        if self.steps == RAISE_AT:
            if self.mode == "keyboard":
                raise KeyboardInterrupt
            if self.mode == "oserror":
                raise OSError("unity went away")
            if self.mode == "sigterm":
                print("READY", flush=True)

        # Idle rather than spin while the parent gets around to sending the signal.
        if self.mode == "sigterm" and self.steps >= RAISE_AT:
            time.sleep(0.01)

        obs = np.full((ARENAS, DIM), self.steps, np.float32)
        infos = [{"reward_terms": np.zeros(len(TERMS), np.float32)} for _ in range(ARENAS)]
        return obs, np.ones(ARENAS, np.float32), np.zeros(ARENAS, bool), infos

    def take_timers(self):
        return dict(self.timers)

    def close(self):
        pass

    def seed(self, seed=None):
        return [None] * ARENAS

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * ARENAS

    def get_attr(self, attr_name, indices=None):
        if attr_name == "render_mode":
            return [None] * ARENAS
        raise NotImplementedError("arenas live in unity")

    def set_attr(self, attr_name, value, indices=None):
        raise NotImplementedError("arenas live in unity")

    def env_method(self, method_name, *args, indices=None, **kwargs):
        raise NotImplementedError("arenas live in unity")


def main():
    mode, tensorboard, policy, name = sys.argv[1:]

    train.TENSORBOARD = tensorboard
    train.UNITY_LOGS = os.path.join(tensorboard, "unity")
    rl.export.ROOT = policy
    train.make = lambda **kwargs: FakeSim(mode)

    # Only the steps mode is meant to reach the limit; the others end before it.
    total = TOTAL_STEPS if mode == "steps" else 10 ** 9
    sys.argv = ["train.py", "--arch", "mlp", "--run-name", name, "--hidden", "8",
                "--n-steps", str(N_STEPS), "--batch-size", str(N_STEPS),
                "--total-steps", str(total)]
    train.main()


if __name__ == "__main__":
    main()
