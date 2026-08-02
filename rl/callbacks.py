"""Per rollout distributions, not only their means.

ML-Agents logs the episode return as a histogram next to the scalar, because the mean
hides the shape. A line follower is bimodal: it either leaves the line at once or runs
to the step limit, and the mean sits between the two humps where nothing ever happens.
The same argument applies to what the robot actually did and saw, which is why the
actions and the features get histograms too.

SB3's tensorboard writer turns any recorded array into add_histogram; the other output
formats only understand scalars, hence the exclude.
"""

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from rl.vec_sim import DUTY
from shared.features import DIM, NAMES

HISTOGRAM = ("stdout", "log", "json", "csv")
ACTIONS = ("duty_left", "duty_right")


class Stats(BaseCallback):
    def __init__(self):
        super().__init__()
        self.returns = []
        self.lengths = []
        self.terminated = []
        self.rewards = []
        self.actions = []
        self.features = []

    def _on_step(self):
        self.rewards.append(np.asarray(self.locals["rewards"], np.float32))
        self.actions.append(np.asarray(self.locals["clipped_actions"], np.float32))
        # Frame stacking widens the observation; the newest frame is last.
        self.features.append(np.asarray(self.locals["new_obs"], np.float32)[:, -DIM:])

        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if not done:
                continue
            # VecMonitor put the episode's return and length here.
            self.returns.append(info["episode"]["r"])
            self.lengths.append(info["episode"]["l"])
            self.terminated.append(0.0 if info.get("TimeLimit.truncated") else 1.0)

        return True

    def _on_rollout_end(self):
        record = self.logger.record

        if self.returns:
            record("rollout/ep_rew_hist", np.array(self.returns, np.float32), exclude=HISTOGRAM)
            record("rollout/ep_len_hist", np.array(self.lengths, np.float32), exclude=HISTOGRAM)
            record("rollout/terminated_frac", float(np.mean(self.terminated)))
            record("rollout/episodes", len(self.returns))

        record("rollout/reward_hist", np.concatenate(self.rewards), exclude=HISTOGRAM)

        # Duty as unity receives it, so saturation is readable against the +-100 clamp.
        actions = np.concatenate(self.actions) * DUTY
        for i, name in enumerate(ACTIONS):
            record("action/" + name, actions[:, i], exclude=HISTOGRAM)
        record("action/saturated_frac", float(np.mean(np.abs(actions) >= DUTY)))

        # What the network saw. These are the distributions the brick has to reproduce.
        features = np.concatenate(self.features)
        for i, name in enumerate(NAMES):
            record("features/" + name, features[:, i], exclude=HISTOGRAM)

        values = self.model.rollout_buffer.values.flatten()
        record("train/value_estimate", float(values.mean()))
        record("train/value_hist", values, exclude=HISTOGRAM)

        for collected in (self.returns, self.lengths, self.terminated,
                          self.rewards, self.actions, self.features):
            collected.clear()
