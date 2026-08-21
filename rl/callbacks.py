"""Per rollout distributions, not only their means.

ML-Agents logs the episode return as a histogram next to the scalar, because the mean
hides the shape. A line follower is bimodal: it either leaves the line at once or runs
to the step limit, and the mean sits between the two humps where nothing ever happens.
The same argument applies to what the robot actually did and saw, which is why the
actions and the features get histograms too.

SB3's tensorboard writer turns any recorded array into add_histogram; the other output
formats only understand scalars, hence the exclude.
"""

import time

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from shared.action import VOLTS
from shared.features import DIM, NAMES

HISTOGRAM = ("stdout", "log", "json", "csv")
ACTIONS = ("volts_left", "volts_right")


class Stats(BaseCallback):
    def __init__(self):
        super().__init__()
        self.returns = []
        self.lengths = []
        self.terminated = []
        self.rewards = []
        self.terms = []
        self.pending = None
        self.actions = []
        self.features = []

    def _on_training_start(self):
        self.names = self.training_env.unwrapped.reward_terms
        # An episode spans rollouts, so the accumulator outlives them.
        self.pending = np.zeros((self.training_env.num_envs, len(self.names)), np.float32)
        self.clock = time.perf_counter()
        self.timesteps = self.num_timesteps

    def _on_step(self):
        self.rewards.append(np.asarray(self.locals["rewards"], np.float32))
        self.pending += np.asarray([info["reward_terms"] for info in self.locals["infos"]],
                                   np.float32)
        self.actions.append(np.asarray(self.locals["clipped_actions"], np.float32))
        # Frame stacking widens the observation; the newest frame is last.
        self.features.append(np.asarray(self.locals["new_obs"], np.float32)[:, -DIM:])

        for i, (done, info) in enumerate(zip(self.locals["dones"], self.locals["infos"])):
            if not done:
                continue
            # VecMonitor put the episode's return and length here.
            self.returns.append(info["episode"]["r"])
            self.lengths.append(info["episode"]["l"])
            self.terminated.append(0.0 if info.get("TimeLimit.truncated") else 1.0)
            # The terminal step is already in pending, so the row is the episode's total.
            self.terms.append(self.pending[i].copy())
            self.pending[i] = 0.0

        return True

    def _on_rollout_end(self):
        record = self.logger.record

        # SB3's time/fps is the average since the run started, so a one time change in
        # throughput decays into it instead of showing up where it happened.
        now = time.perf_counter()
        record("time/rollout_fps", (self.num_timesteps - self.timesteps) / (now - self.clock))
        self.clock = now
        self.timesteps = self.num_timesteps

        if self.returns:
            record("rollout/ep_rew_hist", np.array(self.returns, np.float32), exclude=HISTOGRAM)
            record("rollout/ep_len_hist", np.array(self.lengths, np.float32), exclude=HISTOGRAM)
            record("rollout/terminated_frac", float(np.mean(self.terminated)))
            record("rollout/episodes", len(self.returns))

        rewards = np.concatenate(self.rewards)
        record("rollout/reward_hist", rewards, exclude=HISTOGRAM)

        # Per term, what it contributed to an episode's return: the means add up to
        # rollout/ep_rew_mean, so the terms are readable against each other and against
        # the score they explain. The histogram is over episodes.
        if self.terms:
            terms = np.array(self.terms, np.float32)
            for i, name in enumerate(self.names):
                record("reward/" + name, float(terms[:, i].mean()))
                record("reward_hist/" + name, terms[:, i], exclude=HISTOGRAM)

        # Volts as unity receives it, so saturation is readable against the +-VOLTS clamp.
        actions = np.concatenate(self.actions) * VOLTS
        for i, name in enumerate(ACTIONS):
            record("action/" + name, actions[:, i], exclude=HISTOGRAM)
        record("action/saturated_frac", float(np.mean(np.abs(actions) >= VOLTS)))

        # What the network saw. These are the distributions the brick has to reproduce.
        features = np.concatenate(self.features)
        for i, name in enumerate(NAMES):
            record("features/" + name, features[:, i], exclude=HISTOGRAM)

        values = self.model.rollout_buffer.values.flatten()
        record("train/value_estimate", float(values.mean()))
        record("train/value_hist", values, exclude=HISTOGRAM)

        for collected in (self.returns, self.lengths, self.terminated,
                          self.rewards, self.terms, self.actions, self.features):
            collected.clear()
