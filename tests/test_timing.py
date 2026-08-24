"""The wall clock split the Stats callback writes, against a fake unity.

Fake sims rather than a build: the point is that the phases add up to the iteration,
which is arithmetic on perf_counter and does not need a simulator to be true.
"""

import time

import numpy as np
import pytest

from stable_baselines3.common.callbacks import BaseCallback

from conftest import ROOT  # noqa: F401  (puts the project on sys.path)
from rl.builder import build
from rl.callbacks import Stats
from rl.vec_sim import PHASES, SimVecEnv
from shared.observation import DIM as OBS_DIM

ARENAS = 4
STEPS = 8
TERMS = ["ProgressReward", "StepPenalty"]


class FakeSim:
    """One unity process, answering instantly and always mid episode."""

    def __init__(self, arenas=ARENAS):
        self.arenas = arenas
        self.action_dim = 2
        self.reward_terms = TERMS
        self.sent = 0
        self.closed = False

    def step_async(self, actions):
        assert len(actions) == self.arenas
        self.sent += 1

    def step_wait(self):
        return self._state()

    def reset(self, randomize_scenario=True, randomize_physics=True):
        return self._state()

    def close(self):
        self.closed = True

    def _state(self):
        from bridge.sim_robot import Step

        # The observation clock has to advance, or Features divides by a zero dt.
        row = lambda: [float(self.sent) * 0.0667] + [0.0] * (OBS_DIM - 1)
        return Step(
            obs=[row() for _ in range(self.arenas)],
            terminal_obs=[[0.0] * OBS_DIM for _ in range(self.arenas)],
            reward=[0.0] * self.arenas,
            terms=[[0.0] * len(TERMS) for _ in range(self.arenas)],
            terminated=[False] * self.arenas,
            truncated=[False] * self.arenas,
            episode=[0] * self.arenas,
            step=[self.sent] * self.arenas,
        )


class Spy(BaseCallback):
    """Keeps every scalar Stats records. The logger only exists once learn() has
    started, so the wrapper goes on from inside a callback of its own."""

    def __init__(self):
        super().__init__()
        self.scalars = {}

    def _on_training_start(self):
        record = self.model.logger.record

        def spy(key, value, exclude=None):
            if not isinstance(value, np.ndarray) or value.size == 1:
                self.scalars[key] = float(value)
            record(key, value, exclude)

        self.model.logger.record = spy

    def _on_step(self):
        return True


@pytest.fixture
def trained():
    """Two iterations, with the scalars of the last one. Two because update and log
    are only measured once a previous rollout has ended."""
    env = SimVecEnv([FakeSim(), FakeSim()])
    model = build("mlp", env, n_steps=STEPS, batch_size=STEPS * env.num_envs)

    spy = Spy()
    model.learn(total_timesteps=2 * STEPS * env.num_envs, callback=[spy, Stats()])
    return env, spy.scalars


def test_env_phases_are_all_recorded(trained):
    _, scalars = trained

    assert set("time/env_%s_s" % phase for phase in PHASES) <= set(scalars)


def test_collect_splits_into_its_phases(trained):
    """No wall clock inside a rollout is unaccounted for."""
    _, s = trained

    parts = sum(s["time/env_%s_s" % phase] for phase in PHASES)
    parts += s["time/stats_step_s"] + s["time/policy_s"]
    assert parts == pytest.approx(s["time/collect_s"], abs=1e-9)


def test_iteration_splits_into_collect_update_and_log(trained):
    """The update and the logging of the previous iteration fill the gap around it."""
    _, s = trained

    parts = s["time/collect_s"] + s["time/update_s"] + s["time/stats_log_s"]
    assert parts == pytest.approx(s["time/iteration_s"], abs=1e-9)


def test_taking_the_timers_resets_them(trained):
    env, _ = trained
    env.step_async(np.zeros((env.num_envs, 2), np.float32))
    env.step_wait()

    assert sum(env.take_timers().values()) > 0.0
    assert env.take_timers() == dict.fromkeys(PHASES, 0.0)


def test_the_timers_measure_the_phase_they_name():
    """A sim that sleeps in step_wait moves wait, and nothing else."""
    slow = FakeSim()
    slow.step_wait = lambda: (time.sleep(0.05), FakeSim.step_wait(slow))[1]
    env = SimVecEnv([slow])
    env.reset()

    env.step_async(np.zeros((env.num_envs, 2), np.float32))
    env.step_wait()
    timers = env.take_timers()

    assert timers["wait"] >= 0.05
    assert timers["send"] < 0.05
    assert timers["decode"] < 0.05
