"""The loop both entrypoints share, driven by a fake robot and a fake clock."""

import csv
import glob
import os

import pytest

from shared import logs, runner
from shared.policy import Policy
from shared.runner import EpisodeEnded, LOG_COLUMNS, run, steps_for

OBS_DIM = len(LOG_COLUMNS) - 2


class FakeRobot:
    """Observation k is a row of k, so a log row shows which one it was built from."""

    def __init__(self, fail_at=None, end_at=None):
        self.fail_at = fail_at
        self.end_at = end_at
        self.actions = []
        self.stopped = False
        self.tick = 0

    def reset(self):
        return [0.0] * OBS_DIM

    def step(self, action):
        self.actions.append(tuple(action))
        self.tick += 1
        if self.tick == self.fail_at:
            raise OSError("brick went away")
        if self.tick == self.end_at:
            raise EpisodeEnded("episode over")
        return [float(self.tick)] * OBS_DIM

    def stop(self):
        self.stopped = True


class EchoPolicy(Policy):
    """Action derived from the observation, so the pairing in a row is checkable."""

    def act(self, obs):
        return obs[0] + 0.5, -obs[0] - 0.5


class TimedPolicy(Policy):
    duration = 9.0

    def act(self, obs):
        return 0.0, 0.0


@pytest.fixture
def log_root(tmp_path, monkeypatch):
    monkeypatch.setattr(logs, "ROOT", str(tmp_path))
    return tmp_path


def rows(log_root, source="sim"):
    written = glob.glob(os.path.join(str(log_root), source, "*.csv"))
    assert len(written) == 1
    with open(written[0]) as f:
        return list(csv.reader(f))


def test_steps_given_wins():
    assert steps_for(TimedPolicy(), 1.0 / 15.0, 100) == 100


def test_steps_come_from_the_policy_schedule():
    assert steps_for(TimedPolicy(), 1.0 / 15.0) == 135


def test_a_policy_without_a_schedule_runs_unbounded():
    assert steps_for(EchoPolicy(), 1.0 / 15.0) is None


def test_header_is_the_log_columns(log_root):
    run(FakeRobot(), EchoPolicy(), "sim", "echo", 3)

    assert rows(log_root)[0] == LOG_COLUMNS


def test_action_is_logged_with_the_observation_it_was_computed_from(log_root):
    robot = FakeRobot()
    run(robot, EchoPolicy(), "sim", "echo", 4)

    logged = rows(log_root)[1:]
    assert len(logged) == 4
    for tick, row in enumerate(logged):
        assert [float(v) for v in row] == [tick + 0.5, -tick - 0.5] + [float(tick)] * OBS_DIM

    # And the robot was handed the same action that row was written with.
    assert robot.actions == [(t + 0.5, -t - 0.5) for t in range(4)]


def test_every_row_is_as_wide_as_the_header(log_root):
    run(FakeRobot(), EchoPolicy(), "sim", "echo", 3)

    written = rows(log_root)
    assert all(len(row) == len(LOG_COLUMNS) for row in written)


def test_an_unbounded_run_stops_when_the_episode_ends(log_root):
    robot = FakeRobot(end_at=3)
    run(robot, EchoPolicy(), "sim", "echo", None)

    assert robot.stopped
    assert len(rows(log_root)) == 1 + 3


def test_a_fixed_length_run_refuses_to_be_cut_short(log_root):
    robot = FakeRobot(end_at=3)

    with pytest.raises(EpisodeEnded):
        run(robot, EchoPolicy(), "sim", "echo", 10)

    assert robot.stopped


def test_the_log_survives_the_robot_going_away(log_root):
    robot = FakeRobot(fail_at=2)
    run(robot, EchoPolicy(), "sim", "echo", 10)

    assert robot.stopped
    assert len(rows(log_root)) == 1 + 2


def test_the_robot_is_stopped_on_interrupt(log_root):
    class Interrupting(Policy):
        def act(self, obs):
            raise KeyboardInterrupt

    robot = FakeRobot()
    run(robot, Interrupting(), "sim", "echo", 10)

    assert robot.stopped


def test_pacing_left_to_the_robot_counts_no_overruns(log_root):
    assert run(FakeRobot(), EchoPolicy(), "sim", "echo", 5) == 0


def test_a_period_the_loop_cannot_keep_counts_every_step(log_root):
    assert run(FakeRobot(), EchoPolicy(), "sim", "echo", 5, period=0.0) == 5


def test_a_period_the_loop_can_keep_counts_none(log_root):
    assert run(FakeRobot(), EchoPolicy(), "sim", "echo", 3, period=0.01) == 0


def test_only_known_sources_are_logged(log_root):
    with pytest.raises(ValueError):
        run(FakeRobot(), EchoPolicy(), "host", "echo", 1)
