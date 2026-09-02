"""The step lengths the simulator advances by, against the measurement they come from.

The marginal distribution is the easy half and every sampler gets it right. What these
check is the half that matters: consecutive steps stay consecutive, stretches never span
two runs, and the shape of the measured series survives the resampling.
"""

import random

import numpy as np
import pytest

from bridge.step_clock import STAY, StepClock, load

LAGS = (1, 2, 3, 5)


@pytest.fixture(scope="module")
def measured():
    return load()


def drawn(clock, n):
    return [clock.sample() for _ in range(n)]


def autocorrelation(x, lag):
    x = np.asarray(x)
    return float(np.corrcoef(x[:-lag], x[lag:])[0, 1])


def slow_runs(x, threshold):
    """Lengths of the consecutive stretches spent above the threshold."""
    runs, current = [], 0
    for value in x:
        if value > threshold:
            current += 1
        elif current:
            runs.append(current)
            current = 0
    if current:
        runs.append(current)
    return runs


def test_every_draw_is_a_measured_step(measured):
    every = set(v for run in measured for v in run)

    assert set(drawn(StepClock(measured, seed=0), 2000)) <= every


def test_the_same_seed_replays_the_same_steps(measured):
    assert drawn(StepClock(measured, seed=7), 500) == drawn(StepClock(measured, seed=7), 500)


def test_different_seeds_do_not(measured):
    assert drawn(StepClock(measured, seed=7), 500) != drawn(StepClock(measured, seed=8), 500)


def test_a_stretch_is_the_run_read_in_order():
    clock = StepClock([[1.0, 2.0, 3.0, 4.0]], seed=0, stay=1.0)
    taken = drawn(clock, 4)

    assert taken in ([1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 1.0],
                     [3.0, 4.0, 1.0, 2.0], [4.0, 1.0, 2.0, 3.0])


def test_a_stretch_wraps_round_its_run_rather_than_stopping():
    clock = StepClock([[1.0, 2.0, 3.0]], seed=0, stay=1.0)
    taken = drawn(clock, 9)

    # Three laps of the same three steps, whichever one it started on.
    assert sorted(taken) == [1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0]


def test_a_stretch_never_crosses_from_one_run_into_another():
    """The join between two runs is a step no robot ever took."""
    runs = [[1.0] * 50, [9.0] * 50]
    taken = drawn(StepClock(runs, seed=0, stay=1.0), 500)

    assert len(set(taken)) == 1


def test_never_staying_makes_every_draw_independent(measured):
    taken = drawn(StepClock(measured, seed=3, stay=0.0), 20000)

    assert abs(autocorrelation(taken, 1)) < 0.05


def test_the_stretch_length_matches_the_stay_probability():
    runs = [list(range(1000))]
    clock = StepClock(runs, seed=0, stay=0.9)
    taken = drawn(clock, 50000)

    # A stretch continues while the value goes up by exactly one.
    stretches, current = [], 1
    for before, after in zip(taken, taken[1:]):
        if after == before + 1:
            current += 1
        else:
            stretches.append(current)
            current = 1
    stretches.append(current)

    assert np.mean(stretches) == pytest.approx(1 / (1 - 0.9), rel=0.1)


def test_the_resampled_steps_have_the_measured_distribution(measured):
    every = [v for run in measured for v in run]
    taken = drawn(StepClock(measured, seed=1), 40000)

    assert np.mean(taken) == pytest.approx(np.mean(every), rel=0.02)
    assert np.median(taken) == pytest.approx(np.median(every), rel=0.02)
    assert np.percentile(taken, 95) == pytest.approx(np.percentile(every, 95), rel=0.05)


def test_the_resampled_steps_keep_the_measured_correlation(measured):
    """What independent draws would destroy: a slow step predicts the next one."""
    every = [v for run in measured for v in run]
    taken = drawn(StepClock(measured, seed=1), 40000)

    for lag in LAGS:
        assert autocorrelation(taken, lag) == pytest.approx(
            autocorrelation(every, lag), abs=0.05)


def test_the_resampled_steps_keep_the_measured_slow_stretches(measured):
    every = [v for run in measured for v in run]
    threshold = np.percentile(every, 95)
    taken = drawn(StepClock(measured, seed=1), 40000)

    assert np.mean(slow_runs(taken, threshold)) == pytest.approx(
        np.mean(slow_runs(every, threshold)), rel=0.15)


def test_a_series_of_one_step_still_draws_it():
    assert drawn(StepClock([[0.02]], seed=0), 5) == [0.02] * 5


def test_nothing_to_draw_from_is_refused():
    with pytest.raises(ValueError):
        StepClock([], seed=0)

    with pytest.raises(ValueError):
        StepClock([[]], seed=0)


def test_the_measured_steps_ship_with_the_repo(measured):
    """cloud_train.sh provisions the training host by git pull, so a file left
    untracked would be missing there and nowhere else."""
    assert measured
    assert sum(len(run) for run in measured) > 1000
    assert all(0.0 < v < 1.0 for run in measured for v in run)


def test_the_default_stay_keeps_stretches_worth_having():
    assert 10 <= 1 / (1 - STAY) <= 30


def test_the_clock_draws_from_its_own_stream(measured):
    """Two instances seeded alike must not be steered by anything shared."""
    random.seed(0)
    first = drawn(StepClock(measured, seed=5), 200)
    random.seed(999)
    second = drawn(StepClock(measured, seed=5), 200)

    assert first == second
