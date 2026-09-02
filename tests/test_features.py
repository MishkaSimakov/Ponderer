"""What the network is actually handed, on both sides of the transfer.

The same code runs on the host and on the brick, so a mistake here is not a simulation
bug or a hardware bug but a silent disagreement between them.
"""

import math

import pytest

from bridge.step_clock import load
from shared import features
from shared.features import (DIM, DT_CLIP, DT_REF, NAMES, Features, USE_DT, scaled_dt)
from shared.observation import COLUMNS, DIM as OBS_DIM

T = COLUMNS.index("t")
LEFT_COLOR = COLUMNS.index("left_color")
RIGHT_COLOR = COLUMNS.index("right_color")
LEFT_POSITION = COLUMNS.index("left_position")
RIGHT_POSITION = COLUMNS.index("right_position")


def observation(t, colors=(0.0, 0.0), positions=(0.0, 0.0)):
    obs = [0.0] * OBS_DIM
    obs[T] = t
    obs[LEFT_COLOR], obs[RIGHT_COLOR] = colors
    obs[LEFT_POSITION], obs[RIGHT_POSITION] = positions
    return obs


def test_the_feature_row_is_as_wide_as_the_names():
    row = Features()
    assert len(row.first(observation(0.0))) == DIM == len(NAMES)
    assert len(row.update(observation(DT_REF))) == DIM


def test_colors_are_scaled_into_the_unit_range():
    row = Features().first(observation(0.0, colors=(100.0, 50.0)))

    assert row[0] == 1.0
    assert row[1] == 0.5


def test_the_first_frame_reads_as_a_step_of_reference_length():
    """No previous frame exists, so the neutral value stands in for one."""
    assert Features().first(observation(0.0))[-1] == 0.0
    assert scaled_dt(DT_REF) == 0.0


def test_dt_is_the_step_measured_against_the_reference():
    row = Features()
    row.first(observation(0.0))

    assert row.update(observation(DT_REF))[-1] == pytest.approx(0.0)
    assert row.update(observation(3 * DT_REF))[-1] == pytest.approx(math.log(2.0))


def test_dt_comes_from_the_observation_clock_and_not_a_constant():
    """A late tick on the brick is a longer step, and the network has to be told."""
    slow, quick = Features(), Features()
    slow.first(observation(0.0))
    quick.first(observation(0.0))

    assert slow.update(observation(0.05))[-1] > quick.update(observation(0.01))[-1]


def test_dt_is_the_step_that_just_ended_not_the_one_ahead():
    row = Features()
    row.first(observation(0.0))
    first = row.update(observation(0.04))[-1]
    second = row.update(observation(0.05))[-1]

    assert first == pytest.approx(scaled_dt(0.04))
    assert second == pytest.approx(scaled_dt(0.01))


def test_a_step_far_longer_than_the_reference_is_bounded():
    assert scaled_dt(DT_REF * math.exp(10.0)) == DT_CLIP


def test_a_step_far_shorter_than_the_reference_is_bounded():
    assert scaled_dt(DT_REF * math.exp(-10.0)) == -DT_CLIP


def test_no_measured_step_is_clipped():
    """The bound catches a step longer than any that has been seen. It must not be
    trimming the ones that have, or the network cannot tell them apart."""
    every = [v for run in load() for v in run]

    assert -DT_CLIP < scaled_dt(min(every)) < scaled_dt(max(every)) < DT_CLIP


def test_speed_is_measured_against_the_observation_clock(monkeypatch):
    monkeypatch.setattr(features, "USE_SPEED", True)
    row = Features()
    row.first(observation(0.0, positions=(0.0, 0.0)))

    left, right = row.update(observation(0.037, positions=(37.0, -74.0)))[2:4]

    assert left == pytest.approx(37.0 / 0.037 / features.SPEED_SCALE)
    assert right == pytest.approx(-74.0 / 0.037 / features.SPEED_SCALE)


def test_dt_goes_last_so_the_speed_flag_cannot_reorder_it():
    assert not USE_DT or NAMES[-1] == "dt"
