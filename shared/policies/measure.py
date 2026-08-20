from shared.observation import COLUMNS
from shared.policy import Policy

T = COLUMNS.index("t")


class MeasurePolicy(Policy):
    """The schedule of experiments/measure.py: idle, both motors on, idle.

    In volts: that run's 20 and 50 percent duty at the 8.6 V it measured."""

    def __init__(self, start=1.0, stop=3.0):
        self.start = start
        self.stop = stop

    def act(self, obs):
        if self.start <= obs[T] < self.stop:
            return 1.7, 4.3

        return 0, 0
