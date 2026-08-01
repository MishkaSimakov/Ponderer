from shared.observation import COLUMNS
from shared.policy import Policy

T = COLUMNS.index("t")


class MeasurePolicy(Policy):
    """The duty schedule of experiments/measure.py: idle, both motors on, idle."""

    def __init__(self, start=1.0, stop=3.0):
        self.start = start
        self.stop = stop

    def act(self, obs):
        if self.start <= obs[T] < self.stop:
            return 20, 50

        return 0, 0
