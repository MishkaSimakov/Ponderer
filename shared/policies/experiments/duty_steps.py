from shared.action import VOLTS
from shared.observation import COLUMNS
from shared.policy import Policy

T = COLUMNS.index("t")

# Fractions of the voltage cap, SECONDS on each, both motors together.
FRACTIONS = [0.0, 0.25, 0.5, 0.75, 1.0, 0.75, 0.5, 0.25, 0.0]
SECONDS = 1.0


class DutyStepsPolicy(Policy):
    """Both motors climb a voltage staircase and come back down."""

    def __init__(self, fractions=FRACTIONS, seconds=SECONDS):
        self.volts = [fraction * VOLTS for fraction in fractions]
        self.seconds = seconds
        self.duration = seconds * len(self.volts)

    def act(self, obs):
        index = int(obs[T] / self.seconds)
        if index >= len(self.volts):
            return 0.0, 0.0
        return self.volts[index], self.volts[index]
