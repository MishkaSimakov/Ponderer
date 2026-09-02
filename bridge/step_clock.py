"""How long a simulated control step lasts, drawn from the brick's own steps.

The brick loop is unpaced, so its step length is a distribution and not a constant, and
the simulator has to reproduce that distribution rather than pick a number. The draws
come from step_durations.csv, which is a measurement and not a fit: nothing is generated
that the hardware did not do.

Drawing each step independently would reproduce the distribution and lose its shape.
Steps are correlated on the brick — a slow one is followed by another about twice as
often as chance would have it, and slow stretches run several steps long — and a robot
that meets one bad step behaves differently from one that meets five. So this is a
stationary bootstrap: with probability STAY the next draw is the step that followed the
last one, otherwise it jumps to a uniformly chosen step. Stretches therefore come out
geometric with mean 1 / (1 - STAY).

Advancing wraps around inside the run it came from and never crosses into another, so
every measured step is equally likely to start a stretch and no stretch contains a
transition that never happened.

One clock per unity process, not per arena: Physics.Simulate covers the whole scene, so
the arenas inside one process advance together and share the step they are given.
"""

import csv
import os
import random

PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "step_durations.csv")

# Mean stretch of 20 steps. Against the measurement this reproduces the autocorrelation
# out to lag 5 and the length of a slow stretch; independent draws give neither.
STAY = 0.95


def load(path=PATH):
    """The measured runs, each a list of step durations in seconds."""
    with open(path) as f:
        rows = list(csv.DictReader(f))

    runs = {}
    for row in rows:
        runs.setdefault(int(row["run"]), []).append(float(row["seconds"]))
    return [runs[key] for key in sorted(runs)]


class StepClock:
    """Step lengths in seconds, resampled from the runs in measurement order."""

    def __init__(self, runs, seed, stay=STAY):
        if not runs or not all(runs):
            raise ValueError("no measured steps to draw from")

        self.runs = runs
        self.stay = stay
        self.rng = random.Random(seed)
        # A jump picks a measured step, so the runs are weighted by their length.
        self.starts = [(r, i) for r, run in enumerate(runs) for i in range(len(run))]
        self.run, self.index = self.rng.choice(self.starts)

    def sample(self):
        """The next step's length, continuing the current stretch or starting one."""
        seconds = self.runs[self.run][self.index]

        if self.rng.random() < self.stay:
            self.index = (self.index + 1) % len(self.runs[self.run])
        else:
            self.run, self.index = self.rng.choice(self.starts)

        return seconds
