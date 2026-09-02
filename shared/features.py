"""Observation to network input, identical on the host and on the brick.

Raw observations carry the episode clock and absolute encoder angles, both of which
grow without bound and cannot be fed to a network. What the policy sees instead is
reflectance, wheel speed when USE_SPEED is on, and how long the previous step took
when USE_DT is on, in units that stay bounded for any episode length.

Both derived features come from the observation's own clock, not the host's: a late
tick on the brick is a longer step, and the feature has to see that or simulation and
hardware stop agreeing.

dt is the step that has just ended, never the one the action about to be chosen will be
held for: that one is decided by the computation that has not happened yet. It says how
long the last action stood, and it predicts the next step only through the correlation
between neighbouring steps.
"""

from math import log

from shared.observation import COLUMNS

T = COLUMNS.index("t")
LEFT_COLOR = COLUMNS.index("left_color")
RIGHT_COLOR = COLUMNS.index("right_color")
LEFT_POSITION = COLUMNS.index("left_position")
RIGHT_POSITION = COLUMNS.index("right_position")

# Set to True to feed wheel speed to the network again.
USE_SPEED = False

# Set to False to hide the step length from the network.
USE_DT = True

NAMES = (["left_color", "right_color"]
         + (["left_speed", "right_speed"] if USE_SPEED else [])
         + (["dt"] if USE_DT else []))
DIM = len(NAMES)

# Reflected light is already 0..100. Wheel speed is scaled by the motor's no load
# speed, 8 V / 0.46 V/(rad/s) is about 17 rad/s, near 1000 deg/s.
COLOR_SCALE = 100.0
SPEED_SCALE = 1000.0

# The step dt is measured against, seconds, and the bound on the log ratio. Frozen:
# they decide what a trained weight means, so nothing derives them from whatever the
# loop currently costs, and making the loop faster must not move them.
DT_REF = 0.02
DT_CLIP = 1.5


class Features:
    """Stateful across a step, reset with the episode: the derived features need the
    previous frame."""

    def first(self, obs):
        """First frame of an episode. No previous frame exists, so speed is zero and
        the step reads as one of reference length."""
        self.t = obs[T]
        self.left = obs[LEFT_POSITION]
        self.right = obs[RIGHT_POSITION]

        row = colors(obs)
        if USE_SPEED:
            row += [0.0, 0.0]
        if USE_DT:
            row += [0.0]
        return row

    def update(self, obs):
        dt = obs[T] - self.t
        row = colors(obs)

        if USE_SPEED:
            row += [(obs[LEFT_POSITION] - self.left) / dt / SPEED_SCALE,
                    (obs[RIGHT_POSITION] - self.right) / dt / SPEED_SCALE]
        if USE_DT:
            row += [scaled_dt(dt)]

        self.t = obs[T]
        self.left = obs[LEFT_POSITION]
        self.right = obs[RIGHT_POSITION]

        return row


def colors(obs):
    return [obs[LEFT_COLOR] / COLOR_SCALE, obs[RIGHT_COLOR] / COLOR_SCALE]


def scaled_dt(dt):
    """A step length as a bounded log ratio to DT_REF.

    Log because the quantity is positive and long tailed: twice the reference reads as
    0.69 and half of it as -0.69, so the steps that make up the bulk of a run keep their
    resolution instead of being crushed against a rare slow one. The bound is there so a
    step longer than any that has been measured cannot blow the input up; it is not
    meant to compress the range that has been.

    math.log and not numpy.log: one numpy call costs about 240 us on the brick and this
    runs inside every step.
    """
    return max(-DT_CLIP, min(DT_CLIP, log(dt / DT_REF)))
