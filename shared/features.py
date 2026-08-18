"""Observation to network input, identical on the host and on the brick.

Raw observations carry the episode clock and absolute encoder angles, both of which
grow without bound and cannot be fed to a network. What the policy sees instead is
reflectance and, when USE_SPEED is on, wheel speed, in units that stay bounded for
any episode length.

Speed comes from the observation's own clock, not the host's: a late tick on the
brick is a longer step, and the feature has to see that or simulation and hardware
stop agreeing.
"""

from shared.observation import COLUMNS

T = COLUMNS.index("t")
LEFT_COLOR = COLUMNS.index("left_color")
RIGHT_COLOR = COLUMNS.index("right_color")
LEFT_POSITION = COLUMNS.index("left_position")
RIGHT_POSITION = COLUMNS.index("right_position")

# Set to True to feed wheel speed to the network again.
USE_SPEED = False

NAMES = ["left_color", "right_color"] + (["left_speed", "right_speed"] if USE_SPEED else [])
DIM = len(NAMES)

# Reflected light is already 0..100. Wheel speed is scaled by the motor's no load
# speed, 8 V / 0.46 V/(rad/s) is about 17 rad/s, near 1000 deg/s.
COLOR_SCALE = 100.0
SPEED_SCALE = 1000.0


class Features:
    """Stateful across a step, reset with the episode: speed needs the previous frame."""

    def first(self, obs):
        """First frame of an episode. No previous frame exists, so speed is zero."""
        self.t = obs[T]
        self.left = obs[LEFT_POSITION]
        self.right = obs[RIGHT_POSITION]
        return colors(obs) + ([0.0, 0.0] if USE_SPEED else [])

    def update(self, obs):
        dt = obs[T] - self.t
        speed = [(obs[LEFT_POSITION] - self.left) / dt / SPEED_SCALE,
                 (obs[RIGHT_POSITION] - self.right) / dt / SPEED_SCALE]

        self.t = obs[T]
        self.left = obs[LEFT_POSITION]
        self.right = obs[RIGHT_POSITION]

        return colors(obs) + (speed if USE_SPEED else [])


def colors(obs):
    return [obs[LEFT_COLOR] / COLOR_SCALE, obs[RIGHT_COLOR] / COLOR_SCALE]
