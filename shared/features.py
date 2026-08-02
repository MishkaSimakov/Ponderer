"""Observation to network input, identical on the host and on the brick.

Raw observations carry the episode clock and absolute encoder angles, both of which
grow without bound and cannot be fed to a network. What the policy sees instead is
reflectance and wheel speed, in units that stay bounded for any episode length.

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

NAMES = ["left_color", "right_color", "left_speed", "right_speed"]
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
        return [obs[LEFT_COLOR] / COLOR_SCALE, obs[RIGHT_COLOR] / COLOR_SCALE, 0.0, 0.0]

    def update(self, obs):
        dt = obs[T] - self.t
        left_speed = (obs[LEFT_POSITION] - self.left) / dt
        right_speed = (obs[RIGHT_POSITION] - self.right) / dt

        self.t = obs[T]
        self.left = obs[LEFT_POSITION]
        self.right = obs[RIGHT_POSITION]

        return [obs[LEFT_COLOR] / COLOR_SCALE, obs[RIGHT_COLOR] / COLOR_SCALE,
                left_speed / SPEED_SCALE, right_speed / SPEED_SCALE]
