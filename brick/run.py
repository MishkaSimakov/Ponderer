#!/usr/bin/env python3
"""Run one policy on the brick.

The ev3dev menu starts a file and passes nothing, so the run is configured here.

    ./brick/run.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.runner import run
from brick.brick_robot import BrickRobot
from shared.policies.experiments.duty_steps import DutyStepsPolicy
from shared.policies.net import latest
from shared.policies.constant import ConstantPolicy

# POLICY = DutyStepsPolicy  # or latest, for the newest export in policy/
# NAME = "duty_steps"  # logs/brick/<NAME>-<utc>.csv

POLICY = latest
NAME = "net"  # logs/brick/<NAME>-<utc>.csv

# POLICY = ConstantPolicy
# NAME = "constant"

STEPS = 1000  # if None, then the policy's own schedule, or until ctrl-c if it has none

policy = POLICY()
robot = BrickRobot()

run(robot, policy, "brick", NAME, STEPS)
