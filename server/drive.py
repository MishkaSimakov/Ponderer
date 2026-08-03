#!/usr/bin/env python3
"""Drive the robot with an exported policy.

    python3 server/drive.py --steps 400

Writes logs/brick/drive-<utc>.csv in the schema arena.py writes in simulation, so the
two runs can be compared row by row.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.brick_robot import BrickRobot
from shared.csv_logger import CsvLogger
from shared.logs import run_prefix
from shared.policies.net import NetPolicy, load, newest
from shared.runner import LOG_COLUMNS, run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--frequency", type=float, default=20.0)
    parser.add_argument("--delay", type=float, default=3.0, help="seconds before moving")
    parser.add_argument("--policy", default=None, help="default: newest in policy/")
    args = parser.parse_args()

    path = args.policy or newest()
    policy = NetPolicy(load(path))
    print("policy %s, arch %s, %d steps at %g Hz"
          % (path, policy.arch, args.steps, args.frequency))

    robot = BrickRobot(1.0 / args.frequency)
    log = run_prefix("brick", "drive") + ".csv"
    logger = CsvLogger(log, LOG_COLUMNS)

    print("moving in %g s, ctrl-c to stop" % args.delay)
    time.sleep(args.delay)

    try:
        # period=None: BrickRobot.step does the waiting, in the right place.
        run(robot, policy, logger, args.steps)
    finally:
        robot.stop()
        logger.close()

    print("%d steps, %d overruns, log %s" % (args.steps, robot.overruns, log))


if __name__ == "__main__":
    main()
