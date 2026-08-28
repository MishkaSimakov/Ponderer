#!/usr/bin/env python3
"""How long each part of one control step takes on the brick.

The parts are shared.runner's, in the order they would run if the action were applied
at a fixed offset after the reading: observe, act, log, then the two halves of the
write. Their sum is what the period has to cover, and the offset has to clear
observe + act + log.

The clock is measured too, because every number here costs two of its calls.

One row per step in logs/brick/<NAME>-<utc>.csv, written after the loop so that
writing it does not land inside a step. The row the loop logs is a real run's row,
so its cost is a real run's cost.

    ./experiments/loop_timing.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brick.brick_robot import BrickRobot
from shared.csv_logger import CsvLogger
from shared.logs import run_prefix
from shared.runner import LOG_COLUMNS
from shared.policies.net import latest
from shared.policies.constant import ConstantPolicy

# POLICY = ConstantPolicy
# NAME = "loop_timing_constant"

POLICY = latest
NAME = "loop_timing"  # logs/brick/<NAME>-<utc>.csv

STEPS = 200
FREQUENCY = 20.0
CLOCK_CALLS = 10000

PHASES = ["observe", "act", "log", "volts", "duty", "body"]

period = 1.0 / FREQUENCY
policy = POLICY()
robot = BrickRobot(period)

start = time.monotonic()
for _ in range(CLOCK_CALLS):
    time.monotonic()
clock = (time.monotonic() - start) / CLOCK_CALLS
print("clock %.1f us per call, %.1f us per timed phase" % (1e6 * clock, 2e6 * clock))

# What the loop logs, so that the timed write is the write a real run does.
prefix = run_prefix("brick", NAME)
row_log = prefix + "-rows.csv"
rows = CsvLogger(row_log, LOG_COLUMNS)
timings = []
overruns = 0

obs = robot.reset()
deadline = time.monotonic()

try:
    for _ in range(STEPS):
        t0 = time.monotonic()
        obs = robot._observe()
        t1 = time.monotonic()
        action = policy.act(obs)
        t2 = time.monotonic()
        rows.log(action[0], action[1], *obs)
        t3 = time.monotonic()
        volts = robot.battery.measured_volts
        t4 = time.monotonic()
        robot.left_motor.duty_cycle_sp = robot._duty(action[0], volts)
        robot.right_motor.duty_cycle_sp = robot._duty(action[1], volts)
        t5 = time.monotonic()

        timings.append([t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4, t5 - t0])

        deadline += period
        lag = deadline - time.monotonic()
        if lag > 0:
            time.sleep(lag)
        else:
            overruns += 1
            deadline = time.monotonic()
except KeyboardInterrupt:
    print("interrupted")
finally:
    robot.stop()
    rows.close()

log = prefix + ".csv"
logger = CsvLogger(log, PHASES)
for timing in timings:
    logger.log(*timing)
logger.close()

print("%d steps at %.0f Hz, %d overruns" % (len(timings), FREQUENCY, overruns))
print("%-8s %8s %8s %8s %8s" % ("ms", "mean", "p50", "p95", "max"))
for i, phase in enumerate(PHASES):
    ordered = sorted(timing[i] for timing in timings)
    print("%-8s %8.2f %8.2f %8.2f %8.2f" % (
        phase,
        1e3 * sum(ordered) / len(ordered),
        1e3 * ordered[len(ordered) // 2],
        1e3 * ordered[int(0.95 * (len(ordered) - 1))],
        1e3 * ordered[-1]))
print("rows %s, timings %s" % (row_log, log))
