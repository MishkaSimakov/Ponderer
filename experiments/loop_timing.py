#!/usr/bin/env python3
"""How long each part of one control step takes on the brick.

The phases are the real loop's, in the real loop's order. shared.runner calls act and
log, then brick.BrickRobot.step reads the battery and converts both duties, sleeps to
the deadline, writes the two duty cycles and observes. So everything except the write
and the observe happens before the deadline, and body is the work one period covers.

act is split into the two halves NetPolicy.act runs: building the features from the
observation, and the forward pass. The second is the inference number.

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

import numpy as np

from brick.brick_robot import BrickRobot
from shared.csv_logger import CsvLogger
from shared.logs import run_prefix
from shared.policies.net import NetPolicy, latest
from shared.runner import LOG_COLUMNS

POLICY = latest
NAME = "loop_timing"  # logs/brick/<NAME>-<utc>.csv

STEPS = 200
FREQUENCY = 20.0
CLOCK_CALLS = 10000

PHASES = ["features", "net", "log", "volts", "duty", "write", "observe", "body"]

period = 1.0 / FREQUENCY
policy = POLICY()
if not isinstance(policy, NetPolicy):
    raise TypeError("the features/net split needs a NetPolicy, got %s" % type(policy))
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
started = False
deadline = time.monotonic()

try:
    for _ in range(STEPS):
        t0 = time.monotonic()
        raw = policy.features.update(obs) if started else policy.features.first(obs)
        x = np.asarray(raw, np.float32)
        started = True
        t1 = time.monotonic()
        action = policy.act_features(x)
        t2 = time.monotonic()
        rows.log(action[0], action[1], *obs)
        t3 = time.monotonic()
        volts = robot.battery.measured_volts
        t4 = time.monotonic()
        left = robot._duty(action[0], volts)
        right = robot._duty(action[1], volts)
        t5 = time.monotonic()

        deadline += period
        lag = deadline - time.monotonic()
        if lag > 0:
            time.sleep(lag)
        else:
            overruns += 1
            deadline = time.monotonic()

        t6 = time.monotonic()
        robot.left_motor.duty_cycle_sp = left
        robot.right_motor.duty_cycle_sp = right
        t7 = time.monotonic()
        obs = robot._observe()
        t8 = time.monotonic()

        timings.append([t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4, t7 - t6, t8 - t7,
                        (t5 - t0) + (t8 - t6)])
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
