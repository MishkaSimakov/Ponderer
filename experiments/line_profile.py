#!/usr/bin/env python3
"""Sweep the color sensor across the line with the medium motor and record
reflectance against encoder angle.

Forward SWEEP degrees, then back to the start, at a constant slow speed.
Rows are buffered and written after the run, so the sd card does not add jitter
to the sample times.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ev3dev2.motor import MediumMotor, OUTPUT_A
from ev3dev2.sensor import INPUT_1
from ev3dev2.sensor.lego import ColorSensor

from shared.csv_logger import CsvLogger
from shared.logs import run_prefix

SWEEP = 375
SPEED = 125  # deg/s

color_sensor = ColorSensor(INPUT_1)
color_sensor.mode = ColorSensor.MODE_COL_REFLECT

motor = MediumMotor(OUTPUT_A)
motor.ramp_up_sp = 0
motor.ramp_down_sp = 0
motor.position = 0

prefix = run_prefix("brick", "line_profile")

rows = []
start_t = time.monotonic()


def sweep(speed, done):
    motor.run_forever(speed_sp=speed)
    while True:
        position = motor.position
        rows.append((time.monotonic() - start_t, position,
                     color_sensor.reflected_light_intensity, speed > 0))
        if done(position):
            break
    motor.stop(stop_action="hold")


try:
    sweep(-SPEED, lambda position: position <= -SWEEP)
    sweep(SPEED, lambda position: position >= 0)
finally:
    motor.stop(stop_action="coast")

logger = CsvLogger(prefix + ".csv", ["t", "position", "color", "forward"])
for row in rows:
    logger.log(*row)
logger.close()

duration = rows[-1][0]
print("%d rows, %.1f Hz" % (len(rows), len(rows) / duration))
