#!/usr/bin/env python3
"""Drive straight, sampling the left color sensor and both
encoders as fast as the loop allows.

Rows are buffered and written after the run, so the sd card does not add jitter
to the sample times.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ev3dev2.motor import Motor, OUTPUT_A, OUTPUT_B
from ev3dev2.sensor import INPUT_3
from ev3dev2.sensor.lego import ColorSensor

from shared.csv_logger import CsvLogger
from shared.logs import run_prefix

DUTY = 20
DURATION = 10

color_sensor = ColorSensor(INPUT_3)
color_sensor.mode = ColorSensor.MODE_COL_REFLECT

left_motor = Motor(OUTPUT_A)
right_motor = Motor(OUTPUT_B)

prefix = run_prefix("brick", "color_response")

rows = []

left_motor.run_direct(duty_cycle_sp=DUTY)
right_motor.run_direct(duty_cycle_sp=DUTY)

start_t = time.monotonic()
try:
    while True:
        t = time.monotonic() - start_t
        if t >= DURATION:
            break
        rows.append((t, color_sensor.reflected_light_intensity,
                     left_motor.position, right_motor.position))
finally:
    left_motor.stop(stop_action="coast")
    right_motor.stop(stop_action="coast")

logger = CsvLogger(prefix + ".csv", ["t", "color", "left_position", "right_position"])
for row in rows:
    logger.log(*row)
logger.close()

print("%d rows, %.1f Hz" % (len(rows), len(rows) / DURATION))
