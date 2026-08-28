#!/usr/bin/env python3
"""Sample every sensor and motor at a fixed period for a fixed duration.

Rows are buffered and written after the run: the sd card must not add jitter to
the sample times it is measuring.

    ./experiments/measure.py [seconds]
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ev3dev2.motor import Motor, OUTPUT_A, OUTPUT_B
from ev3dev2.power import PowerSupply
from ev3dev2.sensor import INPUT_1, INPUT_2, INPUT_3
from ev3dev2.sensor.lego import ColorSensor, UltrasonicSensor

from shared.csv_logger import CsvLogger
from shared.logs import run_prefix

LOG_COLUMNS = ["t", "distance", "left_color", "right_color",
               "left_position", "right_position", "voltage", "current"]

distance_sensor = UltrasonicSensor(INPUT_1)
left_color_sensor = ColorSensor(INPUT_3)
right_color_sensor = ColorSensor(INPUT_2)

left_motor = Motor(OUTPUT_A)
right_motor = Motor(OUTPUT_B)

battery = PowerSupply()

prefix = run_prefix("brick", "measure")

rows = []
start_t = time.monotonic()

started = False
finished = False

while True:
    t = time.monotonic() - start_t

    if t >= 1 and not started:
        started = True

        right_motor.run_direct(duty_cycle_sp=50)
        left_motor.run_direct(duty_cycle_sp=20)

    if t >= 3 and not finished:
        finished = True

        right_motor.stop(stop_action="coast")
        left_motor.stop(stop_action="coast")

    if t >= 5:
        break

    rows.append((
        t,
        distance_sensor.distance_centimeters_continuous,
        0,
        0,
        # left_color_sensor.reflected_light_intensity,
        # right_color_sensor.reflected_light_intensity,
        left_motor.position,
        right_motor.position,
        battery.measured_volts,
        battery.measured_amps,
    ))

logger = CsvLogger(prefix + ".csv", LOG_COLUMNS)
for row in rows:
    logger.log(*row)
logger.close()
