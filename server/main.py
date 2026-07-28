#!/usr/bin/env python3

from ev3dev2.motor import Motor, OUTPUT_A, OUTPUT_B, SpeedPercent
from ev3dev2.sensor import INPUT_1, INPUT_2, INPUT_3
from ev3dev2.sensor.lego import ColorSensor, UltrasonicSensor

from shared.csv_logger import CsvLogger
from shared.logs import run_prefix

import time

frequency = 20  # 20 Hz
period = 1 / frequency

overruns = 0
start_t = time.monotonic()
next_t = start_t

distance_sensor = UltrasonicSensor(INPUT_1)
left_color_sensor = ColorSensor(INPUT_3)
right_color_sensor = ColorSensor(INPUT_2)

left_motor = Motor(OUTPUT_A)
right_motor = Motor(OUTPUT_B)

prefix = run_prefix("brick", "main")
logger = CsvLogger(prefix + ".csv", ["t", "distance", "left_color", "right_color", "left_position", "right_position"])

for i in range(500):
    next_t += period

    # read sensors
    distance = distance_sensor.distance_centimeters_continuous
    left_color = left_color_sensor.reflected_light_intensity
    right_color = right_color_sensor.reflected_light_intensity
    left_position = left_motor.position
    right_position = right_motor.position

    logger.log(time.monotonic() - start_t, distance, left_color, right_color, left_position, right_position)

    lag = next_t - time.monotonic()
    if lag < 0:
        overruns += 1
        next_t = time.monotonic()
    else:
        time.sleep(lag)

logger.close()
left_motor.stop()

with open(prefix + ".log", "w") as f:
    f.write("overruns: " + str(overruns) + "\n")
