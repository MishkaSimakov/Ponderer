#!/usr/bin/env python3
"""Ultrasonic calibration: the screen shows the expected distance and the live
reading, the centre button writes the pair and advances the expected distance.

Step is 1 cm up to 15 cm, 5 cm after that. Rows hit the disk immediately, so the
run can be stopped at any point.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ev3dev2.button import Button
from ev3dev2.display import Display
from ev3dev2.fonts import load
from ev3dev2.sensor import INPUT_1
from ev3dev2.sensor.lego import UltrasonicSensor

from shared.csv_logger import CsvLogger
from shared.logs import run_prefix

PERIOD = 0.1
FONT = load("luBS24")

sensor = UltrasonicSensor(INPUT_1)
button = Button()
display = Display()

logger = CsvLogger(run_prefix("brick", "ultrasonic") + ".csv", ["expected", "measured"])

expected = 0
was_pressed = False

try:
    while True:
        measured = sensor.distance_centimeters_continuous

        display.text_pixels("set %5.1f" % expected, x=5, y=25, font=FONT)
        display.text_pixels("got %5.1f" % measured, x=5, y=70, font=FONT,
                            clear_screen=False)
        display.update()

        pressed = button.enter
        if pressed and not was_pressed:
            logger.log(expected, measured)
            logger.flush()
            print("%5.1f %5.1f" % (expected, measured))
            expected += 1 if expected < 15 else 5
        was_pressed = pressed

        time.sleep(PERIOD)
finally:
    logger.close()
