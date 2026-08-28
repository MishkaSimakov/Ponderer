#!/usr/bin/env python3
"""Reflectance of every surface on the test pad: the screen shows an id, the
centre button measures and writes a row, backspace stops.

Both sensors are read together in each mode, so a pair shares the ambient
conditions of one instant. The sensors are apart, so note down what sits under
each of them for every id, not just which colour was aimed at.
"""

import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ev3dev2.button import Button
from ev3dev2.display import Display
from ev3dev2.fonts import load
from ev3dev2.sensor import INPUT_2, INPUT_3
from ev3dev2.sensor.lego import ColorSensor

from shared.csv_logger import CsvLogger
from shared.logs import run_prefix

SAMPLES = 50
# Sampling at the control rate, so the spread is the noise the policy will see.
PERIOD = 0.05
# The mode switch physically changes the LED colour.
SETTLE = 0.5
FONT = load("luBS24")

SENSOR_COLUMNS = ["ref", "ref_sd", "ref0", "ref1",
                  "red", "green", "blue", "reflect", "reflect_sd"]
COLUMNS = (["id"]
           + ["left_" + c for c in SENSOR_COLUMNS]
           + ["right_" + c for c in SENSOR_COLUMNS])

left_color_sensor = ColorSensor(INPUT_3)
right_color_sensor = ColorSensor(INPUT_2)
button = Button()
display = Display()


def show(text):
    display.text_pixels(text, x=70, y=45, font=FONT)
    display.update()


def wait_for_press():
    """Name of the pressed button, once it is released again."""
    while not button.buttons_pressed:
        time.sleep(0.01)
    pressed = button.buttons_pressed[0]
    while button.buttons_pressed:
        time.sleep(0.01)
    return pressed


def sample(mode, values):
    """Both sensors read together, at the control rate, for one mode."""
    left_color_sensor.mode = mode
    right_color_sensor.mode = mode
    time.sleep(SETTLE)

    left_rows = []
    right_rows = []
    for _ in range(SAMPLES):
        left_rows.append([left_color_sensor.value(i) for i in range(values)])
        right_rows.append([right_color_sensor.value(i) for i in range(values)])
        time.sleep(PERIOD)

    return left_rows, right_rows


def medians(rows):
    return [statistics.median(channel) for channel in zip(*rows)]


def columns(raw, rgb, reflect):
    # ref1 is read with the LED off, so the difference is what cancels the
    # backlight. Per sample, because that is what drops the drift common to both.
    signal = [off - on for on, off in raw]
    percent = [row[0] for row in reflect]

    return ([statistics.median(signal), statistics.pstdev(signal)]
            + medians(raw)
            + medians(rgb)
            + [statistics.median(percent), statistics.pstdev(percent)])


logger = CsvLogger(run_prefix("brick", "colors") + ".csv", COLUMNS)
color_id = 1

try:
    while True:
        show(str(color_id))
        if wait_for_press() == "backspace":
            break

        show("...")
        left_raw, right_raw = sample(ColorSensor.MODE_REF_RAW, 2)
        left_rgb, right_rgb = sample(ColorSensor.MODE_RGB_RAW, 3)
        left_reflect, right_reflect = sample(ColorSensor.MODE_COL_REFLECT, 1)

        row = ([color_id]
               + columns(left_raw, left_rgb, left_reflect)
               + columns(right_raw, right_rgb, right_reflect))
        logger.log(*row)
        # Every id survives an interrupted session.
        logger.flush()
        print(row)
        color_id += 1
finally:
    logger.close()
    display.clear()
    display.update()
