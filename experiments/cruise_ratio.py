#!/usr/bin/env python3
"""Cruise speed per volt for both motors over all duty combinations.

Each combination: settle, then average the encoder speed over the window. The
run pauses for a button click every BATCH measurements so the robot can be put
back. Rows hit the disk immediately, so the run can be stopped at any point.
"""

import itertools
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ev3dev2.button import Button
from ev3dev2.display import Display
from ev3dev2.fonts import load
from ev3dev2.motor import Motor, OUTPUT_A, OUTPUT_B
from ev3dev2.power import PowerSupply

from shared.csv_logger import CsvLogger
from shared.logs import run_prefix

WHEEL_RADIUS = 0.0216  # m, same as LargeMotor.cs
STEP = 0.2
SETTLE = 0.25
WINDOW = 0.75
BATCH = 10

FONT = load("luBS14")

STEPS = int(round(1 / STEP))
DUTIES = [round(i * STEP, 6) for i in range(-STEPS, STEPS + 1)]

left_motor = Motor(OUTPUT_A)
right_motor = Motor(OUTPUT_B)
battery = PowerSupply()
button = Button()
display = Display()

logger = CsvLogger(run_prefix("brick", "cruise_ratio") + ".csv",
                   ["index", "duty_left", "duty_right", "volts",
                    "left_omega", "right_omega", "left_speed", "right_speed",
                    "left_ratio", "right_ratio"])


def stop():
    left_motor.stop(stop_action="coast")
    right_motor.stop(stop_action="coast")


def wait_click():
    while button.enter:
        time.sleep(0.05)
    while not button.enter:
        time.sleep(0.05)


def show(*lines):
    for i, line in enumerate(lines):
        display.text_pixels(line, x=5, y=10 + 25 * i, font=FONT, clear_screen=(i == 0))
    display.update()


combinations = list(itertools.product(DUTIES, DUTIES))

try:
    for index, (duty_left, duty_right) in enumerate(combinations):
        if index % BATCH == 0:
            stop()
            show("place the robot", "%d / %d" % (index, len(combinations)), "press enter")
            wait_click()

        left_motor.run_direct(duty_cycle_sp=int(round(duty_left * 100)))
        right_motor.run_direct(duty_cycle_sp=int(round(duty_right * 100)))
        time.sleep(SETTLE)

        start_volts = battery.measured_volts
        start_left = left_motor.position
        start_right = right_motor.position
        start_t = time.monotonic()

        time.sleep(WINDOW)

        end_t = time.monotonic()
        end_left = left_motor.position
        end_right = right_motor.position
        end_volts = battery.measured_volts

        dt = end_t - start_t
        volts = 0.5 * (start_volts + end_volts)
        left_omega = math.radians(end_left - start_left) / dt
        right_omega = math.radians(end_right - start_right) / dt
        left_speed = left_omega * WHEEL_RADIUS
        right_speed = right_omega * WHEEL_RADIUS

        left_ratio = left_speed / (volts * duty_left) if duty_left else ""
        right_ratio = right_speed / (volts * duty_right) if duty_right else ""

        logger.log(index, duty_left, duty_right, volts,
                   left_omega, right_omega, left_speed, right_speed,
                   left_ratio, right_ratio)
        logger.flush()

        show("%d / %d" % (index + 1, len(combinations)),
             "duty %5.1f %5.1f" % (duty_left, duty_right),
             "w    %5.1f %5.1f" % (left_omega, right_omega))
        print("%3d %5.1f %5.1f %8.3f %8.3f" % (index, duty_left, duty_right,
                                               left_omega, right_omega))
finally:
    stop()
    logger.close()
