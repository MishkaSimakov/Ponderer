#!/usr/bin/env python3
"""What one sensor read and one motor write cost, ev3dev2 against raw descriptors.

ev3dev2 reaches an attribute through a property that seeks its cached handle to zero
and calls read() with no size: readall fstats the file and then reads it twice, once
for the value and once for the end. reflected_light_intensity reads a second attribute
on top of that, because _ensure_mode rereads mode and compares it to COL-REFLECT on
every call. The raw path opens the same file once and reads it with os.pread, which is
one syscall and no wrapper.

Nothing moves: the duty written is zero, so the whole table can be measured with the
robot standing on the desk.

Every number is the mean of REPEATS calls, so the two clock calls bracketing it are
divided by REPEATS too. A tight loop over one attribute is the best case for both
paths, and the per-attribute columns are only there to say where the difference comes
from. observe_property against observe_pread is the comparison that decides anything:
those two are the whole of BrickRobot._observe, before and after.

One row per round in logs/brick/<NAME>-<utc>.csv.

    ./experiments/io_timing.py [rounds] [repeats]
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ev3dev2.motor import Motor, OUTPUT_A, OUTPUT_B
from ev3dev2.power import PowerSupply
from ev3dev2.sensor import INPUT_2, INPUT_3
from ev3dev2.sensor.lego import ColorSensor

from brick.brick_robot import READ_SIZE, open_attribute
from shared.csv_logger import CsvLogger
from shared.logs import run_prefix

NAME = "io_timing"  # logs/brick/<NAME>-<utc>.csv

ROUNDS = 50
REPEATS = 20
CLOCK_CALLS = 10000

# Sweeping the averaging is the point of the script, so both come off the command line.
ROUNDS = int(sys.argv[1]) if len(sys.argv) > 1 else ROUNDS
REPEATS = int(sys.argv[2]) if len(sys.argv) > 2 else REPEATS

left_color = ColorSensor(INPUT_3)
right_color = ColorSensor(INPUT_2)
left_motor = Motor(OUTPUT_A)
right_motor = Motor(OUTPUT_B)
battery = PowerSupply()

# run-direct so that a duty write is the write the control loop makes, zero so that
# measuring it does not drive the robot off the desk.
left_motor.run_direct(duty_cycle_sp=0)
right_motor.run_direct(duty_cycle_sp=0)
left_color.mode = ColorSensor.MODE_COL_REFLECT
right_color.mode = ColorSensor.MODE_COL_REFLECT

left_light = open_attribute(left_color, "value0", os.O_RDONLY)
right_light = open_attribute(right_color, "value0", os.O_RDONLY)
left_tacho = open_attribute(left_motor, "position", os.O_RDONLY)
right_tacho = open_attribute(right_motor, "position", os.O_RDONLY)
left_duty = open_attribute(left_motor, "duty_cycle_sp", os.O_WRONLY)
voltage = open_attribute(battery, "voltage_now", os.O_RDONLY)

start = time.monotonic()


def color_property():
    return left_color.reflected_light_intensity


def color_value():
    """value(0) without the mode reread: the half of the property that is the sensor."""
    return left_color.value(0)


def color_pread():
    return int(os.pread(left_light, READ_SIZE, 0))


def position_property():
    return left_motor.position


def position_pread():
    return int(os.pread(left_tacho, READ_SIZE, 0))


def volts_property():
    return battery.measured_volts


def volts_pread():
    return int(os.pread(voltage, READ_SIZE, 0)) / 1e6


def duty_property():
    left_motor.duty_cycle_sp = 0


def duty_pwrite():
    os.pwrite(left_duty, b"0", 0)


def observe_property():
    """BrickRobot._observe as ev3dev2 spells it."""
    return [time.monotonic() - start,
            left_color.reflected_light_intensity,
            right_color.reflected_light_intensity,
            left_motor.position,
            right_motor.position]


def observe_pread():
    """BrickRobot._observe as it runs now."""
    return [time.monotonic() - start,
            int(os.pread(left_light, READ_SIZE, 0)),
            int(os.pread(right_light, READ_SIZE, 0)),
            int(os.pread(left_tacho, READ_SIZE, 0)),
            int(os.pread(right_tacho, READ_SIZE, 0))]


MEASUREMENTS = [color_property, color_value, color_pread,
                position_property, position_pread,
                volts_property, volts_pread,
                duty_property, duty_pwrite,
                observe_property, observe_pread]


def timed(measurement):
    """Seconds per call, over REPEATS of them."""
    begin = time.monotonic()
    for _ in range(REPEATS):
        measurement()
    return (time.monotonic() - begin) / REPEATS


clock_start = time.monotonic()
for _ in range(CLOCK_CALLS):
    time.monotonic()
clock = (time.monotonic() - clock_start) / CLOCK_CALLS
print("clock %.1f us per call, %.1f us per number at %d repeats"
      % (1e6 * clock, 2e6 * clock / REPEATS, REPEATS))

for measurement in MEASUREMENTS:
    timed(measurement)  # the first call of each opens and caches ev3dev2's handle

rounds = []
try:
    for _ in range(ROUNDS):
        rounds.append([timed(measurement) for measurement in MEASUREMENTS])
except KeyboardInterrupt:
    print("interrupted")
finally:
    left_motor.stop(stop_action="coast")
    right_motor.stop(stop_action="coast")

names = [measurement.__name__ for measurement in MEASUREMENTS]
log = run_prefix("brick", NAME) + ".csv"
logger = CsvLogger(log, names)
for row in rounds:
    logger.log(*row)
logger.close()

print("%d rounds of %d calls" % (len(rounds), REPEATS))
print("%-18s %8s %8s %8s %8s" % ("us", "mean", "p50", "p95", "max"))
for i, name in enumerate(names):
    ordered = sorted(row[i] for row in rounds)
    print("%-18s %8.1f %8.1f %8.1f %8.1f" % (
        name,
        1e6 * sum(ordered) / len(ordered),
        1e6 * ordered[len(ordered) // 2],
        1e6 * ordered[int(0.95 * (len(ordered) - 1))],
        1e6 * ordered[-1]))
print("log %s" % log)
