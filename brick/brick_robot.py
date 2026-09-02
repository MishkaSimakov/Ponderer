"""The hardware behind the interface shared.runner expects.

Observation order and units are RobotController.Observe's, so a log written here lines
up column for column with logs/sim/arena-*.csv.

The action is armature voltage, not duty: duty follows from the battery voltage
measured in the same step, so the policy's command means the same speed at any charge.
voltage_now already includes the sag under load, which is the voltage the motor
actually sees.

The step reads and writes sysfs itself. ev3dev2 finds the devices, sets the sensor
mode and stops the motors, but none of its attribute properties are in the loop: each
seeks its handle to zero and calls read() with no size, which fstats the file and then
reads it twice, and reflected_light_intensity reads the mode file first to compare it
against COL-REFLECT. os.pread on a handle opened once is one syscall for the same
number. The mode is set in __init__ and nothing moves it afterwards, so nothing needs
to reread it.

The loop is unpaced. The action is written the instant it exists and the observation is
read right after it, so a step lasts exactly one pass of the loop and the t it returns
marks when the action landed. The distribution those steps make is what the simulator
draws its own step length from.
"""

import os
import time

from ev3dev2.motor import Motor, OUTPUT_A, OUTPUT_B
from ev3dev2.power import PowerSupply
from ev3dev2.sensor import INPUT_2, INPUT_3
from ev3dev2.sensor.lego import ColorSensor

# Wider than any attribute read here: three digits of reflectance, ten of encoder
# counts, eight of microvolts.
READ_SIZE = 32


def open_attribute(device, name, flags):
    """One sysfs attribute of a device ev3dev2 has already found.

    _path is the directory it matched to the port, and the only place the device
    number appears.
    """
    return os.open(device._path + "/" + name, flags)


class BrickRobot:
    def __init__(self):
        self.left_color = ColorSensor(INPUT_3)
        self.right_color = ColorSensor(INPUT_2)
        self.left_motor = Motor(OUTPUT_A)
        self.right_motor = Motor(OUTPUT_B)
        self.battery = PowerSupply()

        self.left_color.mode = ColorSensor.MODE_COL_REFLECT
        self.right_color.mode = ColorSensor.MODE_COL_REFLECT

        self.left_light = open_attribute(self.left_color, "value0", os.O_RDONLY)
        self.right_light = open_attribute(self.right_color, "value0", os.O_RDONLY)
        self.left_tacho = open_attribute(self.left_motor, "position", os.O_RDONLY)
        self.right_tacho = open_attribute(self.right_motor, "position", os.O_RDONLY)
        self.left_duty = open_attribute(self.left_motor, "duty_cycle_sp", os.O_WRONLY)
        self.right_duty = open_attribute(self.right_motor, "duty_cycle_sp", os.O_WRONLY)
        self.voltage = open_attribute(self.battery, "voltage_now", os.O_RDONLY)

    def reset(self):
        # run-direct takes duty straight, without the driver's speed PID.
        self.left_motor.run_direct(duty_cycle_sp=0)
        self.right_motor.run_direct(duty_cycle_sp=0)
        self.left_motor.position = 0
        self.right_motor.position = 0

        self.start = time.monotonic()
        return self._observe()

    def step(self, action):
        # The sag this carries is the previous action's: that is what the motors are
        # still doing until the write below.
        volts = int(os.pread(self.voltage, READ_SIZE, 0)) / 1e6
        left = self._duty(action[0], volts)
        right = self._duty(action[1], volts)

        os.pwrite(self.left_duty, str(left).encode(), 0)
        os.pwrite(self.right_duty, str(right).encode(), 0)
        return self._observe()

    @staticmethod
    def _duty(command, battery):
        """The duty that puts command volts on the motor at this battery."""
        return int(round(max(-100.0, min(100.0, 100.0 * command / battery))))

    def stop(self):
        self.left_motor.stop(stop_action="coast")
        self.right_motor.stop(stop_action="coast")

    def _observe(self):
        return [time.monotonic() - self.start,
                int(os.pread(self.left_light, READ_SIZE, 0)),
                int(os.pread(self.right_light, READ_SIZE, 0)),
                int(os.pread(self.left_tacho, READ_SIZE, 0)),
                int(os.pread(self.right_tacho, READ_SIZE, 0))]
