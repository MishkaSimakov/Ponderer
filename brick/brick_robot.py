"""The hardware behind the interface shared.runner expects.

Observation order and units are RobotController.Observe's, so a log written here lines
up column for column with logs/sim/arena-*.csv.

The action is armature voltage, not duty: duty follows from the battery voltage
measured in the same step, so the policy's command means the same speed at any charge.
measured_volts already includes the sag under load, which is the voltage the motor
actually sees.

step waits before writing the action, not after. The brick spends most of a period
computing, so the action can only land at the end of the one it was computed in; sleeping
to that instant instead of writing as soon as the number exists makes it a fixed point of
the period rather than one that moves with however long inference took. Bridge.StepAll
applies the action at the same point of its step, and Bridge.controlPeriod is the same
constant as run.py's FREQUENCY. The pacing lives here, which is why runner.run is called
with period=None.
"""

import time

from ev3dev2.motor import Motor, OUTPUT_A, OUTPUT_B
from ev3dev2.power import PowerSupply
from ev3dev2.sensor import INPUT_2, INPUT_3
from ev3dev2.sensor.lego import ColorSensor


class BrickRobot:
    def __init__(self, period):
        self.period = period
        self.left_color = ColorSensor(INPUT_3)
        self.right_color = ColorSensor(INPUT_2)
        self.left_motor = Motor(OUTPUT_A)
        self.right_motor = Motor(OUTPUT_B)
        self.battery = PowerSupply()
        self.overruns = 0

    def reset(self):
        # run-direct takes duty straight, without the driver's speed PID.
        self.left_motor.run_direct(duty_cycle_sp=0)
        self.right_motor.run_direct(duty_cycle_sp=0)
        self.left_motor.position = 0
        self.right_motor.position = 0

        self.start = time.monotonic()
        self.deadline = self.start
        return self._observe()

    def step(self, action):
        # Before the sleep, so that only the two duty writes stand between the deadline
        # and the action. The sag it carries is the previous action's either way: that is
        # what the motors are still doing until the write below.
        volts = self.battery.measured_volts
        left = self._duty(action[0], volts)
        right = self._duty(action[1], volts)

        self.deadline += self.period
        lag = self.deadline - time.monotonic()
        if lag > 0:
            time.sleep(lag)
        else:
            self.overruns += 1
            self.deadline = time.monotonic()

        self.left_motor.duty_cycle_sp = left
        self.right_motor.duty_cycle_sp = right
        return self._observe()

    @staticmethod
    def _duty(command, battery):
        return int(round(max(-100.0, min(100.0, 100.0 * command / battery))))

    def stop(self):
        self.left_motor.stop(stop_action="coast")
        self.right_motor.stop(stop_action="coast")

    def _observe(self):
        return [time.monotonic() - self.start,
                self.left_color.reflected_light_intensity,
                self.right_color.reflected_light_intensity,
                self.left_motor.position,
                self.right_motor.position]
