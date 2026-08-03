"""The hardware behind the interface shared.runner expects.

Observation order and units are RobotController.Observe's, so a log written here lines
up column for column with logs/sim/arena-*.csv.

step waits between writing duty and reading sensors. In simulation unity advances its
own clock inside step; here the world advances in real time, so reading straight after
writing would put a whole control period of actuation latency into the hardware log and
nowhere else. That is why runner.run is called with period=None: the pacing lives here.
"""

import time

from ev3dev2.motor import Motor, OUTPUT_A, OUTPUT_B
from ev3dev2.sensor import INPUT_1, INPUT_2, INPUT_3
from ev3dev2.sensor.lego import ColorSensor, UltrasonicSensor


class BrickRobot:
    def __init__(self, period):
        self.period = period
        self.distance = UltrasonicSensor(INPUT_1)
        self.left_color = ColorSensor(INPUT_3)
        self.right_color = ColorSensor(INPUT_2)
        self.left_motor = Motor(OUTPUT_A)
        self.right_motor = Motor(OUTPUT_B)
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
        self.left_motor.duty_cycle_sp = int(round(action[0]))
        self.right_motor.duty_cycle_sp = int(round(action[1]))

        self.deadline += self.period
        lag = self.deadline - time.monotonic()
        if lag > 0:
            time.sleep(lag)
        else:
            self.overruns += 1
            self.deadline = time.monotonic()

        return self._observe()

    def stop(self):
        self.left_motor.stop(stop_action="coast")
        self.right_motor.stop(stop_action="coast")

    def _observe(self):
        return [time.monotonic() - self.start,
                self.distance.distance_centimeters_continuous,
                self.left_color.reflected_light_intensity,
                self.right_color.reflected_light_intensity,
                self.left_motor.position,
                self.right_motor.position]
