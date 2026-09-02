#!/usr/bin/env python3
"""Run one policy in a single arena of a running unity, paced so unity shows it live.

    python arena.py --policy shared.policies.experiments.duty_steps:DutyStepsPolicy
    python arena.py --steps 400 --speed 0.5

The same run on the brick is brick/run.py.
"""

import argparse
import importlib
import time

from bridge.sim_robot import Simulation, SimRobot
from bridge.step_clock import StepClock, load
from shared.observation import COLUMNS
from shared.runner import run

T = COLUMNS.index("t")


def resolve(spec):
    """module.path:attribute to the attribute itself."""
    module, _, attribute = spec.partition(":")
    return getattr(importlib.import_module(module), attribute)


class PacedRobot:
    """Holds a run back to the speed a human watches it at, by the observation clock.

    The loop is unpaced on both sides, so a sim run would otherwise draw as fast as the
    socket allows. This is the only place in a run that sleeps, which is why it lives
    beside the entrypoint that wants it rather than in shared.runner.
    """

    def __init__(self, robot, speed):
        self.robot = robot
        self.speed = speed

    def reset(self):
        obs = self.robot.reset()
        self.t = obs[T]
        self.deadline = time.monotonic()
        return obs

    def step(self, action):
        obs = self.robot.step(action)
        self.deadline += (obs[T] - self.t) / self.speed
        self.t = obs[T]

        lag = self.deadline - time.monotonic()
        if lag > 0:
            time.sleep(lag)
        else:
            # Falling behind the picture is not an error, only a slower picture.
            self.deadline = time.monotonic()
        return obs

    def stop(self):
        self.robot.stop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="shared.policies.net:latest",
                        help="module:attribute, called with no arguments")
    parser.add_argument("--name", default=None, help="log name, default: the policy's module")
    parser.add_argument("--steps", type=int, default=None,
                        help="default: the policy's own schedule, or one episode if it has none")
    parser.add_argument("--port", type=int, default=5005)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--randomize-scenario", action="store_true")
    parser.add_argument("--randomize-physics", action="store_true")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="wall clock pacing only, <1 slows the picture down")
    args = parser.parse_args()

    sim = Simulation(StepClock(load(), args.seed), port=args.port,
                     session_seed=args.seed)
    robot = SimRobot(sim, seed=args.seed, randomize_scenario=args.randomize_scenario,
                     randomize_physics=args.randomize_physics)
    policy = resolve(args.policy)()
    name = args.name or args.policy.partition(":")[0].rpartition(".")[2]

    run(PacedRobot(robot, args.speed), policy, "sim", name, args.steps)


if __name__ == "__main__":
    main()
