#!/usr/bin/env python3
"""Run one policy in a single arena of a running unity, paced so unity shows it live.

    python arena.py --policy shared.policies.experiments.duty_steps:DutyStepsPolicy
    python arena.py --steps 400 --speed 0.5

The same run on the brick is brick/run.py.
"""

import argparse
import importlib

from bridge.sim_robot import Simulation, SimRobot
from shared.runner import run, steps_for


def resolve(spec):
    """module.path:attribute to the attribute itself."""
    module, _, attribute = spec.partition(":")
    return getattr(importlib.import_module(module), attribute)


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
    parser.add_argument("--speed", type=float, default=1.0,
                        help="wall clock pacing only, <1 slows the picture down")
    args = parser.parse_args()

    sim = Simulation(port=args.port, session_seed=args.seed)
    robot = SimRobot(sim, seed=args.seed, randomize_scenario=args.randomize_scenario)
    policy = resolve(args.policy)()
    name = args.name or args.policy.partition(":")[0].rpartition(".")[2]

    overruns = run(robot, policy, "sim", name, steps_for(policy, sim.dt, args.steps),
                   period=sim.dt / args.speed)
    print("%d overruns" % overruns)


if __name__ == "__main__":
    main()
