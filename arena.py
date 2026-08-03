#!/usr/bin/env python3
"""Run one policy in a single arena, paced at the control period, so unity shows it live.

    from arena import run
    run(lambda: MyPolicy(0.4))

    python arena.py --policy mypolicies:Follower

make_policy is called again whenever an episode ends, because a policy is stateless
across episodes.
"""

import argparse
import importlib
import time

from bridge.sim_robot import Simulation
from shared.csv_logger import CsvLogger
from shared.logs import run_prefix
from shared.policies.constant import ConstantPolicy
from shared.runner import LOG_COLUMNS

from shared.policies.net import NetPolicy, load, newest


def run(make_policy, port=5005, seed=0, steps=None, randomize_scenario=False, log=None, speed=1.0):
    log = log or run_prefix("sim", "arena") + ".csv"
    print("arena: port %d, seed %d, steps %s, speed %g, log %s" % (port, seed, steps, speed, log))
    sim = Simulation(port=port, session_seed=seed)
    if sim.arenas != 1:
        raise ValueError("arena mode needs a single arena simulation, got %d" % sim.arenas)

    logger = CsvLogger(log, LOG_COLUMNS)
    policy = make_policy()
    obs = sim.reset(seeds=[seed], randomize_scenario=randomize_scenario).obs[0]
    print("running, ctrl-c to stop")

    start = time.monotonic()
    deadline = start
    overruns = 0
    episode = 0
    episode_steps = 0
    episode_return = 0.0
    tick = 0

    try:
        while steps is None or tick < steps:
            action = policy.act(obs)
            logger.log(action[0], action[1], *obs)
            state = sim.step([action])
            obs = state.obs[0]
            tick += 1
            episode_steps += 1
            episode_return += state.reward[0]

            if state.terminated[0] or state.truncated[0]:
                print("episode %d %s after %d steps, return %.3f" % (
                    episode,
                    "terminated" if state.terminated[0] else "truncated",
                    episode_steps, episode_return))
                policy = make_policy()
                episode += 1
                episode_steps = 0
                episode_return = 0.0

            deadline += sim.dt / speed
            lag = deadline - time.monotonic()
            if lag > 0:
                time.sleep(lag)
            else:
                overruns += 1
                deadline = time.monotonic()
    except KeyboardInterrupt:
        pass
    except OSError as error:
        # Unity stopped. The log written so far is still worth keeping.
        print("simulation stopped: %s" % error)

    print("%d steps, %d overruns" % (tick, overruns))
    logger.close()
    sim.close()


def resolve(spec):
    """module.path:attribute to the attribute itself."""
    module, _, attribute = spec.partition(":")
    return getattr(importlib.import_module(module), attribute)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True, help="module:attribute, called with no arguments")
    parser.add_argument("--port", type=int, default=5005)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--randomize-scenario", action="store_true")
    parser.add_argument("--speed", type=float, default=1.0, help="wall clock pacing only, <1 slows the picture down")
    parser.add_argument("--log", default=None, help="default: logs/sim/arena-<timestamp>.csv")
    args = parser.parse_args()

    run(lambda: NetPolicy(load(newest())), port=args.port, seed=args.seed, steps=args.steps,
        randomize_scenario=args.randomize_scenario, log=args.log, speed=args.speed)


if __name__ == "__main__":
    main()
