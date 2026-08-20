"""One run of one policy: the same code in simulation and on the brick.

Order is read, decide, write, wait. The action is logged in the same row as the
observation it was computed from; misaligning this reads as actuation latency that
is not there. The row's time is the one inside the observation, not the host clock:
only the first is comparable between brick and simulation.

The entrypoints are arena.py and brick/run.py. They differ in the robot they build
and in nothing else.
"""

import time

from shared.csv_logger import CsvLogger
from shared.logs import run_prefix
from shared.observation import COLUMNS

LOG_COLUMNS = ["volts_left", "volts_right"] + COLUMNS


class EpisodeEnded(Exception):
    """The world ended the episode, so the run cannot go on in the same one."""


def steps_for(policy, period, steps=None):
    """steps, or as many as the policy's own schedule needs, or None for unbounded."""
    if steps is not None:
        return steps
    return None if policy.duration is None else int(round(policy.duration / period))


def run(robot, policy, source, name, steps, period=None):
    """period=None leaves the pacing to the robot, which is what the brick wants."""
    log = run_prefix(source, name) + ".csv"
    logger = CsvLogger(log, LOG_COLUMNS)
    print("%s: %s, %s steps" % (source, name, "unbounded" if steps is None else steps))

    obs = robot.reset()
    deadline = time.monotonic()
    overruns = 0
    tick = 0

    try:
        while steps is None or tick < steps:
            action = policy.act(obs)
            logger.log(action[0], action[1], *obs)
            obs = robot.step(action)
            tick += 1

            if period is None:
                continue

            deadline += period
            lag = deadline - time.monotonic()
            if lag > 0:
                time.sleep(lag)
            else:
                overruns += 1
                deadline = time.monotonic()
    except EpisodeEnded as error:
        # Only a run of no fixed length may end this way; a fixed one was cut short.
        if steps is not None:
            raise
        print(error)
    except KeyboardInterrupt:
        print("interrupted")
    except OSError as error:
        # The other side stopped. The log written so far is still worth keeping.
        print("robot stopped: %s" % error)
    finally:
        robot.stop()
        logger.close()

    print("ran %d steps, log %s" % (tick, log))
    return overruns
