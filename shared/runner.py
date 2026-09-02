"""One run of one policy: the same code in simulation and on the brick.

Order is read, decide, write: the action lands as soon as it exists and the observation
that follows is read right after it. The action is logged in the same row as the
observation it was computed from; misaligning this reads as actuation latency that is
not there. The row's time is the one inside the observation, not the host clock: only
the first is comparable between brick and simulation.

The loop holds no clock of its own. How long a step takes is whatever the robot and the
network cost, and the length of a run is counted in steps or in seconds off the
observation clock, never in a rate.

The entrypoints are arena.py and brick/run.py. They differ in the robot they build
and in nothing else.
"""

from shared.csv_logger import CsvLogger
from shared.logs import run_prefix
from shared.observation import COLUMNS

LOG_COLUMNS = ["volts_left", "volts_right"] + COLUMNS
T = COLUMNS.index("t")


class EpisodeEnded(Exception):
    """The world ended the episode, so the run cannot go on in the same one."""


def run(robot, policy, source, name, steps=None):
    """One run, bounded by steps, or by the policy's own schedule, or by neither.

    A step count given here wins. Without one, a policy that declares a duration runs
    until the observation clock reaches it; a policy that declares none runs until the
    episode ends or ctrl-c.
    """
    log = run_prefix(source, name) + ".csv"
    logger = CsvLogger(log, LOG_COLUMNS)
    seconds = policy.duration if steps is None else None

    if steps is not None:
        bound = "%d steps" % steps
    elif seconds is not None:
        bound = "%.1f s" % seconds
    else:
        bound = "unbounded"
    print("%s: %s, %s" % (source, name, bound))

    obs = robot.reset()
    tick = 0

    try:
        while (steps is None or tick < steps) and (seconds is None or obs[T] < seconds):
            action = policy.act(obs)
            logger.log(action[0], action[1], *obs)
            obs = robot.step(action)
            tick += 1
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
