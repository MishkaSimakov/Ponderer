import time

from shared.observation import COLUMNS

LOG_COLUMNS = ["volts_left", "volts_right"] + COLUMNS


def run(robot, policy, logger, steps, period=None):
    """Canonical loop, identical in simulation and on the brick.

    Order is read, decide, write, wait. The action is logged in the same row as
    the observation it was computed from; misaligning this reads as actuation
    latency that is not there. The row's time is the one inside the observation,
    not the host clock: only the first is comparable between brick and simulation.

    period=None runs unpaced, which is what lock-step training wants.
    """
    obs = robot.reset()
    start = time.monotonic()
    deadline = start
    overruns = 0

    for _ in range(steps):
        action = policy.act(obs)
        logger.log(action[0], action[1], *obs)
        obs = robot.step(action)

        if period is None:
            continue

        deadline += period
        lag = deadline - time.monotonic()
        if lag > 0:
            time.sleep(lag)
        else:
            overruns += 1
            deadline = time.monotonic()

    return overruns
