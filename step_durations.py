#!/usr/bin/env python3
"""Turn brick run logs into the step durations the simulator replays.

The t column of a run log is the robot's own clock, so the gap between two rows is how
long that step took, measured by the loop that actually deploys and including everything
it does. Each log stays its own run in the output: a stretch of consecutive steps must
never span two of them, because the join between two runs is a step no robot ever took.

    python step_durations.py logs/brick/net-20260802-010720.csv logs/brick/net-...csv
"""

import argparse
import csv
import os

import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bridge",
                   "step_durations.csv")
COLUMNS = ["run", "seconds"]


def durations(path, skip):
    """Step lengths in seconds, from the gaps between a run log's clock readings."""
    with open(path) as f:
        rows = list(csv.DictReader(f))

    t = np.array([float(row["t"]) for row in rows])
    if not np.all(np.diff(t) > 0):
        raise ValueError("%s: t is not strictly increasing" % path)

    # The first gap spans reset(), which zeroes both tachos and starts run-direct.
    gaps = np.diff(t)[skip:]
    if len(gaps) < 100:
        raise ValueError("%s: only %d steps, too few to resample" % (path, len(gaps)))
    return gaps


def summarize(name, gaps):
    ms = gaps * 1e3
    print("%s: n=%d, %.1f s" % (name, len(ms), gaps.sum()))
    print("  ms  mean %.1f  p50 %.1f  p95 %.1f  min %.1f  max %.1f  max/p50 %.1fx"
          % (ms.mean(), np.median(ms), np.percentile(ms, 95), ms.min(), ms.max(),
             ms.max() / np.median(ms)))
    # Consecutive steps are correlated on the brick, which is why the simulator
    # resamples stretches of them rather than drawing each one independently.
    print("  %.1f Hz mean, lag-1 autocorrelation %.2f"
          % (1.0 / gaps.mean(), np.corrcoef(ms[:-1], ms[1:])[0, 1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", help="logs/brick/<name>-<utc>.csv, one per run")
    parser.add_argument("--skip", type=int, default=1, help="leading gaps to drop per run")
    parser.add_argument("--out", default=OUT)
    args = parser.parse_args()

    runs = [durations(path, args.skip) for path in args.logs]
    for path, gaps in zip(args.logs, runs):
        summarize(os.path.basename(path), gaps)
    summarize("all runs", np.concatenate(runs))

    # Written here rather than through CsvLogger: this one is tracked, and csv's
    # default line terminator would put CRLF into the repository.
    with open(args.out, "w") as f:
        f.write(",".join(COLUMNS) + "\n")
        for run, gaps in enumerate(runs):
            for gap in gaps:
                f.write("%d,%.6f\n" % (run, gap))
    print("wrote %s" % args.out)


if __name__ == "__main__":
    main()
