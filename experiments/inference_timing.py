#!/usr/bin/env python3
"""How long one forward pass takes on the brick, by architecture and size.

No hardware: numpy and shared/ only, so this runs over ssh with nothing plugged in,
and on the host as a sanity check. What it measures is NetPolicy.act_features, the
same object brick/run.py runs, fed synthetic weights under the key names rl/export.py
writes. There is no second implementation of the network here to drift from net.py.

Two passes per configuration. The plain pass is the honest number. The instrumented
pass wraps net.linear to say which layer the time went to; its total is larger by the
cost of the wrapper, so the two are reported apart.

A step this small is a few thousand multiply-accumulates. Against the numpy floor
printed at the top, the report says whether a forward pass costs arithmetic or costs
per-call dispatch.

Full log in logs/brick/inference_timing-<utc>.csv, one row per phase per step.

    ./experiments/inference_timing.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from shared.csv_logger import CsvLogger
from shared.features import DIM
from shared.logs import run_prefix
from shared.policies import net

NAME = "inference_timing"  # logs/brick/<NAME>-<utc>.csv

ARCHS = ["mlp", "lstm"]
HIDDEN = [8, 16, 32, 64]
STEPS = 200
WARMUP = 20
CLOCK_CALLS = 10000
SEED = 0

COLUMNS = ["arch", "hidden", "pass", "step", "phase", "seconds"]


def params(arch, hidden):
    """Synthetic weights under rl/export.py's key names, at rl/builder.py's shapes.

    mlp is net_arch=[hidden, hidden]; lstm is one lstm layer then net_arch=[hidden].
    """
    rng = np.random.RandomState(SEED)

    def rand(*shape):
        return (rng.randn(*shape) / np.sqrt(shape[-1])).astype(np.float32)

    out = {"arch": np.array(arch)}

    if arch == "lstm":
        out["lstm.ih.w"] = rand(4 * hidden, DIM)
        out["lstm.ih.b"] = rand(4 * hidden)
        out["lstm.hh.w"] = rand(4 * hidden, hidden)
        out["lstm.hh.b"] = rand(4 * hidden)
        out["hidden"] = np.array(hidden)
        head_in = hidden
        depth = 1
    else:
        head_in = DIM
        depth = 2

    for i in range(depth):
        out["pi.%d.w" % i] = rand(hidden, head_in)
        out["pi.%d.b" % i] = rand(hidden)
        head_in = hidden

    out["pi_layers"] = np.array(depth)
    out["action.w"] = rand(2, hidden)
    out["action.b"] = rand(2)
    return out


def size(p):
    """Weights, and the multiply-accumulates one step of them costs."""
    weights = sum(v.size for v in p.values() if v.dtype == np.float32)
    macs = sum(v.shape[0] * v.shape[1] for v in p.values()
               if v.dtype == np.float32 and v.ndim == 2)
    return weights, macs


def inputs():
    """One feature vector per step, generated up front so it is not timed."""
    rng = np.random.RandomState(SEED + 1)
    return rng.uniform(-1.0, 1.0, (WARMUP + STEPS, DIM)).astype(np.float32)


def plain(policy, sequence):
    """act_features with nothing wrapped around it."""
    for x in sequence[:WARMUP]:
        action = policy.act_features(x)

    times = []
    for x in sequence[WARMUP:]:
        t = time.monotonic()
        action = policy.act_features(x)
        times.append(time.monotonic() - t)

    if not np.isfinite(action).all():
        raise ValueError("state diverged")
    return times


def instrumented(policy, sequence):
    """Per linear call, plus what is left over: activations and the elementwise gates."""
    calls = []
    original = net.linear

    def timed(p, name, x):
        t = time.monotonic()
        y = original(p, name, x)
        calls.append((name, time.monotonic() - t))
        return y

    net.linear = timed
    try:
        for x in sequence[:WARMUP]:
            policy.act_features(x)

        steps = []
        for x in sequence[WARMUP:]:
            del calls[:]
            t = time.monotonic()
            policy.act_features(x)
            total = time.monotonic() - t
            steps.append((list(calls), total))
    finally:
        net.linear = original

    return steps


def floor():
    """One numpy call on the smallest possible operands: the per-call price."""
    a = np.zeros(1, np.float32)
    for _ in range(100):
        np.add(a, a)

    start = time.monotonic()
    for _ in range(CLOCK_CALLS // 10):
        np.add(a, a)
    return (time.monotonic() - start) / (CLOCK_CALLS // 10)


def stats(times):
    ordered = sorted(times)
    return (sum(ordered) / len(ordered),
            ordered[len(ordered) // 2],
            ordered[int(0.95 * (len(ordered) - 1))])


def main():
    """The sweep, its report, and the path of the log it wrote."""
    start = time.monotonic()
    for _ in range(CLOCK_CALLS):
        time.monotonic()
    clock = (time.monotonic() - start) / CLOCK_CALLS

    print("clock %.1f us per call, %.1f us per timed phase" % (1e6 * clock, 2e6 * clock))
    print("numpy %.1f us per call on scalar operands" % (1e6 * floor()))
    print("input dim %d, %d steps after %d warmup\n" % (DIM, STEPS, WARMUP))

    log = run_prefix("brick", NAME) + ".csv"
    logger = CsvLogger(log, COLUMNS)
    sequence = inputs()
    report = []
    breakdowns = []

    for arch in ARCHS:
        for hidden in HIDDEN:
            p = params(arch, hidden)
            weights, macs = size(p)

            times = plain(net.NetPolicy(p), sequence)
            for step, seconds in enumerate(times):
                logger.log(arch, hidden, "plain", step, "total", seconds)

            mean, p50, p95 = stats(times)
            report.append((arch, hidden, weights, macs, mean, p50, p95))

            phases = {}
            for step, (calls, total) in enumerate(instrumented(net.NetPolicy(p), sequence)):
                for name, seconds in calls:
                    logger.log(arch, hidden, "instrumented", step, name, seconds)
                    phases.setdefault(name, []).append(seconds)
                rest = total - sum(seconds for _, seconds in calls)
                logger.log(arch, hidden, "instrumented", step, "elementwise", rest)
                logger.log(arch, hidden, "instrumented", step, "total", total)
                phases.setdefault("elementwise", []).append(rest)
            breakdowns.append((arch, hidden, phases))

    logger.close()

    print("%-6s %6s %8s %8s %9s %9s %9s %8s"
          % ("arch", "hidden", "weights", "macs", "mean us", "p50 us", "p95 us", "Hz"))
    for arch, hidden, weights, macs, mean, p50, p95 in report:
        print("%-6s %6d %8d %8d %9.1f %9.1f %9.1f %8.0f"
              % (arch, hidden, weights, macs, 1e6 * mean, 1e6 * p50, 1e6 * p95, 1 / mean))

    print("\nwhere the time goes, mean us per step (instrumented, so totals run high)")
    for arch, hidden, phases in breakdowns:
        parts = sorted(phases.items(), key=lambda kv: -sum(kv[1]))
        print("%-6s %6d  %s" % (arch, hidden, "  ".join(
            "%s %.1f" % (name, 1e6 * sum(v) / len(v)) for name, v in parts)))

    print("\nlog %s" % log)
    return log


if __name__ == "__main__":
    main()
