#!/usr/bin/env python3
"""LSTM step timing. All four gates share one matrix, so a step is a single
matvec over [x; h] plus elementwise work.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

INPUTS = 4
OUTPUTS = 2
HIDDEN = [8, 16, 32, 64]
ITERATIONS = 200


def make_weights(hidden, dtype):
    scale = 1.0 / np.sqrt(INPUTS + hidden)

    def rand(*shape):
        return (np.random.randn(*shape) * scale).astype(dtype)

    return rand(4 * hidden, INPUTS + hidden), rand(4 * hidden), rand(OUTPUTS, hidden), rand(OUTPUTS)


def step(weights, bias, out_weights, out_bias, x, h, c):
    gates = weights @ np.concatenate((x, h)) + bias
    i, f, g, o = np.split(gates, 4)

    c = 1 / (1 + np.exp(-f)) * c + 1 / (1 + np.exp(-i)) * np.tanh(g)
    h = 1 / (1 + np.exp(-o)) * np.tanh(c)

    return out_weights @ h + out_bias, h, c


def measure(hidden, dtype):
    weights, bias, out_weights, out_bias = make_weights(hidden, dtype)
    x = np.random.randn(INPUTS).astype(dtype)
    h = np.zeros(hidden, dtype)
    c = np.zeros(hidden, dtype)

    y, h, c = step(weights, bias, out_weights, out_bias, x, h, c)

    start = time.monotonic()
    for _ in range(ITERATIONS):
        y, h, c = step(weights, bias, out_weights, out_bias, x, h, c)
    end = time.monotonic()

    if not np.isfinite(y).all():
        raise ValueError("state diverged at hidden=%d" % hidden)

    period = (end - start) / ITERATIONS
    params = 4 * hidden * (INPUTS + hidden + 1) + OUTPUTS * (hidden + 1)
    macs = 4 * hidden * (INPUTS + hidden) + OUTPUTS * hidden

    print("%-8s hidden %3d  params %5d  macs %6d  %8.1f us  %6.0f Hz"
          % (np.dtype(dtype).name, hidden, params, macs, period * 1e6, 1 / period))


for dtype in [np.float16, np.float32, np.float64]:
    for hidden in HIDDEN:
        measure(hidden, dtype)
