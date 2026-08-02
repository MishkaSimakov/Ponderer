"""Run an exported policy on plain numpy. This is what the brick executes.

No torch and no pickle: one npz of weights written by rl/export.py, read here. The
action is the clipped mean, because a deployed policy does not sample.

Every matrix multiply goes through linear(). Quantization replaces that one function
with int8 weights and int32 accumulation; nothing else in this file changes.
"""

import os

import numpy as np

from shared.features import Features
from shared.policy import Policy

ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "policy")

DUTY = 100.0
EPS = 1e-5


def linear(params, name, x):
    return x @ params[name + ".w"].T + params[name + ".b"]


def layer_norm(params, name, x):
    centered = x - x.mean(-1, keepdims=True)
    normed = centered / np.sqrt((centered * centered).mean(-1, keepdims=True) + EPS)
    return normed * params[name + ".w"] + params[name + ".b"]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


class NetPolicy(Policy):
    def __init__(self, params):
        self.p = params
        self.arch = params["arch"].item()
        self.features = Features()
        self.started = False

        if self.arch == "lstm":
            hidden = int(params["hidden"])
            self.h = np.zeros(hidden, np.float32)
            self.c = np.zeros(hidden, np.float32)
        elif self.arch == "transformer":
            window = int(params["window"])
            # Zero padded, like VecFrameStack, which clears the stack on every reset.
            self.window = np.zeros((window, int(params["frame"])), np.float32)
            self.mask = np.triu(np.full((window, window), -np.inf, np.float32), 1)

    def act(self, obs):
        x = self.features.update(obs) if self.started else self.features.first(obs)
        self.started = True
        return self.act_features(np.asarray(x, np.float32))

    def act_features(self, x):
        """Separate from act so check_export.py can compare networks, not feature code."""
        if self.arch == "mlp":
            h = x
        elif self.arch == "lstm":
            h = self._lstm(x)
        else:
            h = self._transformer(x)

        for i in range(int(self.p["pi_layers"])):
            h = np.tanh(linear(self.p, "pi.%d" % i, h))

        action = np.clip(linear(self.p, "action", h), -1.0, 1.0) * DUTY
        return float(action[0]), float(action[1])

    def _lstm(self, x):
        gates = linear(self.p, "lstm.ih", x) + linear(self.p, "lstm.hh", self.h)
        i, f, g, o = np.split(gates, 4)

        self.c = sigmoid(f) * self.c + sigmoid(i) * np.tanh(g)
        self.h = sigmoid(o) * np.tanh(self.c)
        return self.h

    def _transformer(self, x):
        self.window[:-1] = self.window[1:]
        self.window[-1] = x

        z = linear(self.p, "enc.project", self.window) + self.p["enc.pos"]
        for i in range(int(self.p["layers"])):
            z = layer_norm(self.p, "enc.%d.norm1" % i, z + self._attention(i, z))
            hidden = np.maximum(linear(self.p, "enc.%d.ff1" % i, z), 0.0)
            z = layer_norm(self.p, "enc.%d.norm2" % i, z + linear(self.p, "enc.%d.ff2" % i, hidden))

        # The newest frame is last, so the last token is the one to act on.
        return z[-1]

    def _attention(self, i, z):
        window, dim = z.shape
        heads = int(self.p["heads"])
        size = dim // heads

        parts = np.split(linear(self.p, "enc.%d.qkv" % i, z), 3, axis=1)
        q, k, v = [t.reshape(window, heads, size).transpose(1, 0, 2) for t in parts]

        weights = softmax(q @ k.transpose(0, 2, 1) / np.sqrt(size) + self.mask)
        merged = (weights @ v).transpose(1, 0, 2).reshape(window, dim)
        return linear(self.p, "enc.%d.proj" % i, merged)


def load(path):
    data = np.load(path)
    return {name: data[name] for name in data.files}


def newest():
    """Most recently exported policy. rsync -t keeps mtimes, so this holds on the brick."""
    files = [os.path.join(ROOT, f) for f in os.listdir(ROOT) if f.endswith(".npz")]
    if not files:
        raise IOError("no exports in %s" % ROOT)
    return max(files, key=os.path.getmtime)


def latest():
    """Factory: arena.py --policy shared.policies.net:latest, and the same on the brick."""
    return NetPolicy(load(newest()))
