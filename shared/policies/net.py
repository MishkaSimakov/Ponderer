"""Run an exported policy on plain numpy. This is what the brick executes.

No torch and no pickle: one npz of weights written by rl/export.py, read here. The
action is the clipped mean, because a deployed policy does not sample.

A step is a few thousand multiply-accumulates, and one numpy call costs about 240 us
on the brick, so the step is priced in calls and not in arithmetic. The mlp and lstm
paths are built to make as few as possible: every bias is folded into its matrix as a
last column so a layer is one matvec, every buffer is allocated once, and each layer
writes its activation straight into the next one's input. The transformer path still
goes through linear(), which is the readable form; it is not deployed.
"""

import os

import numpy as np

from shared.action import VOLTS
from shared.features import DIM, Features, NAMES
from shared.policy import Policy

ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "policy")

EPS = 1e-5

# Bound once: on the brick a global-then-attribute lookup is not free at 20 Hz.
dot = np.dot
tanh = np.tanh
clip = np.clip
multiply = np.multiply


def linear(params, name, x):
    return x @ params[name + ".w"].T + params[name + ".b"]


def layer_norm(params, name, x):
    centered = x - x.mean(-1, keepdims=True)
    normed = centered / np.sqrt((centered * centered).mean(-1, keepdims=True) + EPS)
    return normed * params[name + ".w"] + params[name + ".b"]


def softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def folded(params, name, scale=1.0):
    """[W | b] as one matrix, so one matvec applies the bias too."""
    w = params[name + ".w"]
    b = params[name + ".b"]
    return (np.concatenate((w, b.reshape(-1, 1)), 1) * scale).astype(np.float32)


def check_width(width):
    """The export's input against the feature vector shared/features.py now builds.

    Without this a stale export survives construction and dies inside the first forward
    pass, as a broadcast error on a buffer, with nothing naming the feature vector.
    """
    if width != DIM:
        raise ValueError("export takes %d inputs, features give %d: %s"
                         % (width, DIM, ", ".join(NAMES)))


def inlet(size):
    """Input buffer for a folded layer: size values, then the 1 the bias multiplies."""
    return np.ones(size + 1, np.float32)


class Dense:
    """y = w @ x, the bias being w's last column against x's last entry, a fixed 1.

    x belongs to whatever runs before, which writes into it, so a layer boundary
    costs no copy.
    """

    def __init__(self, name, w, x):
        self.name = name
        self.w = w
        self.x = x
        self.y = np.zeros(w.shape[0], np.float32)
        self.into = None

    def forward(self):
        return dot(self.w, self.x, out=self.y)


class NetPolicy(Policy):
    def __init__(self, params):
        self.p = params
        self.arch = params["arch"].item()
        self.features = Features()
        self.started = False

        if self.arch == "lstm":
            head_in = self._build_lstm()
            self.core = self._recurrent
        elif self.arch == "mlp":
            width = params["pi.0.w"].shape[1]
            check_width(width)
            head_in = inlet(width)
            self.source = head_in[:-1]
            self.core = self._feed
        else:
            head_in = inlet(self._build_transformer())
            self.source = head_in[:-1]
            self.core = self._attend

        self.head = []
        x = head_in
        for i in range(int(params["pi_layers"])):
            layer = Dense("pi.%d" % i, folded(params, "pi.%d" % i), x)
            x = inlet(layer.y.size)
            layer.into = x[:-1]
            self.head.append(layer)

        # clip(W h + b, -1, 1) * VOLTS is clip(VOLTS W h + VOLTS b, -VOLTS, VOLTS).
        self.action = Dense("action", folded(params, "action", VOLTS), x)

    def _build_lstm(self):
        """Returns the head's input buffer, which is [h; 1] inside the gate input."""
        p = self.p
        hidden = int(p["hidden"])
        dim = p["lstm.ih.w"].shape[1]
        check_width(dim)

        # torch keeps the four gates in one matrix as i, f, g, o and adds both biases.
        self.gw = np.concatenate(
            (p["lstm.ih.w"], p["lstm.hh.w"],
             (p["lstm.ih.b"] + p["lstm.hh.b"]).reshape(-1, 1)), 1).astype(np.float32)

        # [x; h; 1]. h lives here, so the head reads it and the next step's matvec
        # takes it without either of them copying anything.
        self.gates_in = np.ones(dim + hidden + 1, np.float32)
        self.source = self.gates_in[:dim]
        self.h = self.gates_in[dim:dim + hidden]
        self.h[:] = 0.0
        self.c = np.zeros(hidden, np.float32)
        self.gates = np.zeros(4 * hidden, np.float32)
        self.tmp = np.zeros(hidden, np.float32)

        self.i = self.gates[:hidden]
        self.f = self.gates[hidden:2 * hidden]
        self.g = self.gates[2 * hidden:3 * hidden]
        self.o = self.gates[3 * hidden:]

        # sigmoid(z) = 0.5 tanh(0.5 z) + 0.5, so a single tanh covers all four gates:
        # i, f and o carry the halves, g carries one and zero and stays a plain tanh.
        # The inner half is folded into the matrix, so the step does not scale at all.
        self.post = np.full(4 * hidden, 0.5, np.float32)
        self.shift = np.full(4 * hidden, 0.5, np.float32)
        self.post[2 * hidden:3 * hidden] = 1.0
        self.shift[2 * hidden:3 * hidden] = 0.0
        self.gw *= self.post.reshape(-1, 1)

        return self.gates_in[dim:]

    def _build_transformer(self):
        """Returns the model dimension, which is what the head takes."""
        p = self.p
        size = int(p["window"])
        check_width(int(p["frame"]))
        # Zero padded, like VecFrameStack, which clears the stack on every reset.
        self.window = np.zeros((size, int(p["frame"])), np.float32)
        self.mask = np.triu(np.full((size, size), -np.inf, np.float32), 1)
        return p["enc.project.w"].shape[0]

    def act(self, obs):
        # The feature list goes straight into the input buffer: building an array of
        # it first would be one more numpy call for two numbers.
        values = self.features.update(obs) if self.started else self.features.first(obs)
        self.started = True
        self.core(values)
        return self._forward()

    def act_features(self, x):
        """Separate from act so tests/test_export.py compares networks, not features."""
        self.core(x)
        return self._forward()

    def _feed(self, x):
        self.source[:] = x

    def _recurrent(self, x):
        self.source[:] = x
        self._step()

    def _attend(self, x):
        self.source[:] = self._transformer(np.asarray(x, np.float32))

    def _forward(self):
        for layer in self.head:
            tanh(layer.forward(), out=layer.into)

        y = self.action.forward()
        clip(y, -VOLTS, VOLTS, out=y)
        # tolist is one call for both numbers; indexing them is two numpy scalars.
        return y.tolist()

    def _step(self):
        gates = self.gates
        dot(self.gw, self.gates_in, out=gates)
        tanh(gates, out=gates)
        gates *= self.post
        gates += self.shift

        self.c *= self.f
        multiply(self.i, self.g, out=self.tmp)
        self.c += self.tmp
        tanh(self.c, out=self.tmp)
        multiply(self.o, self.tmp, out=self.h)

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
