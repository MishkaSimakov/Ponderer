"""The trained network and the one the brick runs must produce the same volts.

Models are built here rather than loaded from a run, so this covers every
architecture on every checkout instead of whichever one was trained last. Both
sides are driven by the same feature sequence: what is compared is the network and
the export, not shared/features.py, which is literally the same code on both.
"""

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from torch import nn

from rl import export as exporter
from rl.builder import ARCHS, build
from shared.action import VOLTS
from shared.features import DIM
from shared.policies.net import NetPolicy, load

WINDOW = 4
HIDDEN = 8
STEPS = 64
# Volts. The host trains in float64 torch, the brick infers in float32 numpy.
TOLERANCE = 1e-3


class StubEnv(gym.Env):
    """Never stepped by a trainer: the models here are built, not learned."""

    observation_space = spaces.Box(-np.inf, np.inf, (DIM,), np.float32)
    action_space = spaces.Box(-1.0, 1.0, (2,), np.float32)

    def __init__(self):
        self.tick = 0

    def reset(self, seed=None, options=None):
        self.tick = 0
        return np.zeros(DIM, np.float32), {}

    def step(self, action):
        self.tick += 1
        return np.full(DIM, self.tick, np.float32), 0.0, False, False, {}


def model_for(arch):
    return build(arch, DummyVecEnv([StubEnv]), seed=0, hidden=HIDDEN, window=WINDOW,
                 n_steps=16, batch_size=16)


def exported(model, arch, tmp_path, monkeypatch):
    monkeypatch.setattr(exporter, "ROOT", str(tmp_path))
    return load(exporter.export(model, arch, "test"))


def features(steps=STEPS):
    """Colors are 0..1, speeds are signed, as shared/features.py produces them."""
    rng = np.random.default_rng(0)
    return np.concatenate([rng.random((steps, 2)),
                           rng.uniform(-1.0, 1.0, (steps, DIM - 2))],
                          axis=1).astype(np.float32)


def trained_actions(model, arch, sequence, window):
    state = None
    starts = np.ones(1, bool)
    stacked = np.zeros(window * DIM, np.float32)
    actions = []

    for x in sequence:
        if arch == "transformer":
            stacked[:-DIM] = stacked[DIM:]
            stacked[-DIM:] = x
            obs = stacked
        else:
            obs = x

        action, state = model.predict(obs[None], state=state, episode_start=starts,
                                      deterministic=True)
        starts = np.zeros(1, bool)
        actions.append(action[0] * VOLTS)

    return np.array(actions)


@pytest.mark.parametrize("arch", ARCHS)
def test_export_matches_the_trained_policy(arch, tmp_path, monkeypatch):
    model = model_for(arch)
    params = exported(model, arch, tmp_path, monkeypatch)
    assert params["arch"].item() == arch

    sequence = features()
    window = int(params["window"]) if arch == "transformer" else 1
    reference = trained_actions(model, arch, sequence, window)

    policy = NetPolicy(params)
    ported = np.array([policy.act_features(x) for x in sequence])

    assert np.abs(reference - ported).max() < TOLERANCE


@pytest.mark.parametrize("arch", ARCHS)
def test_exported_actions_are_capped_at_the_voltage(arch, tmp_path, monkeypatch):
    policy = NetPolicy(exported(model_for(arch), arch, tmp_path, monkeypatch))
    ported = np.array([policy.act_features(x) for x in features()])

    assert np.abs(ported).max() <= VOLTS


def test_frame_stack_puts_the_newest_frame_last():
    """What NetPolicy._transformer's ring buffer and callbacks.py's slice assume."""
    env = VecFrameStack(DummyVecEnv([StubEnv]), WINDOW)
    env.reset()
    for tick in range(1, WINDOW + 2):
        obs = env.step(np.zeros((1, 2), np.float32))[0]
        assert obs[0, -DIM:].tolist() == [tick] * DIM


def test_export_rejects_a_head_activation_net_py_cannot_run():
    model = model_for("mlp")
    model.policy.mlp_extractor.policy_net[1] = nn.ReLU()

    with pytest.raises(ValueError, match="tanh"):
        exporter.export(model, "mlp", "test")


def test_export_rejects_a_multi_layer_lstm():
    model = RecurrentPPO("MlpLstmPolicy", DummyVecEnv([StubEnv]), seed=0, n_steps=16,
                         batch_size=16,
                         policy_kwargs=dict(lstm_hidden_size=HIDDEN, n_lstm_layers=2,
                                            net_arch=[HIDDEN]))

    with pytest.raises(ValueError, match="single lstm layer"):
        exporter.export(model, "lstm", "test")


def test_export_rejects_pre_norm_transformer_blocks():
    model = model_for("transformer")
    for layer in model.policy.features_extractor.encoder.layers:
        layer.norm_first = True

    with pytest.raises(ValueError, match="post norm"):
        exporter.export(model, "transformer", "test")
