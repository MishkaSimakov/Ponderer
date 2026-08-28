"""The benchmark must measure the network the brick would actually run.

Synthetic weights are only worth timing if they carry the same keys, at the same
shapes, that rl/export.py writes for a real model. That is what is checked here,
against models built in memory the way tests/test_export.py builds them, so no
trained run and no brick are needed.
"""

import csv

import pytest
from stable_baselines3.common.vec_env import DummyVecEnv

from conftest import ROOT  # noqa: F401  (puts the project on sys.path)
from experiments import inference_timing as bench
from rl import export as exporter
from rl.builder import build
from shared import logs
from shared.action import VOLTS
from shared.policies.net import NetPolicy
from test_export import StubEnv

HIDDEN = 8


def real_keys(arch, tmp_path, monkeypatch):
    """The npz an actual export writes for the same architecture and width."""
    model = build(arch, DummyVecEnv([StubEnv]), seed=0, hidden=HIDDEN, n_steps=16,
                  batch_size=16)
    monkeypatch.setattr(exporter, "ROOT", str(tmp_path))
    from shared.policies.net import load

    return load(exporter.export(model, arch, "test"))


@pytest.mark.parametrize("arch", bench.ARCHS)
def test_synthetic_weights_carry_the_keys_export_writes(arch, tmp_path, monkeypatch):
    real = real_keys(arch, tmp_path, monkeypatch)
    synthetic = bench.params(arch, HIDDEN)

    assert set(synthetic) == set(real)


@pytest.mark.parametrize("arch", bench.ARCHS)
def test_synthetic_weights_have_the_shapes_export_writes(arch, tmp_path, monkeypatch):
    real = real_keys(arch, tmp_path, monkeypatch)
    synthetic = bench.params(arch, HIDDEN)

    assert {k: v.shape for k, v in synthetic.items()} == \
           {k: v.shape for k, v in real.items()}


@pytest.mark.parametrize("arch", bench.ARCHS)
@pytest.mark.parametrize("hidden", bench.HIDDEN)
def test_synthetic_weights_run_and_stay_inside_the_voltage(arch, hidden):
    """A diverging lstm would make every timing after it meaningless."""
    policy = NetPolicy(bench.params(arch, hidden))

    for x in bench.inputs():
        action = policy.act_features(x)
        assert max(abs(a) for a in action) <= VOLTS


def test_macs_grow_with_width():
    """size() has to count the matrices, or the report cannot say what it measured."""
    small = bench.size(bench.params("lstm", 8))[1]
    large = bench.size(bench.params("lstm", 64))[1]

    assert large > 10 * small


def sweep(tmp_path, monkeypatch, archs=("mlp",), hidden=(8,), steps=3, warmup=1):
    monkeypatch.setattr(logs, "ROOT", str(tmp_path))
    monkeypatch.setattr(bench, "ARCHS", list(archs))
    monkeypatch.setattr(bench, "HIDDEN", list(hidden))
    monkeypatch.setattr(bench, "STEPS", steps)
    monkeypatch.setattr(bench, "WARMUP", warmup)
    monkeypatch.setattr(bench, "CLOCK_CALLS", 10)

    with open(bench.main()) as f:
        return list(csv.DictReader(f))


def test_the_log_has_one_plain_row_per_step_of_every_configuration(tmp_path, monkeypatch):
    rows = sweep(tmp_path, monkeypatch, archs=("mlp", "lstm"), hidden=(8, 16), steps=3)

    plain = [r for r in rows if r["pass"] == "plain"]
    assert len(plain) == 2 * 2 * 3
    assert all(r["phase"] == "total" for r in plain)
    assert sorted(set(r["step"] for r in plain)) == ["0", "1", "2"]


def test_the_instrumented_phases_add_up_to_the_step_they_split(tmp_path, monkeypatch):
    """elementwise is defined as the remainder, so this is what makes it meaningful."""
    rows = sweep(tmp_path, monkeypatch, archs=("lstm",), steps=3)

    steps = {}
    for row in rows:
        if row["pass"] == "instrumented":
            steps.setdefault(row["step"], {})[row["phase"]] = float(row["seconds"])

    assert len(steps) == 3
    for phases in steps.values():
        total = phases.pop("total")
        assert sum(phases.values()) == pytest.approx(total, abs=1e-9)


def test_the_instrumented_pass_names_every_layer_the_network_runs(tmp_path, monkeypatch):
    rows = sweep(tmp_path, monkeypatch, archs=("lstm",), steps=3)

    named = set(r["phase"] for r in rows if r["pass"] == "instrumented")
    assert named == {"lstm", "pi.0", "action", "elementwise", "total"}


def test_the_wrappers_are_undone_afterwards(tmp_path, monkeypatch):
    """The instrumented pass patches shared state; the plain pass must be plain."""
    from shared.policies import net

    before = (net.Dense.forward, net.NetPolicy._step, net.linear)
    sweep(tmp_path, monkeypatch, steps=3)

    assert (net.Dense.forward, net.NetPolicy._step, net.linear) == before
