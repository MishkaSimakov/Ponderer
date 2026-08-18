#!/usr/bin/env python3
"""Compare the trained network with the one the brick will run.

Both sides are driven by the same feature sequence, so what is compared is the network
and the export, not shared/features.py, which is literally the same code on both.

    python check_export.py mlp-20260802-120000
"""

import argparse
import os

import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO

from rl.export import ROOT as POLICY_ROOT
from shared.features import DIM
from shared.logs import ROOT as LOGS_ROOT
from shared.policies.net import DUTY, NetPolicy, load

ALGOS = {"mlp": PPO, "lstm": RecurrentPPO, "transformer": PPO}


def trained_actions(model, arch, features, window):
    state = None
    starts = np.ones(1, bool)
    stacked = np.zeros(window * DIM, np.float32)
    actions = []

    for x in features:
        if arch == "transformer":
            stacked[:-DIM] = stacked[DIM:]
            stacked[-DIM:] = x
            obs = stacked
        else:
            obs = x

        action, state = model.predict(obs[None], state=state, episode_start=starts,
                                      deterministic=True)
        starts = np.zeros(1, bool)
        actions.append(action[0] * DUTY)

    return np.array(actions)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("name", help="run name, as passed to train.py --run-name")
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--tolerance", type=float, default=1e-3, help="duty percent")
    args = parser.parse_args()

    params = load(os.path.join(POLICY_ROOT, args.name + ".npz"))
    arch = params["arch"].item()
    model = ALGOS[arch].load(os.path.join(LOGS_ROOT, "tb", args.name + "_1", "model"))

    rng = np.random.default_rng(0)
    features = np.concatenate([rng.random((args.steps, 2)),
                               rng.uniform(-1.0, 1.0, (args.steps, DIM - 2))],
                              axis=1).astype(np.float32)

    window = int(params["window"]) if arch == "transformer" else 1
    reference = trained_actions(model, arch, features, window)

    policy = NetPolicy(params)
    ported = np.array([policy.act_features(x) for x in features])

    error = np.abs(reference - ported).max()
    print("%s: %d steps, max duty error %.3e" % (arch, args.steps, error))
    if error > args.tolerance:
        raise SystemExit("export does not match the trained policy")


if __name__ == "__main__":
    main()
