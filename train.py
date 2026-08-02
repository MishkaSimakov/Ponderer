#!/usr/bin/env python3
"""RL entrypoint. PPO from stable-baselines3, architecture picked by --arch.

The arena count comes from unity (--arenas on its command line), not from here.
Watch a run with: tensorboard --logdir logs/tb
"""

import argparse
import os
import time

from stable_baselines3.common.callbacks import CheckpointCallback

from rl.builder import ARCHS, build
from rl.callbacks import OutcomeCallback
from rl.export import export
from shared.logs import ROOT

TENSORBOARD = os.path.join(ROOT, "tb")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=ARCHS, default="mlp")
    parser.add_argument("--port", type=int, default=5005)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total-steps", type=int, default=1_000_000)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--window", type=int, default=8, help="transformer context, frames")
    parser.add_argument("--n-steps", type=int, default=256, help="rollout length per arena")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--no-randomize-scenario", action="store_true")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    name = args.run_name or args.arch + "-" + time.strftime("%Y%m%d-%H%M%S", time.gmtime())

    # SB3 appends _1 to a fresh name and _2 to a taken one, which would split events
    # and checkpoints across two directories. Refuse the second run instead.
    run = os.path.join(TENSORBOARD, name + "_1")
    if os.path.exists(run):
        raise SystemExit("run %s already exists" % run)

    model = build(
        args.arch,
        port=args.port,
        seed=args.seed,
        hidden=args.hidden,
        window=args.window,
        randomize_scenario=not args.no_randomize_scenario,
        tensorboard_log=TENSORBOARD,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        ent_coef=args.ent_coef,
    )

    # A run is normally stopped by hand, so ctrl-c still has to save and export.
    try:
        model.learn(
            total_timesteps=args.total_steps,
            tb_log_name=name,
            callback=[OutcomeCallback(),
                      CheckpointCallback(save_freq=args.n_steps * 20, save_path=run,
                                         name_prefix="model")],
        )
    except KeyboardInterrupt:
        print("interrupted")

    model.save(os.path.join(run, "model"))
    print("exported %s" % export(model, args.arch, name))
    model.env.close()


if __name__ == "__main__":
    main()
