#!/usr/bin/env python3
"""RL entrypoint. PPO from stable-baselines3, architecture picked by --arch.

With --env, unity players are launched headless, one per --num-envs, on
consecutive ports from --base-port. Without it, that many players must already be
listening there and the arena count comes from unity's own --arenas.

Watch a run with: tensorboard --logdir logs/tb
"""

import argparse
import os
import signal
import sys
import time

from stable_baselines3.common.callbacks import CheckpointCallback

from rl.builder import ARCHS, build
from rl.callbacks import Stats
from rl.export import export
from rl.vec_sim import make
from shared.logs import ROOT

TENSORBOARD = os.path.join(ROOT, "tb")
UNITY_LOGS = os.path.join(ROOT, "unity")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=ARCHS, default="mlp")
    parser.add_argument("--env", default=None, help="unity player to launch; default: attach to a running one")
    parser.add_argument("--num-envs", type=int, default=1, help="unity instances in parallel")
    parser.add_argument("--arenas", type=int, default=None, help="arenas per instance, only with --env")
    parser.add_argument("--base-port", type=int, default=5005, help="first port, one per instance")
    parser.add_argument("--graphics", action="store_true", help="show the players instead of running headless")
    parser.add_argument("--timeout-wait", type=float, default=60.0, help="seconds to wait for a player")
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
    parser.add_argument("--env-args", nargs=argparse.REMAINDER, default=[],
                        help="everything after it goes to the player's command line")
    args = parser.parse_args()

    if args.env is None and (args.arenas is not None or args.env_args or args.graphics):
        raise SystemExit("--arenas, --graphics and --env-args only apply when --env launches unity")

    name = args.run_name or args.arch + "-" + time.strftime("%Y%m%d-%H%M%S", time.gmtime())

    # SB3 appends _1 to a fresh name and _2 to a taken one, which would split events
    # and checkpoints across two directories. Refuse the second run instead.
    run = os.path.join(TENSORBOARD, name + "_1")
    if os.path.exists(run):
        raise SystemExit("run %s already exists" % run)

    # kill sends SIGTERM, and only an exception unwinds to the shutdown below.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit("terminated"))

    env = make(
        env=args.env,
        num_envs=args.num_envs,
        arenas=args.arenas if args.arenas is not None else 1,
        base_port=args.base_port,
        seed=args.seed,
        graphics=args.graphics,
        timeout=args.timeout_wait,
        log_prefix=os.path.join(UNITY_LOGS, name),
        extra=args.env_args,
        randomize_scenario=not args.no_randomize_scenario,
    )

    # A run ends by hand or because unity stopped. Either way the weights are the point
    # of the run, so saving and exporting happen anyway.
    try:
        model = build(
            args.arch,
            env,
            seed=args.seed,
            hidden=args.hidden,
            window=args.window,
            tensorboard_log=TENSORBOARD,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            ent_coef=args.ent_coef,
        )

        try:
            model.learn(
                total_timesteps=args.total_steps,
                tb_log_name=name,
                callback=[Stats(),
                          CheckpointCallback(save_freq=args.n_steps * 20, save_path=run,
                                             name_prefix="model")],
            )
        except KeyboardInterrupt:
            print("interrupted")
        except OSError as error:
            print("simulation stopped: %s" % error)

        model.save(os.path.join(run, "model"))
        print("exported %s" % export(model, args.arch, name))
    finally:
        env.close()


if __name__ == "__main__":
    main()
