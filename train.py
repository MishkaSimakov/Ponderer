#!/usr/bin/env python3
"""RL entrypoint. Stub: drives every arena and assembles transitions, learns nothing.

The arena count comes from unity (--arenas on its command line), not from here.
"""

import argparse

from bridge.sim_robot import Simulation
from shared.policies.constant import ConstantPolicy


def make_policy():
    return ConstantPolicy((100.0, 100.0))


def learn(batch):
    """No algorithm yet: transitions are dropped."""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5005)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=1000)
    args = parser.parse_args()

    sim = Simulation(port=args.port, session_seed=args.seed)
    policies = [make_policy() for _ in range(sim.arenas)]
    state = sim.reset()
    episodes = 0

    for _ in range(args.steps):
        actions = [policies[i].act(state.obs[i]) for i in range(sim.arenas)]
        following = sim.step(actions)

        batch = []
        for i in range(sim.arenas):
            done = following.terminated[i] or following.truncated[i]
            # The successor of a finished episode is terminal_obs: obs already
            # belongs to the next episode, which unity started on its own.
            next_obs = following.terminal_obs[i] if done else following.obs[i]
            batch.append((state.obs[i], actions[i], following.reward[i], next_obs,
                          following.terminated[i], following.truncated[i]))

            if done:
                policies[i] = make_policy()
                episodes += 1

        learn(batch)
        state = following

    print("%d steps, %d episodes" % (args.steps, episodes))
    sim.close()


if __name__ == "__main__":
    main()
