from collections import namedtuple

from bridge.connection import Connection, PROTOCOL_VERSION
from shared.runner import EpisodeEnded

# obs is the observation to act on next; terminal_obs is the last observation of a
# finished episode and is meaningful only where terminated or truncated is set.
# terms breaks reward down by reward component, in the order of Simulation.reward_terms.
Step = namedtuple("Step", "obs terminal_obs reward terms terminated truncated episode step")


class Simulation:
    """All arenas of one Unity process, one round trip per control step.

    The clock is this process's alone. Physics.Simulate covers the whole unity scene, so
    every arena inside one process advances by the same step, and separate processes draw
    separate sequences.
    """

    def __init__(self, clock, port=5005, host="127.0.0.1", session_seed=0, timeout=30.0):
        self.conn = Connection(host, port, timeout=timeout)
        info = self.conn.request({
            "cmd": "handshake",
            "version": PROTOCOL_VERSION,
            "session_seed": int(session_seed),
        })

        if info["version"] != PROTOCOL_VERSION:
            raise RuntimeError("protocol version mismatch: unity %d" % info["version"])

        self.clock = clock
        self.arenas = info["arenas"]
        self.obs_dim = info["obs_dim"]
        self.action_dim = info["action_dim"]
        self.reward_terms = info["reward_terms"]
        print("handshake: %d arenas, obs_dim %d, action_dim %d, session_seed %d"
              % (self.arenas, self.obs_dim, self.action_dim, session_seed))
        print("reward terms: %s" % ", ".join(self.reward_terms))

    def reset(self, seeds=None, randomize_scenario=True, randomize_physics=False):
        message = {
            "cmd": "reset",
            "randomize_scenario": randomize_scenario,
            "randomize_physics": randomize_physics,
        }
        if seeds is not None:
            if len(seeds) != self.arenas:
                raise ValueError("expected %d seeds" % self.arenas)
            message["seeds"] = [int(s) for s in seeds]
        print("reset: seeds %s, scenario %s, physics %s"
              % (seeds, randomize_scenario, randomize_physics))
        return self._unpack(self.conn.request(message))

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def step_async(self, actions):
        """Sending without reading lets several unity processes compute at once."""
        if len(actions) != self.arenas:
            raise ValueError("expected %d actions" % self.arenas)
        flat = [float(v) for action in actions for v in action]
        self.conn.send({"cmd": "step", "actions": flat, "dt": self.clock.sample()})

    def step_wait(self):
        return self._unpack(self.conn.recv())

    def close(self):
        print("quit")
        # Unity may have stopped on its own, and then there is nobody left to tell.
        # Closing is the one place where a dead socket is not an error.
        try:
            self.conn.send({"cmd": "quit"})
        except OSError as error:
            print("could not send quit: %s" % error)
        self.conn.close()

    def _unpack(self, response):
        dim = self.obs_dim
        terms = len(self.reward_terms)
        split = lambda key, n: [response[key][i * n:(i + 1) * n] for i in range(self.arenas)]
        return Step(
            obs=split("obs", dim),
            terminal_obs=split("terminal_obs", dim),
            reward=response["reward"],
            terms=split("terms", terms),
            terminated=response["terminated"],
            truncated=response["truncated"],
            episode=response["episode"],
            step=response["step"],
        )


class SimRobot:
    """Single arena, matching the interface shared.runner expects.

    One episode, never two: unity auto resets, and a run that walked into the next
    episode would be a log of two runs glued together with the clock rewound. The
    end of the episode is where a run of no fixed length stops.
    """

    def __init__(self, simulation, seed=None, randomize_scenario=False,
                 randomize_physics=False):
        if simulation.arenas != 1:
            raise ValueError("SimRobot needs a single arena simulation")
        self.sim = simulation
        self.seed = seed
        self.randomize_scenario = randomize_scenario
        self.randomize_physics = randomize_physics

    def reset(self):
        seeds = None if self.seed is None else [self.seed]
        return self.sim.reset(seeds=seeds, randomize_scenario=self.randomize_scenario,
                              randomize_physics=self.randomize_physics).obs[0]

    def step(self, action):
        state = self.sim.step([action])
        if state.terminated[0] or state.truncated[0]:
            raise EpisodeEnded("episode %s at step %d: the arena has already reset"
                               % ("terminated" if state.terminated[0] else "truncated",
                                  state.step[0]))
        return state.obs[0]

    def stop(self):
        """Same place in a run as BrickRobot.stop: let go of the hardware."""
        self.sim.close()
