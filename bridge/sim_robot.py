from collections import namedtuple

from bridge.connection import Connection, PROTOCOL_VERSION

# obs is the observation to act on next; terminal_obs is the last observation of a
# finished episode and is meaningful only where terminated or truncated is set.
Step = namedtuple("Step", "obs terminal_obs reward terminated truncated episode step")


class Simulation:
    """All arenas of one Unity process, one round trip per control step."""

    def __init__(self, port=5005, host="127.0.0.1", session_seed=0, timeout=30.0):
        self.conn = Connection(host, port, timeout=timeout)
        info = self.conn.request({
            "cmd": "handshake",
            "version": PROTOCOL_VERSION,
            "session_seed": int(session_seed),
        })

        if info["version"] != PROTOCOL_VERSION:
            raise RuntimeError("protocol version mismatch: unity %d" % info["version"])

        self.arenas = info["arenas"]
        self.dt = info["dt"]
        self.obs_dim = info["obs_dim"]
        self.action_dim = info["action_dim"]
        print("handshake: %d arenas, dt %.4f, obs_dim %d, action_dim %d, session_seed %d"
              % (self.arenas, self.dt, self.obs_dim, self.action_dim, session_seed))

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
        self.conn.send({"cmd": "step", "actions": flat})

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
        split = lambda key: [response[key][i * dim:(i + 1) * dim] for i in range(self.arenas)]
        return Step(
            obs=split("obs"),
            terminal_obs=split("terminal_obs"),
            reward=response["reward"],
            terminated=response["terminated"],
            truncated=response["truncated"],
            episode=response["episode"],
            step=response["step"],
        )


class SimRobot:
    """Single arena, matching the interface shared.runner expects."""

    def __init__(self, simulation, seed=None):
        if simulation.arenas != 1:
            raise ValueError("SimRobot needs a single arena simulation")
        self.sim = simulation
        self.seed = seed

    def reset(self):
        seeds = None if self.seed is None else [self.seed]
        return self.sim.reset(seeds=seeds, randomize_physics=False).obs[0]

    def step(self, action):
        return self.sim.step([action]).obs[0]
