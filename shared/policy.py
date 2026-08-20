class Policy(object):
    """Stateless across episodes: a new episode means a new instance."""

    # Seconds the policy's own schedule takes. None: it has none, so a run of it
    # lasts until the episode ends or the number of steps asked for is reached.
    duration = None

    def act(self, obs):
        raise NotImplementedError
