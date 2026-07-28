class Policy(object):
    """Stateless across episodes: a new episode means a new instance."""

    def act(self, obs):
        raise NotImplementedError
