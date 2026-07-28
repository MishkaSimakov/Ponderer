class Policy(object):
    """Stateless across episodes: a new episode means a new instance."""

    def act(self, obs):
        raise NotImplementedError


class ConstantPolicy(Policy):
    def __init__(self, action):
        self.action = tuple(action)

    def act(self, obs):
        return self.action
