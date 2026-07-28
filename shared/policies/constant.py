from shared.policy import Policy


class ConstantPolicy(Policy):
    def __init__(self, action):
        self.action = tuple(action)

    def act(self, obs):
        return self.action