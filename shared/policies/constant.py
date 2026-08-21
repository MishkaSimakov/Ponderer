from shared.policy import Policy


class ConstantPolicy(Policy):
    def __init__(self, action=(0.6, 0.6)):
        self.action = tuple(action)

    def act(self, obs):
        return self.action
