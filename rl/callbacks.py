from collections import deque

from stable_baselines3.common.callbacks import BaseCallback


class OutcomeCallback(BaseCallback):
    """Share of finished episodes that terminated rather than hit the step limit.

    SB3 logs returns and lengths but not which of the two ways an episode ended, and
    for line following that is the whole question: left the line, or survived.
    """

    def __init__(self, window=100):
        super().__init__()
        self.outcomes = deque(maxlen=window)

    def _on_step(self):
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done:
                self.outcomes.append(0.0 if info.get("TimeLimit.truncated") else 1.0)

        if self.outcomes:
            self.logger.record("rollout/terminated_frac",
                               sum(self.outcomes) / len(self.outcomes))
        return True
