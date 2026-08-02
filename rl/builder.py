"""One architecture switch, three ready made trainers."""

from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack, VecMonitor

from rl.extractors import CausalWindow
from rl.vec_sim import SimVecEnv

ARCHS = ("mlp", "lstm", "transformer")


def build(arch, port=5005, seed=0, hidden=32, window=8, randomize_scenario=True,
          tensorboard_log=None, **ppo_kwargs):
    # VecMonitor is what turns dones into the episode returns and lengths that end up
    # in tensorboard; unity's own step counter is zeroed by the auto reset.
    env = VecMonitor(SimVecEnv(port=port, seed=seed, randomize_scenario=randomize_scenario))
    common = dict(seed=seed, tensorboard_log=tensorboard_log, verbose=1, **ppo_kwargs)

    if arch == "mlp":
        return PPO("MlpPolicy", env,
                   policy_kwargs=dict(net_arch=[hidden, hidden]), **common)

    if arch == "lstm":
        return RecurrentPPO("MlpLstmPolicy", env,
                            policy_kwargs=dict(lstm_hidden_size=hidden, n_lstm_layers=1,
                                               net_arch=[hidden]), **common)

    if arch == "transformer":
        return PPO("MlpPolicy", VecFrameStack(env, window),
                   policy_kwargs=dict(net_arch=[hidden],
                                      features_extractor_class=CausalWindow,
                                      features_extractor_kwargs=dict(window=window,
                                                                     model_dim=hidden)),
                   **common)

    raise ValueError("unknown arch %r, expected one of %s" % (arch, ARCHS))
