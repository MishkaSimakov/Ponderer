"""Causal transformer over a window of frames.

A window rather than a recurrent state: VecFrameStack already hands the last K frames
over flattened, oldest first, and on the brick the same thing is a ring buffer. Nothing
recurrent has to survive the export, and the cost of a step is constant.
"""

import torch as th
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class CausalWindow(BaseFeaturesExtractor):
    def __init__(self, observation_space, window, model_dim=32, heads=4, layers=1):
        super().__init__(observation_space, model_dim)

        stacked = observation_space.shape[0]
        if stacked % window != 0:
            raise ValueError("window %d does not divide observation %d" % (window, stacked))

        self.window = window
        self.frame = stacked // window
        self.project = nn.Linear(self.frame, model_dim)
        self.position = nn.Parameter(th.zeros(window, model_dim))
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(model_dim, heads, 4 * model_dim,
                                       dropout=0.0, batch_first=True),
            layers)
        self.register_buffer("mask", nn.Transformer.generate_square_subsequent_mask(window))

    def forward(self, observations):
        x = observations.view(-1, self.window, self.frame)
        x = self.project(x) + self.position
        # The newest frame is last, so the last token is the one to act on.
        return self.encoder(x, mask=self.mask, is_causal=True)[:, -1]
