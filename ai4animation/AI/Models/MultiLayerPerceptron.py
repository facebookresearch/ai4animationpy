# Copyright (c) Meta Platforms, Inc. and affiliates.
import torch.nn as nn
from ai4animation.AI.Library import Defaults, Losses
from ai4animation.AI.Library.Blocks import LinearBlock
from ai4animation.AI.Library.Statistics import RunningStatistics


class Model(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim,
        dropout=Defaults.Dropout,
        activation=Defaults.Activation,
    ):
        super(Model, self).__init__()

        self.InputStatistics = RunningStatistics(input_dim)
        self.OutputStatistics = RunningStatistics(output_dim)

        self.Layers = LinearBlock(
            input_dim, output_dim, hidden_dim, dropout, activation
        )

    def input_dim(self):
        return self.Layers.L1.InputSize

    def output_dim(self):
        return self.Layers.L3.OutputSize

    def forward(self, x):
        z = self.InputStatistics.Normalize(x)
        z = self.Layers(z)
        y = self.OutputStatistics.Denormalize(z)
        return y

    def learn(self, input, output, update_statistics):
        if update_statistics:
            self.InputStatistics.Update(input)
            self.OutputStatistics.Update(output)

        input = self.InputStatistics.Normalize(input)
        output = self.OutputStatistics.Normalize(output)
        prediction = self.Layers(input)

        loss = Losses.MSE(prediction, output)

        return {"Y": self.OutputStatistics.Denormalize(prediction)}, {"MSE Loss": loss}
