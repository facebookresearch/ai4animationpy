# Copyright (c) Meta Platforms, Inc. and affiliates.
import torch.nn as nn
from ai4animation.AI.Library import Defaults, Losses
from ai4animation.AI.Library.Blocks import LinearBlock
from ai4animation.AI.Library.Layers import CodebookLayer
from ai4animation.AI.Library.Statistics import RunningStatistics


class Model(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        encoder_dim,
        decoder_dim,
        channels,
        classes,
        dropout=Defaults.Dropout,
        activation=Defaults.Activation,
    ):
        super(Model, self).__init__()

        self.InputStatistics = RunningStatistics(input_dim)
        self.OutputStatistics = RunningStatistics(output_dim)

        self.Codebook = CodebookLayer(channels, classes)
        self.Encoder = LinearBlock(
            input_dim, self.Codebook.dimensions(), encoder_dim, dropout, activation
        )
        self.Decoder = LinearBlock(
            self.Codebook.dimensions(), output_dim, decoder_dim, dropout, activation
        )

    def input_dim(self):
        return self.Encoder.L1.InputSize

    def output_dim(self):
        return self.Decoder.L3.OutputSize

    def forward(self, x, sample=False):
        x = self.InputStatistics.Normalize(x)
        z = self.Encoder(x)
        z = self.Codebook(z, sample)
        y = self.Decoder(z)
        y = self.OutputStatistics.Denormalize(y)
        return y

    def learn(self, inputs, outputs, update_statistics):
        if update_statistics:
            self.InputStatistics.Update(inputs)
            self.OutputStatistics.Update(outputs)

        inputs = self.InputStatistics.Normalize(inputs)
        outputs = self.OutputStatistics.Normalize(outputs)

        x = self.Encoder(inputs)
        z = self.Codebook(x, True)
        y = self.Decoder(z)

        loss = Losses.MSE(y, outputs)

        return {"Y": self.OutputStatistics.Denormalize(y), "Z": z}, {"MSE Loss": loss}
