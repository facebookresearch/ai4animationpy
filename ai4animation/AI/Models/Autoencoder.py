# Copyright (c) Meta Platforms, Inc. and affiliates.
import torch.nn as nn
from ai4animation.AI.Library import Defaults, Losses
from ai4animation.AI.Library.Blocks import LinearBlock
from ai4animation.AI.Library.Statistics import RunningStatistics


class Model(nn.Module):
    def __init__(
        self,
        feature_dim,
        embedding_dim,
        hidden_dim,
        dropout=Defaults.Dropout,
        activation=Defaults.Activation,
    ):
        super(Model, self).__init__()

        self.Statistics = RunningStatistics(feature_dim)

        self.Encoder = LinearBlock(
            feature_dim, embedding_dim, hidden_dim, dropout, activation
        )
        self.Decoder = LinearBlock(
            embedding_dim, feature_dim, hidden_dim, dropout, activation
        )

    def feature_dim(self):
        return self.Encoder.L1.InputSize

    def embedding_dim(self):
        return self.Decoder.L1.InputSize

    def forward(self, x):
        x = self.Statistics.Normalize(x)
        z = self.Encoder(x)
        z = self.Decoder(z)
        y = self.Statistics.Denormalize(z)
        return y

    def learn(self, features, update_statistics):
        if update_statistics:
            self.Statistics.Update(features)

        features = self.Statistics.Normalize(features)
        embedding = self.Encoder(features)
        prediction = self.Decoder(embedding)

        loss = Losses.MSE(prediction, features)

        reconstruction = self.Statistics.Denormalize(prediction)
        return {"Y": reconstruction, "Z": embedding}, {"MSE Loss": loss}
