import torch
import torch.nn as nn
from ai4animation.AI import Modules, Stats


class Model(nn.Module):
    def __init__(self, feature_dim, hidden_dim, latent_dim, dropout, manifold=None):
        super(Model, self).__init__()

        self.Stats = Stats.RunningStats(feature_dim)

        self.FDim = feature_dim
        self.HDim = hidden_dim
        self.LDim = latent_dim

        self.Encoder = Modules.LinearEncoder(
            feature_dim, hidden_dim, latent_dim, dropout
        )
        self.Manifold = manifold
        self.Decoder = Modules.LinearEncoder(
            latent_dim, hidden_dim, feature_dim, dropout
        )

    def forward(self, x, return_latent=False):
        z = self.encode(x)
        y = self.decode(z)
        if return_latent:
            return y, z
        else:
            return y

    def encode(self, x):
        x = self.Stats.Normalize(x)
        z = self.Encoder(x)
        if self.Manifold is not None:
            z = self.Manifold(z)
        return z

    def decode(self, z):
        y = self.Decoder(z)
        y = self.Stats.Denormalize(y)
        return y

    def learn(self, features, update_stats):
        features = (
            self.Stats.UpdateAndNormalize(features)
            if update_stats
            else self.Stats.Normalize(features)
        )

        z = self.Encoder(features)
        y = self.Decoder(z)

        mse_fn = nn.MSELoss()
        pred = self.Stats.Denormalize(y)
        loss = mse_fn(y, features)
        return {"Y": pred, "Z": z}, {"MSE": loss}
